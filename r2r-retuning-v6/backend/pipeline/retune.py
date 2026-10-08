"""Algorithm 1, Steps 4-7, run live: K_p* and T_I from the identified twin.

The gain atlas answers only for protocols on its grid. This module answers for
any RunSpec -- a hand-edited plant, a drift, an off-grid acquisition -- by doing
what the paper's field engineer does after drift is suspected:

  Step 4  identify theta-hat under the protocol          (pipeline.identify)
  Step 5  build the twin and check it on a validation step response
  Step 6  retune on the twin: hierarchical grid search, optionally a few-shot
          BO on the plant seeded from the twin's optimum
  Step 7  score the delivered gains on the plant against C_target

Everything the paper leaves open is a named, documented choice here, and every
one of them is listed in `ASSUMPTIONS` and returned with the result. The
reasoning and the literature behind each is in
docs/retuning-assumptions/ASSUMPTIONS-REPORT.md.

Nothing is invented: a failed identification returns no gains at all.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from backend.models.controller import ControllerConfig
from backend.models.equations import R2RParameters
from backend.models.simulation import SimulationConfig, simulate
from backend.validation import retuning_paper_protocol as AP
from backend.validation.retuning import plant_auto_ti_s
from backend.validation.retuning_paper_eval import (
    PAPER_EVAL_KEY,
    STEP_FRACTION,
    T_SIM_S,
    T_STEP_S,
    PaperEvaluator,
)
from backend.validation.retuning_tiered_eval import TIERS as TIER_NAMES, TieredEvaluator
from backend.validation.retuning_structures import BY_NAME as STRUCTURE_BY_NAME, BASELINE as SHARED_STRUCTURE
from backend.validation.retuning_structure_search import search as structure_search

from . import cache, zone_gains
from .identify import identify
from .payloads import DriftSpec, RunSpec, run_hash
from .plant import drifted_params

# Bump whenever the way a retune is COMPUTED changes (see identify.py for why a
# content-addressed cache needs a computation version as well).
# v2 (2026-09-15): the cost experiment, box and few-shot BO follow the authors'
# reply (`../FE sample/Q1a.pdf`) instead of the dashboard's reconstruction.
# v3 (2026-09-16): with the T_I tie-break off, the twin optimum is delivered
# exactly as searched, so a live retune is comparable with the offline campaign.
# v5 (2026-09-29): the scoring cost is selectable per run (RetuneOptions.cost_tier).
# 2026-10-07: the DEFAULT IS NOW "T1", the repaired cost (user's decision, after the
# 60-cell grid). T0, the authors' Eq. (12), stays selectable and bit-identical.
# The cache key carries the tier, so T0 results already cached stay valid.
RETUNE_VERSION = "v5"
CACHE_KIND = f"retune-{RETUNE_VERSION}"

# Scoring cost. "T0" is the paper's Eq. (12) exactly; "T1" repairs its three
# defects (absolute error scale across plants whose setpoints span 12-360 N, a
# discontinuous settling term, and three different span aggregators in one
# scalar); "T2" adds actuation realism; "T3" adds the full roll. See
# backend/validation/retuning_tiered_eval.py.
DEFAULT_COST_TIER = "T1"

# Gain structure. "shared-2D" is the paper: one (K_p*, T_I) for all three tension
# zones, searched by the authors' HGS -- the DEFAULT, so nothing moves. The others
# give zones their own gains and are searched by a dimension-generic routine:
#   outfeeder-only-2D  freeze UW and RW at the shared optimum, tune the nip
#   ends-only-2D       freeze the nip, tune UW and RW together
#   ends-tied-4D       UW and RW share a pair (identical machines), nip its own
#   full-6D            a free pair per zone
# The in-feeder is the velocity master and has no tension gain.
# 2026-10-07 (user): TWO structures are used. Stage 1 "full-6D" finds a free pair
# per zone and SAVES the six gains for the plant (pipeline.zone_gains); stage 2
# "outfeeder-only-2D" -- the default -- freezes UW and RW at the saved gains and
# runs the authors' 2-D HGS on the out-feeder only. With nothing saved yet, stage 2
# runs stage 1 for this case first. (The older structures stay callable.)
PER_ZONE_STRUCTURE = "full-6D"
OUTFEEDER_STRUCTURE = "outfeeder-only-2D"
DEFAULT_GAIN_STRUCTURE = OUTFEEDER_STRUCTURE

# The method adopted on 2026-10-06 (multiloop/reports/SEQUENTIAL-HGS.pdf): the
# authors' 2-D HGS run on one zone at a time -- UW, RW, out-feeder -- starting
# from the commissioned gains, two sweeps. No 6-D search, no real-plant trials.
# Matched or beat the 6-D per-zone search in 57/60 cells. UW and RW end up FIXED
# at what HGS found for them; the out-feeder is then already at its optimum.
SEQUENTIAL_STRUCTURE = "sequential-2D"
SEQUENTIAL_ORDER = (("UW", 0), ("RW", 2), ("OutFeeder", 1))
SEQUENTIAL_SWEEPS = 2
GAIN_STRUCTURES = (*STRUCTURE_BY_NAME, SEQUENTIAL_STRUCTURE)

# Eq. (12) on the authors' experiment: 30 s, all three references stepped +20 %
# at 5 s, metrics over the 25 s after the step, no clamp / noise / filter.
EVAL_MODEL_KEY = PAPER_EVAL_KEY
HGS_BUDGET = sum((AP.HGS_STAGES["coarse_side"] ** 2, AP.HGS_STAGES["lhs"],
                  AP.HGS_STAGES["fine_side"] ** 2, AP.HGS_STAGES["polish"]))
TI_PROFILE_POINTS = 121

# Assumption A6: ties in a flat cost valley are broken toward the smaller T_I.
#
# OFF by default since 2026-09-16. It was introduced when T_I was searched as a
# multiple of the plant's own heuristic over [0.02, 100] -- up to thousands of
# seconds, where the cost really is flat and the optimum pinned at the box edge.
# The authors' box is absolute [0.5, 30] s, so the ceiling already bounds the
# integral action, and the tie-break can deliver gains that cost MORE on the
# plant than the twin optimum it replaces (P01, drift D06: 0.1770 vs 0.1734).
# The paper's HGS-only arm applies the twin optimum directly, so that is what
# the dashboard delivers. Set `ti_flat_tolerance` > 0 to re-enable it.
DEFAULT_TI_FLAT_TOLERANCE = 0.0

ASSUMPTIONS: tuple[dict[str, str], ...] = (
    {"id": "A1", "what": "Gain search box", "source": "authors' reply (Q1b)",
     "value": f"K_p* in [{AP.KP_BOUNDS[0]:g}, {AP.KP_BOUNDS[1]:g}] 1/s log-uniform; T_I in [{AP.TI_BOUNDS_S[0]:g}, {AP.TI_BOUNDS_S[1]:g}] s absolute, uniform",
     "why": "Not printed in the paper; given by the authors. They note the 30 s T_I ceiling truncates the search (59 of 60 twin optima sit on it)."},
    {"id": "A2", "what": "T_I units", "source": "authors' reply (Q1b)",
     "value": "absolute seconds, the same box for twin and plant",
     "why": "No per-plant anchor: the box is absolute, so nothing rescales on transfer."},
    {"id": "A3", "what": "Cost experiment (Eq. 12)", "source": "authors' reply (Q1a, D4, E5)",
     "value": f"RK4 dt 1 ms, {T_SIM_S:g} s; all three references +{100*STEP_FRACTION:g} % at {T_STEP_S:g} s; metrics over the 25 s after the step on true tensions; no clamp, noise or filter; feed-forward friction on measured speed, speed reference held post-step",
     "why": "Not printed in the paper; described by the authors, who also confirmed an independent transcription matches their costs to a median ratio of 1.0015."},
    {"id": "A4", "what": "Stage split of the 2,805 twin evaluations", "source": "dashboard choice",
     "value": "30x30 coarse (K_p* log, T_I linear) / 1,000 Latin hypercube / 30x30 fine around the incumbent / 5-point local polish",
     "why": "S8.1 names the stages; neither the paper nor the reply gives their sizes or spacing."},
    {"id": "A5", "what": "Few-shot BO on the plant (HGS+BO(5))", "source": "authors' reply (Q1b, Q3)",
     "value": "gp_minimize, EI, noise 1e-6; x0 = the first 2 points of the +-30 % star around the twin optimum with their costs as y0; 3 random points; 0 EI steps",
     "why": "This is how the paper's arm ran. The authors state it takes no model-guided step."},
    {"id": "A6", "what": "T_I in a flat cost valley", "source": "dashboard choice (off by default)",
     "value": "disabled: the twin optimum is delivered as searched, as the paper's HGS-only arm does",
     "why": "The authors' box caps T_I at 30 s, so the flat-valley problem this addressed does not arise; with tolerance > 0 the tie-break can raise the cost on the plant."},
    {"id": "A7", "what": "Step 5 twin validation and epsilon", "source": "dashboard choice",
     "value": "prediction RMSE of twin vs plant on the A3 step experiment at the commissioned gains; epsilon = the same for the pre-drift twin",
     "why": "Sec. 5.1 reads epsilon off commissioning but defines RMSE_y as tracking error, which cannot validate a twin."},
    {"id": "A8", "what": "Step 7 C_target", "source": "paper Sec. 5.1 wording",
     "value": "S of the commissioned gains on the pre-drift plant, same experiment; margins default 1.0",
     "why": "'C_target as the retuning cost (12) recorded there'; no slack factor is printed."},
    {"id": "A9", "what": "Step 7 restore rule", "source": "dashboard choice",
     "value": "deliver the twin's gains only if their S on the line is strictly below the commissioned gains' S on the same drifted line; otherwise restore the commissioned gains. C_target is still checked and reported on what is delivered",
     "why": "Restoring on the C_target test alone would undo a retune that helps but cannot reach the pre-drift cost; never restoring ships gains that make the line worse when theta-hat is badly wrong (P10, MARE ~280 %)."},
)


@dataclass(frozen=True)
class RetuneOptions:
    bo_refine: bool = False
    epsilon_margin: float = 1.0
    c_target_margin: float = 1.0
    ti_flat_tolerance: float = DEFAULT_TI_FLAT_TOLERANCE
    cost_tier: str = DEFAULT_COST_TIER
    gain_structure: str = DEFAULT_GAIN_STRUCTURE

    def __post_init__(self) -> None:
        for name in ("epsilon_margin", "c_target_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        tol = float(self.ti_flat_tolerance)
        if not math.isfinite(tol) or not 0.0 <= tol <= 0.5:
            raise ValueError("ti_flat_tolerance must be in [0, 0.5]")
        if self.cost_tier not in TIER_NAMES:
            raise ValueError(f"cost_tier must be one of {TIER_NAMES}, got {self.cost_tier!r}")
        if self.gain_structure not in GAIN_STRUCTURES:
            raise ValueError(f"gain_structure must be one of "
                             f"{GAIN_STRUCTURES}, got {self.gain_structure!r}")


# --------------------------------------------------------------------------- #
# pure pieces
# --------------------------------------------------------------------------- #
def twin_params_from_estimates(
    physical: R2RParameters, estimates: Mapping[str, float]
) -> R2RParameters:
    """Invert the Eq. (6) ratios into a plant: J = R^2 / k_t, f = k_f * J.

    Geometry, span lengths and set-points are known from commissioning and are
    not identified; the same inversion `retuning.identify_twin` performs.
    """

    radii = tuple(float(r) for r in physical.roller_radius_m)
    kt = [float(estimates[f"kt_{name}"]) for name in ("UW", "Nip", "RW")]
    kf = [float(estimates[f"kf_{name}"]) for name in ("UW", "Nip", "RW")]
    inertia = tuple(
        r * r / k if k > 0 else j for r, k, j in zip(radii, kt, physical.inertia_kg_m2)
    )
    friction = [k * j for k, j in zip(kf, inertia)]
    return replace(
        physical,
        EA=float(estimates["EA"]),
        inertia_kg_m2=inertia,
        kf_UW=friction[0],
        kf_Nip=friction[1],
        kf_RW=friction[2],
    )


def simc_integral_time_s(params: R2RParameters, line_speed_m_s: float, kp_star: float) -> float:
    """Skogestad's SIMC integral time for the span model, as a REFERENCE only.

    Linearised span tension is first order with tau_1 = L / v0 as seen by K_p*,
    so SIMC's tau_I = min(tau_1, 4 (tau_c + theta)) with K_p* = 1 / (tau_c + theta)
    gives T_I = min(L / v0, 4 / K_p*). The shortest span is the fastest.
    Skogestad (2003), J. Process Control 13(4):291-309.
    """

    tau_1 = min(float(length) for length in params.span_length_m) / float(line_speed_m_s)
    return min(tau_1, 4.0 / float(kp_star))


def resolve_integral_time(
    scales: Sequence[float], costs: Sequence[float], tolerance: float
) -> dict[str, Any]:
    """Break a flat T_I valley toward the smallest T_I (assumption A6).

    Only the CONTIGUOUS run of grid points around the best cost counts: a
    disconnected dip at small T_I is a different basin, not the same flat
    valley, and jumping to it would change the controller's behaviour class.
    """

    values = [float(c) if math.isfinite(float(c)) else math.inf for c in costs]
    if not values or all(math.isinf(v) for v in values):
        raise ValueError("no finite cost on the T_I profile")
    best_index = min(range(len(values)), key=values.__getitem__)
    best = values[best_index]
    threshold = best * (1.0 + float(tolerance)) if best > 0 else best + float(tolerance)
    lo = best_index
    while lo - 1 >= 0 and values[lo - 1] <= threshold:
        lo -= 1
    hi = best_index
    while hi + 1 < len(values) and values[hi + 1] <= threshold:
        hi += 1
    return {
        "best_index": best_index,
        "chosen_index": lo,
        "best_scale": float(scales[best_index]),
        "chosen_scale": float(scales[lo]),
        "valley_scales": [float(scales[lo]), float(scales[hi])],
        "threshold": threshold,
        "tolerance": float(tolerance),
    }


def prediction_fit(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE (N) and NRMSE fit (%) of a predicted three-channel response.

    fit = 100 (1 - ||y - y_hat|| / ||y - mean(y)||), per channel, then averaged
    -- the System Identification Toolbox's `compare` convention.
    """

    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - reference
    rmse = float(np.mean(np.sqrt(np.mean(error ** 2, axis=0))))
    fits = []
    for ch in range(reference.shape[1]):
        spread = np.linalg.norm(reference[:, ch] - reference[:, ch].mean())
        if spread > 0:
            fits.append(100.0 * (1.0 - np.linalg.norm(error[:, ch]) / spread))
    return {"rmse_N": rmse, "fit_percent": float(np.mean(fits)) if fits else math.nan}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
class _Scorer:
    """The retuning cost for one plant; T_I in seconds.

    `tier="T0"` is the authors' Eq. (12) and routes to `PaperEvaluator`, i.e. the
    exact object this class has always used -- so a T0 scorer is byte-for-byte what
    it was before tiers existed. T1-T3 route to the tiered evaluator, which takes
    per-zone gains; a scalar broadcasts to all three tension zones.
    """

    def __init__(self, params: R2RParameters, line_speed_m_s: float,
                 tier: str = DEFAULT_COST_TIER):
        if tier not in TIER_NAMES:
            raise ValueError(f"tier must be one of {TIER_NAMES}, got {tier!r}")
        self.tier = tier
        self._params, self._v0 = params, line_speed_m_s
        self._vec: TieredEvaluator | None = None
        self.evaluator = (PaperEvaluator(params, line_speed_m_s) if tier == "T0"
                          else TieredEvaluator(params, line_speed_m_s, tier=tier))
        if tier != "T0":
            self._vec = self.evaluator

    @property
    def vec(self) -> TieredEvaluator:
        """The per-zone evaluator, built on demand.

        At T0 the scalar path deliberately keeps using `PaperEvaluator`, so a T0
        run is byte-identical to the pre-tier pipeline. Per-zone gains need the
        tiered evaluator even at T0 -- where it delegates to the kernel proven
        equal to `PaperEvaluator` at 0 ULP, so the two agree by construction.
        """
        if self._vec is None:
            self._vec = TieredEvaluator(self._params, self._v0, tier=self.tier)
        return self._vec

    def batch_vec(self, kp: np.ndarray, ti: np.ndarray) -> np.ndarray:
        """`kp[n, 3]`, `ti[n, 3]` -> `cost[n]`, one gain pair per tension zone."""
        return self.vec.costs(kp, ti)

    def breakdown_vec(self, kp, ti) -> dict[str, float | bool | None]:
        b = self.vec.breakdown(kp, ti)
        return {**b, "finite": b["S"] is not None}

    @staticmethod
    def _spread(values) -> np.ndarray:
        """Scalars (or a 1-D list of them) -> the (n, 3) the tiered kernel wants."""
        a = np.asarray(values, dtype=float)
        if a.ndim == 0:
            a = a.reshape(1)
        return np.repeat(a.reshape(-1, 1), 3, axis=1)

    def batch(self, kps: Sequence[float], tis: Sequence[float]) -> np.ndarray:
        if self.tier == "T0":
            return self.evaluator.costs(kps, tis)
        return self.evaluator.costs(self._spread(kps), self._spread(tis))

    def scalar(self, kp: float, ti: float) -> float:
        if self.tier == "T0":
            return self.evaluator.cost(kp, ti)
        return float(self.evaluator.costs(self._spread(kp), self._spread(ti))[0])

    def breakdown(self, kp: float, ti_s: float) -> dict[str, float | bool | None]:
        b = (self.evaluator.breakdown(kp, ti_s) if self.tier == "T0"
             else self.evaluator.breakdown(self._spread(kp)[0], self._spread(ti_s)[0]))
        return {**b, "finite": b["S"] is not None}


def _validation_response(
    params: R2RParameters, line_speed_m_s: float, kp_star: float, ti_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """True tensions after the A3 step (all channels +20 % at 5 s), 30 s record."""

    base = np.asarray(params.tension_ref_N, dtype=float)
    step = STEP_FRACTION * base

    def profile(t_s: float) -> tuple[float, float, float]:
        return tuple(float(v) for v in step) if t_s >= T_STEP_S else (0.0, 0.0, 0.0)  # type: ignore[return-value]

    config = SimulationConfig(
        duration_s=T_SIM_S,
        dt_s=0.001,
        controller_sample_time_s=0.001,
        log_sample_time_s=0.005,
        line_speed_m_s=line_speed_m_s,
        sensor_lpf_hz=None,
        controller_tracks_drift=False,
    )
    controller = ControllerConfig(
        target_tension_N=tuple(base),
        line_speed_m_s=line_speed_m_s,
        Kp_star_m_s_per_N=float(kp_star),
        TI_s=float(ti_s),
        paper_velocity_gain_enabled=True,
        high_ea_kp_cap_enabled=False,
        velocity_correction_limit_fraction=None,
    )
    result = simulate(params=params, controller_config=controller, config=config,
                      excitation=profile, write_output=False)
    times = np.array([float(r["time_s"]) for r in result.rows])
    tensions = np.array([[float(r["T1"]), float(r["T2"]), float(r["T3"])] for r in result.rows])
    keep = times > T_STEP_S
    return times[keep], tensions[keep]


ZONES = ("UW", "OutFeeder", "RW")


def _fmt3(v) -> str:
    """[1.0, 1.0, 1.0] -> '1.000'; differing zones -> '[a, b, c]'."""
    vals = [float(x) for x in v]
    return f"{vals[0]:.3f}" if len(set(vals)) == 1 else "[" + ", ".join(f"{x:.3f}" for x in vals) + "]"


def _three(v) -> list[float]:
    """Scalar -> one value per tension zone; a 3-vector passes through."""
    a = np.asarray(v, dtype=float).reshape(-1)
    return [float(a[0])] * 3 if a.size == 1 else [float(x) for x in a]


def _gains_block(kp, ti_s, **extra: Any) -> dict[str, Any]:
    """One gain pair per tension zone (UW, out-feeder, RW).

    `kp_star` / `ti_s` stay SCALARS whenever all three zones share a value, so
    every existing consumer and every stored payload keeps its shape. When the
    zones differ they are None and the per-zone lists are the only truth -- a
    consumer that assumed a scalar then fails loudly instead of quietly using one
    zone's gain for all three.
    """
    kps, tis = _three(kp), _three(ti_s)
    kp_uniform = len(set(kps)) == 1
    ti_uniform = len(set(tis)) == 1
    return {
        "kp_star": kps[0] if kp_uniform else None,
        "ti_s": tis[0] if ti_uniform else None,
        "kp_star_per_zone": kps,
        "ti_s_per_zone": tis,
        "zones": list(ZONES),
        "per_zone": not (kp_uniform and ti_uniform),
        "kp_on_bound": [AP._bound(k, AP.KP_BOUNDS) for k in kps],
        "ti_on_bound": [AP._bound(t, AP.TI_BOUNDS_S) for t in tis],
        **extra,
    }


def _hgs_one_zone(scorer: _Scorer, kp3: np.ndarray, ti3: np.ndarray, zone: int):
    """The authors' HGS on one zone's (K_p*, T_I); the other two zones held fixed."""

    def batch(kps, tis):
        n = len(kps)
        kp = np.tile(kp3, (n, 1)); ti = np.tile(ti3, (n, 1))
        kp[:, zone] = np.asarray(kps, float); ti[:, zone] = np.asarray(tis, float)
        return scorer.batch_vec(kp, ti)

    h = AP.hierarchical_grid_search(batch)
    # HGS does not seed its grid with the incumbent, so it can land on a grid
    # point marginally WORSE than where this zone already was. Keep whichever is
    # better: the stage then never raises the twin cost. (Seen live: the final
    # stage going 0.15203 -> 0.15302 before this guard.)
    current = float(scorer.batch_vec(kp3[None, :], ti3[None, :])[0])
    if not (float(h.cost) < current):
        return kp3.copy(), ti3.copy(), current, int(h.evaluations)
    kp, ti = kp3.copy(), ti3.copy()
    kp[zone], ti[zone] = float(h.kp), float(h.ti)
    return kp, ti, float(h.cost), int(h.evaluations)


def _sequential_hgs(scorer: _Scorer, start_kp: float, start_ti: float):
    """One zone at a time, SEQUENTIAL_SWEEPS sweeps, from the commissioned pair."""

    kp, ti = np.full(3, float(start_kp)), np.full(3, float(start_ti))
    stages, total = [], 0
    for sweep in range(1, SEQUENTIAL_SWEEPS + 1):
        for name, z in SEQUENTIAL_ORDER:
            kp, ti, s_tw, n = _hgs_one_zone(scorer, kp, ti, z); total += n
            stages.append({"sweep": sweep, "zone": name,
                           "kp_star_per_zone": kp.tolist(), "ti_s_per_zone": ti.tolist(),
                           "S_twin": s_tw, "twin_evals": n})
    return kp, ti, stages[-1]["S_twin"], total, stages


def _bo_refine(scorer: _Scorer, kp: float, ti_s: float) -> "AP.BORun":
    """Assumption A5: the paper's HGS+BO(5), exactly as the authors ran it."""

    return AP.run_bo(scorer.scalar, budget=5, seed=0, warm_start=AP.star(kp, ti_s))


def _retune_key(run: RunSpec, options: RetuneOptions, frozen: Mapping[str, Any] | None = None) -> str:
    """`frozen`: the saved UW/RW gains a two-stage retune freezes -- part of the answer,
    so part of the key. Absent (None) keeps every older key unchanged."""
    content = {"run": run_hash(run), "options": asdict(options), "v": RETUNE_VERSION}
    if frozen is not None:
        content["frozen"] = {"kp": [float(v) for v in frozen["kp"]], "ti": [float(v) for v in frozen["ti"]]}
    blob = json.dumps(content, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# the protocol
# --------------------------------------------------------------------------- #
def retune(run: RunSpec, options: RetuneOptions | None = None, *, use_cache: bool = True) -> dict[str, Any]:
    """Run Algorithm 1 Steps 4-7 for one RunSpec."""

    options = options or RetuneOptions()
    stored = (zone_gains.load(run, options.cost_tier)
              if options.gain_structure == OUTFEEDER_STRUCTURE else None)
    frozen = ({"kp": stored["kp_star_per_zone"], "ti": stored["ti_s_per_zone"]} if stored else None)
    key = _retune_key(run, options, frozen)
    if use_cache:
        hit = cache.load(CACHE_KIND, key)
        if hit is not None:
            pz = (hit.get("search") or {}).get("per_zone_gains")
            if hit.get("status") == "ok" and pz and pz.get("saved") and not pz.get("reused"):
                # re-commit: the store may have been forgotten since this was cached
                _save_per_zone(run, options, pz)
            return apply_restore({**hit, "cached": True})

    started = time.time()
    base, drifted, meta = drifted_params(run.plant, run.drift)
    v0 = float(meta["v_ref_m_s"])
    steps: list[dict[str, Any]] = []

    # -- Step 4: identify the drifted plant -------------------------------- #
    try:
        ident = identify(run)
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        ident = {"converged": False, "failure": f"{type(exc).__name__}: {exc}"}
    steps.append({
        "step": 4, "name": "Run system identification",
        "status": "pass" if ident.get("converged") else "fail",
        "detail": (f"MARE_theta = {ident['MARE_theta_pct']:.2f} %" if ident.get("converged")
                   else ident.get("failure") or "the estimator did not converge"),
    })
    common = {
        "retune_version": RETUNE_VERSION,
        "retune_key": key,
        "run_hash": run_hash(run),
        "plant_id": meta["plant_id"],
        "drift_applied": not run.drift.is_identity,
        "options": asdict(options),
        "assumptions": list(ASSUMPTIONS),
        "eval_model": EVAL_MODEL_KEY if options.cost_tier == "T0" else
                      f"{EVAL_MODEL_KEY}+{options.cost_tier}",
        "cost_tier": options.cost_tier,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if not ident.get("converged"):
        payload = {
            **common, "status": "identification_failed", "steps": steps,
            "identification": {"converged": False, "failure": ident.get("failure")},
            "gains": None,
            "note": "No gains: a failed identification is no twin, and no twin is no retune.",
            "seconds": time.time() - started,
        }
        cache.store(CACHE_KIND, key, payload)
        return {**payload, "cached": False}

    twin = twin_params_from_estimates(drifted, ident["estimates"])

    # -- commissioning baseline: the pre-drift line and its gains ---------- #
    base_scorer = _Scorer(base, v0, options.cost_tier)
    commissioning = AP.hierarchical_grid_search(base_scorer.batch)
    commissioned = _gains_block(commissioning.kp, commissioning.ti)
    c_target = float(commissioning.cost)

    # -- Step 6: retune on the twin ---------------------------------------- #
    twin_scorer = _Scorer(twin, v0, options.cost_tier)
    structure = STRUCTURE_BY_NAME.get(options.gain_structure)
    seq_stages = None
    per_zone = outfeeder_meta = None
    if options.gain_structure == SEQUENTIAL_STRUCTURE:
        # start from the commissioned pair -- what the line is running when drift
        # is suspected -- then the authors' HGS on each zone in turn, twice
        hgs_kp, hgs_ti, hgs_cost, n_ev, seq_stages = _sequential_hgs(
            twin_scorer, commissioning.kp, commissioning.ti)
        search_meta = {"search": f"sequential HGS (authors' 2-D, one zone at a time, "
                                 f"{SEQUENTIAL_SWEEPS} sweeps)",
                       "evaluations": n_ev,
                       "stage_counts": {f"sweep{s['sweep']}-{s['zone']}": s["twin_evals"]
                                        for s in seq_stages},
                       "ends_fixed_at": "what HGS found for UW and RW; the out-feeder is "
                                        "then already at its optimum"}
    elif options.gain_structure in (PER_ZONE_STRUCTURE, OUTFEEDER_STRUCTURE):
        # Stage 1: the six gains -- searched here, or reused from the store.
        if stored is None:
            six = structure_search(twin_scorer.batch_vec, STRUCTURE_BY_NAME[PER_ZONE_STRUCTURE],
                                   baseline=None, seed=0)
            per_zone = {"kp_star_per_zone": [float(v) for v in six.kp],
                        "ti_s_per_zone": [float(v) for v in six.ti],
                        "S_twin": float(six.cost), "twin_evals": int(six.evaluations),
                        "search": "structure search (full-6D)", "reused": False,
                        "source_run_hash": run_hash(run)}
        else:
            per_zone = {"kp_star_per_zone": list(stored["kp_star_per_zone"]),
                        "ti_s_per_zone": list(stored["ti_s_per_zone"]),
                        "S_twin": stored.get("S_twin"), "twin_evals": 0,
                        "search": "saved six gains", "reused": True,
                        "source_run_hash": stored.get("source_run_hash"),
                        "saved_at": stored.get("saved_at")}
        # saved (or not) at the end, once step 7 has judged the case: _commit_per_zone
        kp6 = np.asarray(per_zone["kp_star_per_zone"], dtype=float)
        ti6 = np.asarray(per_zone["ti_s_per_zone"], dtype=float)
        if options.gain_structure == PER_ZONE_STRUCTURE:
            hgs_kp, hgs_ti = kp6, ti6
            hgs_cost = float(twin_scorer.batch_vec(kp6[None, :], ti6[None, :])[0])
            search_meta = {"search": "structure search (full-6D)", "evaluations": per_zone["twin_evals"],
                           "stage_counts": {"full-6D": per_zone["twin_evals"]}}
            outfeeder_meta = None
        else:
            # Stage 2: UW and RW frozen at the six gains; the authors' HGS on the out-feeder.
            hgs_kp, hgs_ti, hgs_cost, n_nip = _hgs_one_zone(twin_scorer, kp6, ti6, zone=1)
            outfeeder_meta = {"search": "HGS (authors')", "evaluations": n_nip,
                              "kp_star": float(hgs_kp[1]), "ti_s": float(hgs_ti[1]), "S_twin": hgs_cost,
                              "kept_saved_value": bool(hgs_kp[1] == kp6[1] and hgs_ti[1] == ti6[1])}
            search_meta = {"search": "two-stage: six gains, then the out-feeder by HGS (authors')",
                           "evaluations": per_zone["twin_evals"] + n_nip,
                           "stage_counts": {"full-6D": per_zone["twin_evals"], "outfeeder-HGS": n_nip},
                           "ends_fixed_at": "the saved six gains (UW, RW)"}
    elif options.gain_structure == SHARED_STRUCTURE:
        # the authors' own HGS, untouched
        hgs = AP.hierarchical_grid_search(twin_scorer.batch)
        hgs_kp, hgs_ti, hgs_cost = hgs.kp, hgs.ti, float(hgs.cost)
        search_meta = {"search": "HGS (authors')", "evaluations": hgs.evaluations,
                       "stage_counts": dict(hgs.stage_counts)}
    else:
        # A restricted structure freezes the other zones AT the shared optimum, so
        # it needs that optimum first -- found on the same twin, same cost tier.
        baseline = None
        if structure.needs_baseline:
            b = AP.hierarchical_grid_search(twin_scorer.batch)
            baseline = (np.full(3, b.kp), np.full(3, b.ti))
        res = structure_search(twin_scorer.batch_vec, structure, baseline=baseline, seed=0)
        hgs_kp, hgs_ti, hgs_cost = res.kp, res.ti, float(res.cost)
        search_meta = {"search": f"structure search ({structure.name}, {structure.dim}-D)",
                       "evaluations": res.evaluations,
                       "stage_counts": dict(res.stage_counts),
                       "froze_others_at_shared_optimum": structure.needs_baseline}
    hgs_only = _gains_block(hgs_kp, hgs_ti, S_twin=hgs_cost,
                            gain_structure=options.gain_structure, **search_meta)

    shared_gains = not hgs_only["per_zone"]
    if shared_gains:
        tis = np.unique(np.append(
            np.linspace(AP.TI_BOUNDS_S[0], AP.TI_BOUNDS_S[1], TI_PROFILE_POINTS), hgs_ti))
        profile_costs = twin_scorer.batch(np.full(tis.shape, hgs_kp), tis)
    else:
        # no single K_p* to hold fixed while T_I sweeps
        tis, profile_costs = np.empty(0), np.empty(0)
    if shared_gains and options.ti_flat_tolerance > 0:
        resolution = resolve_integral_time(tis, profile_costs, options.ti_flat_tolerance)
        recommended = _gains_block(hgs_kp, resolution["chosen_scale"],
                                   S_twin=float(profile_costs[resolution["chosen_index"]]),
                                   ti_resolution={**resolution, "valley_ti_s": list(resolution["valley_scales"])})
    else:
        # Tie-break off: deliver the twin optimum exactly as the search found it,
        # which is what the paper's HGS-only arm applies. Refining T_I along a
        # 1-D slice at the optimal K_p* would hand over a different gain pair and
        # break comparability with the offline campaign.
        recommended = _gains_block(hgs_kp, hgs_ti, S_twin=hgs_cost,
                                   gain_structure=options.gain_structure,
                                   ti_resolution={"disabled": "ti_flat_tolerance = 0; "
                                                              "the twin optimum is delivered as searched"}
                                   if shared_gains else
                                   {"not_applicable": "per-zone gains: no single K_p* to profile T_I against"})

    # -- reference: the drifted plant's own optimum ------------------------ #
    plant_scorer = _Scorer(drifted, v0, options.cost_tier)
    if options.gain_structure == OUTFEEDER_STRUCTURE:
        # the floor of THIS structure: the out-feeder by the authors' HGS on the true
        # plant, UW and RW at the same saved gains (the BO campaign's "floor")
        fkp, fti, _, _ = _hgs_one_zone(plant_scorer, kp6, ti6, zone=1)
        reference = _gains_block(fkp, fti, note="out-feeder by HGS on the true drifted plant, UW and RW "
                                                "at the saved six gains; unavailable on a real line")
    elif options.gain_structure == PER_ZONE_STRUCTURE:
        six_true = structure_search(plant_scorer.batch_vec, STRUCTURE_BY_NAME[PER_ZONE_STRUCTURE],
                                    baseline=None, seed=0)
        reference = _gains_block(six_true.kp, six_true.ti, note="six-gain search on the true drifted "
                                                                "plant; unavailable on a real line")
    else:
        truth = AP.hierarchical_grid_search(plant_scorer.batch)
        reference = _gains_block(truth.kp, truth.ti,
                                 note="HGS on the true drifted plant; unavailable on a real line")

    for block in (commissioned, hgs_only, recommended, reference):
        # breakdown_vec, not breakdown: a block may carry three different pairs.
        # At T0 the per-zone evaluator is the kernel proven equal to PaperEvaluator
        # at 0 ULP, so a uniform block scores exactly what it always did.
        block["on_plant"] = plant_scorer.breakdown_vec(
            block["kp_star_per_zone"], block["ti_s_per_zone"])
    # what the twin predicts for the gains running today: the screen's "twin, gains today"
    commissioned["S_twin"] = float(twin_scorer.batch_vec(
        np.asarray(commissioned["kp_star_per_zone"], dtype=float)[None, :],
        np.asarray(commissioned["ti_s_per_zone"], dtype=float)[None, :])[0])
    S_reference = reference["on_plant"]["S"]

    if shared_gains:
        simc_ti = simc_integral_time_s(twin, v0, hgs_kp)
        simc = {
            "kp_star": float(hgs_kp), "ti_s": simc_ti,
            "formula": "min(L_min / v0, 4 / K_p*)  (Skogestad 2003, SIMC)",
            "on_twin": twin_scorer.breakdown(hgs_kp, simc_ti),
            "on_plant": plant_scorer.breakdown(hgs_kp, simc_ti),
            "note": "Reference only: SIMC assumes no feedforward, and Eq. (12) scores a set-point step.",
        }
    else:
        simc = {"not_applicable": "SIMC gives one T_I from one K_p*; these gains are per zone"}

    # Name the source for what it actually is: with the tie-break off (the
    # default) the delivered pair is the twin optimum exactly as searched.
    delivered_source = ("recommended (HGS-only, T_I valley tie-break)" if options.ti_flat_tolerance > 0
                        else "recommended (HGS-only, twin optimum as searched)")
    delivered = {**recommended, "source": delivered_source}
    bo_block = None
    if options.bo_refine and not shared_gains:
        bo_block = {"not_applicable": "the authors' BO(5) searches the 2-D "
                                      "(K_p*, T_I) box; these gains are per zone"}
    elif options.bo_refine:
        # NOT `run`: that name is this function's RunSpec argument, still needed
        # below for the commissioning identification.
        bo_run = _bo_refine(plant_scorer, hgs_kp, hgs_ti)
        bo_kp, bo_ti = bo_run.points[bo_run.best_index]
        bo_cost = bo_run.costs[bo_run.best_index]
        bo_block = _gains_block(bo_kp, bo_ti, real_evaluations=len(bo_run.costs), split=bo_run.split,
                                trajectory=[{"kp_star": k, "ti_s": t, "S": c}
                                            for (k, t), c in zip(bo_run.points, bo_run.costs)])
        bo_block["on_plant"] = plant_scorer.breakdown(bo_kp, bo_ti)
        if bo_cost < (recommended["on_plant"]["S"] or math.inf):
            delivered = {**bo_block, "source": "HGS+BO(5), the authors' construction: best of star and random points"}
    steps.append({
        "step": 6, "name": "Retune on the digital twin",
        "status": "pass",
        "detail": (f"{search_meta['search']}: {search_meta['evaluations']} twin evaluations -> "
                   f"K_p* = {_fmt3(hgs_only['kp_star_per_zone'])}, "
                   f"T_I = {_fmt3(hgs_only['ti_s_per_zone'])} s"
                   + (f"; BO refine used {bo_block['real_evaluations']} plant evaluations" if bo_block else "")),
    })

    # -- Step 5: twin validation ------------------------------------------- #
    kp_c, ti_c = commissioned["kp_star"], commissioned["ti_s"]
    _, plant_response = _validation_response(drifted, v0, kp_c, ti_c)
    _, twin_response = _validation_response(twin, v0, kp_c, ti_c)
    validation = prediction_fit(plant_response, twin_response)

    commissioning_ident = identify(RunSpec(run.plant, DriftSpec(), run.protocol))
    epsilon = None
    if commissioning_ident.get("converged"):
        base_twin = twin_params_from_estimates(base, commissioning_ident["estimates"])
        _, base_response = _validation_response(base, v0, kp_c, ti_c)
        _, base_twin_response = _validation_response(base_twin, v0, kp_c, ti_c)
        epsilon = prediction_fit(base_response, base_twin_response)["rmse_N"]
    twin_ok = epsilon is not None and validation["rmse_N"] <= options.epsilon_margin * epsilon + 1e-12
    steps.insert(1, {
        "step": 5, "name": "Build digital twin and validate",
        "status": "pass" if twin_ok else ("unknown" if epsilon is None else "fail"),
        "detail": (f"prediction RMSE {validation['rmse_N']:.4f} N (fit {validation['fit_percent']:.1f} %)"
                   + (f" vs epsilon {options.epsilon_margin:g} x {epsilon:.4f} N" if epsilon is not None
                      else "; commissioning twin did not converge, epsilon unknown")),
    })

    # -- Step 7: validate and restore -------------------------------------- #
    S_delivered = delivered["on_plant"]["S"]
    accepted = S_delivered is not None and S_delivered <= options.c_target_margin * c_target + 1e-12
    steps.append({
        "step": 7, "name": "Validate and restore",
        "status": "pass" if accepted else "fail",
        "detail": (f"S on plant {S_delivered:.4f} vs C_target {options.c_target_margin:g} x {c_target:.4f}"
                   if S_delivered is not None else "delivered gains diverge on the plant"),
    })

    payload = {
        **common,
        "status": "ok",
        "steps": steps,
        "identification": {
            "converged": True,
            "MARE_theta_pct": ident["MARE_theta_pct"],
            "excitation": ident.get("excitation"),
            "rows": ident.get("rows"),
            "identify_run_hash": ident.get("run_hash"),
        },
        "search": {
            "per_zone_gains": per_zone,
            "outfeeder": outfeeder_meta,
            "sequential_stages": seq_stages,
            "budget": search_meta["evaluations"],
            "gain_structure": options.gain_structure,
            "zones": list(ZONES),
            "kp_bounds": list(AP.KP_BOUNDS),
            "ti_bounds_s": list(AP.TI_BOUNDS_S),
            "source": "authors' reply of 2026-09-15 (Q1b)",
            "plant_auto_ti_s": plant_auto_ti_s(base, v0),
        },
        "gains": {
            "delivered": delivered,
            "hgs_only": hgs_only,
            "recommended": recommended,
            "commissioned": commissioned,
            "reference_on_plant": reference,
            "hgs_bo": bo_block,
            "simc_reference": simc,
        },
        "ratio_vs_reference": (S_delivered / S_reference
                               if S_delivered is not None and S_reference not in (None, 0) else None),
        "twin_validation": {**validation, "epsilon_N": epsilon, "margin": options.epsilon_margin,
                            "passed": twin_ok, "validation_gains": "commissioned"},
        "acceptance": {"S_delivered": S_delivered, "C_target": c_target,
                       "margin": options.c_target_margin, "passed": accepted},
        "ti_profile": [{"ti_s": float(t), "S_twin": (float(c) if math.isfinite(c) else None)}
                       for t, c in zip(tis[::2], profile_costs[::2])],
        "seconds": time.time() - started,
    }
    # The cache holds the search as run; the restore is derived from it on the way
    # out, so results cached before the restore existed get it too.
    payload = _commit_per_zone(run, options, payload)
    cache.store(CACHE_KIND, key, payload)
    if options.gain_structure == OUTFEEDER_STRUCTURE and stored is None and per_zone_saved(payload):
        # the six gains were saved during this run: the same click now keys on them
        cache.store(CACHE_KIND, _retune_key(run, options, {"kp": per_zone["kp_star_per_zone"],
                                                            "ti": per_zone["ti_s_per_zone"]}), payload)
    return apply_restore({**payload, "cached": False})


def per_zone_saved(payload: Mapping[str, Any]) -> bool:
    return bool(((payload.get("search") or {}).get("per_zone_gains") or {}).get("saved"))


def _commit_per_zone(run: RunSpec, options: RetuneOptions, payload: dict[str, Any]) -> dict[str, Any]:
    """Save the six gains searched for this case -- only if the line accepted the case.

    A case whose result step 7 restored (the twin's gains did not beat today's on the
    line) must not commission the plant: every later case would freeze UW and RW at
    gains the line refused. Reused gains are already in the store.
    """

    pz = (payload.get("search") or {}).get("per_zone_gains")
    if payload.get("status") != "ok" or not pz:
        return payload
    out = copy.deepcopy(payload)
    pz = out["search"]["per_zone_gains"]
    if pz.get("reused"):
        pz["saved"] = False
        pz["not_saved_because"] = "reused: these six gains were already saved"
    elif apply_restore(out)["restore"]["restored"]:
        pz["saved"] = False
        pz["not_saved_because"] = ("step 7 restored today's gains: the line did not accept this case, "
                                   "so it does not commission the plant")
    else:
        pz["saved"] = True
        _save_per_zone(run, options, pz)
    return out


def _save_per_zone(run: RunSpec, options: RetuneOptions, per_zone: Mapping[str, Any]) -> None:
    zone_gains.save(run, options.cost_tier, per_zone,
                    source_run_hash=per_zone.get("source_run_hash") or run_hash(run),
                    S_twin=per_zone.get("S_twin"))


def apply_restore(payload: dict[str, Any]) -> dict[str, Any]:
    """Step 7's restore (assumption A9): the twin's gains reach the line only if
    they beat the gains it runs today, both scored on the drifted plant.

    Pure and idempotent; the input is not modified. The twin's proposal is kept
    as `gains.twin_candidate` whichever way it goes.
    """

    gains = payload.get("gains")
    if payload.get("status") != "ok" or not gains or "restore" in payload:
        return payload
    out = copy.deepcopy(payload)
    gains = out["gains"]
    candidate, commissioned = gains["delivered"], gains["commissioned"]
    s_candidate = (candidate.get("on_plant") or {}).get("S")
    s_commissioned = (commissioned.get("on_plant") or {}).get("S")
    restored = s_commissioned is not None and (s_candidate is None or not s_candidate < s_commissioned)
    gains["twin_candidate"] = candidate
    if restored:
        gains["delivered"] = {**copy.deepcopy(commissioned),
                              "source": "restored: the commissioned gains (the twin's gains did not "
                                        "beat them on the line)"}
    s_delivered = gains["delivered"]["on_plant"]["S"]
    acceptance = out.get("acceptance") or {}
    c_target, margin = acceptance.get("C_target"), acceptance.get("margin", 1.0)
    accepted = s_delivered is not None and c_target is not None and s_delivered <= margin * c_target + 1e-12
    out["acceptance"] = {**acceptance, "S_delivered": s_delivered, "passed": accepted}
    s_reference = ((gains.get("reference_on_plant") or {}).get("on_plant") or {}).get("S")
    out["ratio_vs_reference"] = (s_delivered / s_reference
                                 if s_delivered is not None and s_reference not in (None, 0) else None)
    out["restore"] = {"restored": restored, "S_candidate": s_candidate,
                      "S_commissioned": s_commissioned, "rule": "A9"}
    cand = "diverges" if s_candidate is None else f"{s_candidate:.4f}"
    target = f"; C_target {margin:g} x {c_target:.4f} {'met' if accepted else 'not met'}" if c_target is not None else ""
    for step in out.get("steps", []):
        if step.get("step") == 7:
            step["status"] = "restored" if restored else ("pass" if accepted else "fail")
            step["detail"] = (f"twin's gains S on plant {cand} vs gains today {s_commissioned:.4f} -> "
                              + ("commissioned gains restored" if restored else "twin's gains delivered")
                              + target)
    return out
