"""The session API.

The only module where the pure pipeline and the session record meet. Routes
here read the session, build a RunSpec, call the pipeline, and write results
back. The pipeline itself never sees the session, and the session never
computes.

Nothing under /validate/* is touched.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.atlas import reader as atlas_reader
from backend.pipeline import identify as identify_pipeline
from backend.pipeline import plant as plant_pipeline
from backend.pipeline import retune as retune_pipeline
from backend.pipeline import step as step_pipeline
from backend.pipeline import trace as trace_pipeline
from backend.pipeline import zone_gains
from backend.pipeline.payloads import (
    DriftSpec,
    ProtocolSpec,
    RunSpec,
    drift_from_dict,
    paper_record_s,
    patch_protocol,
    plant_from_dict,
    protocol_from_dict,
    to_dict as spec_to_dict,
)
from backend.pipeline.twin import active_plant_spec
from backend.session import store
from backend.session.state import SessionState, staleness
from backend.session.state import to_dict as session_to_dict
from backend.validation.retuning import PROTOCOL_FIELD_MATCHED, PROTOCOL_LOGGING_ONLY

router = APIRouter(tags=["session"])


def _preset(name: str, protocol, citation: str) -> dict[str, Any]:
    """Render a campaign `Protocol` as a session ProtocolSpec plus its citation."""

    return {
        "name": name,
        "citation": citation,
        "protocol": spec_to_dict(
            ProtocolSpec(
                T_log_ms=protocol.log_sample_time_s * 1000.0,
                excitation="E_Toggle",
                record_s=protocol.record_duration_s,
                pct_T=protocol.tension_noise_fraction,
                pct_v=protocol.velocity_noise_fraction,
                LPF_T_hz=protocol.tension_lpf_hz,
                LPF_v_hz=protocol.velocity_lpf_hz,
                Kp_star=100.0,
                seed=0,
            )
        ),
    }


# The paper's published acquisition cells. The section 3.4 preset named in the
# originating brief -- E_Toggle, 20 ms, 0.3 %, tension-only, Kp* = 100 -- IS
# `logging_only`; it is not a third entry.
PAPER_PRESETS: dict[str, dict[str, Any]] = {
    "field_matched": _preset(
        "field_matched", PROTOCOL_FIELD_MATCHED,
        "Lee et al. v5, Section 4.2 / Table S9 (field-matched)",
    ),
    "logging_only": _preset(
        "logging_only", PROTOCOL_LOGGING_ONLY,
        "Lee et al. v5, Section 4.2 / Table S9 (logging-only); the Section 3.4 cell",
    ),
}


class RunSpecBody(BaseModel):
    run_spec: dict[str, Any]


class DeriveBody(RunSpecBody):
    """`/plant/derive` stays pure: swap is an input, not a session lookup.

    Resolving swap from the session would make the response depend on state
    the body does not carry. Resolving it in the browser would need the
    theta-hat -> PlantSpec inversion, which is physics the frontend may not
    do. So the caller passes the estimates it wants treated as truth, and the
    server does the inversion.
    """

    swap_with_estimates: dict[str, float] | None = None


class IdentifyBody(RunSpecBody):
    commit: bool = False


class RetuneBody(RunSpecBody):
    """Options for Algorithm 1 Steps 4-7; every default is the paper's reading.

    Defaults are taken from the pipeline, never restated here: a second copy of
    `ti_flat_tolerance` silently overrode the pipeline's own default and kept
    the T_I tie-break switched on after it had been retired.
    """

    bo_refine: bool = False
    epsilon_margin: float = retune_pipeline.RetuneOptions().epsilon_margin
    c_target_margin: float = retune_pipeline.RetuneOptions().c_target_margin
    ti_flat_tolerance: float = retune_pipeline.DEFAULT_TI_FLAT_TOLERANCE
    # Both defaults come from the pipeline, never restated here -- that is exactly
    # the mistake the docstring above records.
    cost_tier: str = retune_pipeline.DEFAULT_COST_TIER
    gain_structure: str = retune_pipeline.DEFAULT_GAIN_STRUCTURE


class BaselineBody(RunSpecBody):
    """RUN PHYSICAL MACHINE: the commissioned gains and their step on the drifted plant."""

    cost_tier: str | None = None
    measured: bool = False
    seed: int = 0


class StepBody(RunSpecBody):
    """The cost test's step response with given gains, on the plant or (with θ̂) the twin.

    `kp_star` / `ti_s` are a scalar shared by the three tension zones or one
    value per zone (UW, out-feeder, RW), exactly as `/retune` reports them.
    """

    kp_star: float | list[float]
    ti_s: float | list[float]
    estimates: dict[str, float] | None = None
    # Run it as the line would see it: the protocol's tension noise and filter
    # in the loop, and a measured trace next to the true one.
    measured: bool = False
    seed: int = 0


class TraceBody(RunSpecBody):
    """θ̂ is optional: with it the twin plays the same record beside the plant."""

    estimates: dict[str, float] | None = None


class TwinBody(BaseModel):
    action: Literal["load", "unload", "swap"]
    run_hash: str | None = None


class ProtocolPatch(BaseModel):
    model_config = {"extra": "forbid"}

    T_log_ms: float | None = None
    excitation: str | None = None
    record_s: float | None = None
    pct_T: float | None = None
    pct_v: float | None = None
    LPF_T_hz: float | None = None
    LPF_v_hz: float | None = None
    Kp_star: float | None = None
    seed: int | None = None


def _spec_or_error(payload: dict[str, Any]) -> RunSpec:
    """Turn a request body into a RunSpec, mapping validation to 400/422.

    Dispatch is on which constructor raised, not on the exception's message:
    a malformed protocol is a client mistake in the request shape (422); an
    unbuildable plant is a physically invalid request (400 naming the field).

    `TypeError` is caught alongside `ValueError` everywhere: the spec
    constructors coerce with `float()`/`int()`, so well-formed JSON carrying a
    wrong-typed field (`{"EA_pct": null}`, `{"R": 5}`) raises `TypeError`, not
    `ValueError`. That is still a client mistake, not a server fault, and must
    not escape as a 500.
    """

    try:
        plant = plant_from_dict(payload["plant"])
        drift = drift_from_dict(payload["drift"])
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"missing {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        protocol = protocol_from_dict(payload["protocol"])
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"missing {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RunSpec(plant=plant, drift=drift, protocol=protocol)


def _session_payload(state: SessionState) -> dict[str, Any]:
    payload = session_to_dict(state)
    payload["staleness"] = staleness(state)
    payload["current_hash"] = state.current_hash
    return payload


@router.get("/session")
def get_session() -> dict[str, Any]:
    return _session_payload(store.load())


@router.put("/session/plant")
def put_plant(payload: dict[str, Any]) -> dict[str, Any]:
    state = store.load()
    try:
        state.plant = plant_from_dict(payload)
        plant_pipeline.params_from_spec(state.plant)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save(state)
    return _session_payload(state)


@router.put("/session/drift")
def put_drift(payload: dict[str, Any]) -> dict[str, Any]:
    state = store.load()
    try:
        state.drift = drift_from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save(state)
    return _session_payload(state)


@router.patch("/session/protocol")
def patch_protocol_route(patch: ProtocolPatch) -> dict[str, Any]:
    """The Adopt button. Only the fields present in the body are changed."""

    state = store.load()
    try:
        state.protocol = patch_protocol(
            state.protocol, patch.model_dump(exclude_unset=True)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store.save(state)
    return _session_payload(state)


@router.put("/session/twin")
def put_twin(body: TwinBody) -> dict[str, Any]:
    state = store.load()
    if body.action == "unload":
        state.twin = None
    elif body.action == "load":
        if not state.theta_hat:
            raise HTTPException(
                status_code=409, detail="identify and commit a result before loading a twin"
            )
        committed_hash = state.theta_hat.get("run_hash")
        if body.run_hash is not None and body.run_hash != committed_hash:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run_hash {body.run_hash!r} is not the committed identification "
                    f"({committed_hash!r}); load without a run_hash to use it"
                ),
            )
        state.twin = {
            "loaded": True,
            "source_run_hash": committed_hash,
            "swapped": False,
            "run_spec": spec_to_dict(state.run_spec),
        }
    else:  # swap
        if not state.twin or not state.twin.get("loaded"):
            raise HTTPException(
                status_code=409, detail="load a twin before swapping columns"
            )
        state.twin = {**state.twin, "swapped": not state.twin.get("swapped", False)}
    store.save(state)
    return _session_payload(state)


@router.post("/session/reset")
def reset_session() -> dict[str, Any]:
    return _session_payload(store.reset())


@router.get("/session/presets")
def get_presets() -> dict[str, Any]:
    return {"presets": PAPER_PRESETS}


@router.post("/plant/derive")
def derive_route(body: DeriveBody) -> dict[str, Any]:
    run = _spec_or_error(body.run_spec)
    estimates = body.swap_with_estimates
    try:
        # theta-hat was fitted against the *drifted* plant, so the twin
        # `plant_spec_from_theta` rebuilds already embodies the drift. Passing
        # `run.drift` on would apply it a second time and the swapped readouts
        # would answer for a plant that does not exist. The identity drift is
        # also the honest `drift_applied: false` -- the twin *is* the drifted
        # plant; there is no second drift left to show.
        effective = RunSpec(
            plant=active_plant_spec(run.plant, estimates, bool(estimates)),
            drift=DriftSpec() if estimates else run.drift,
            protocol=run.protocol,
        )
        payload = plant_pipeline.derive(effective)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload["derived_from"] = "theta_hat" if estimates else "theta_true"
    return payload


@router.post("/identify")
def identify_route(body: IdentifyBody) -> dict[str, Any]:
    run = _spec_or_error(body.run_spec)
    try:
        result = identify_pipeline.identify(run)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if body.commit:
        state = store.load()
        origin = spec_to_dict(run)
        state.plant, state.drift, state.protocol = run.plant, run.drift, run.protocol
        state.last_run = {
            "run_hash": result["run_hash"],
            "n_samples": result["n_samples"],
            "produced_at": result["produced_at"],
            "run_spec": origin,
        }
        state.theta_hat = {
            "run_hash": result["run_hash"],
            "estimates": result["estimates"],
            "rows": result["rows"],
            "MARE_theta_pct": result["MARE_theta_pct"],
            "converged": result["converged"],
            "run_spec": origin,
        }
        store.save(state)
    return result


@router.post("/identify/paper-check")
def paper_check_route(body: RunSpecBody) -> dict[str, Any]:
    """Return the published cell for this protocol, or say there is none.

    A protocol matches a preset only if every acquisition field agrees. Nothing
    is interpolated: an unmatched protocol is `not_published`, not a nearby
    number.
    """

    run = _spec_or_error(body.run_spec)
    current = spec_to_dict(run.protocol)
    for name, preset in PAPER_PRESETS.items():
        published = preset["protocol"]
        if all(
            current[field] == published[field]
            for field in ("T_log_ms", "excitation", "record_s", "pct_T", "pct_v",
                          "LPF_T_hz", "LPF_v_hz", "Kp_star")
        ):
            return {"status": "published", "preset": name,
                    "citation": preset["citation"], "protocol": published}
    return {"status": "not_published",
            "note": "no published cell uses this acquisition protocol"}


def _optional_hz(raw: str | None, field: str) -> float | None:
    """Parse an LPF query value that may legitimately be "no filter".

    A null LPF is a real protocol, not a malformed request, so it must survive
    the query string and reach `on_grid` as None -- where it is correctly
    reported off-grid. Omitting the parameter instead would silently apply the
    50 Hz default and answer a no-filter protocol with the 50 Hz cell, which is
    inventing a value: a worse failure than the 422 this replaces.
    """

    if raw is None or raw.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"{field} must be a number or 'none'"
        ) from exc


@router.get("/atlas")
def atlas_route(
    plant_id: str,
    T_log_ms: float,
    excitation: str,
    record_s: float | None = None,
    pct_T: float = 0.003,
    pct_v: float = 0.003,
    LPF_T_hz: str = "50.0",
    LPF_v_hz: str = "50.0",
    Kp_star: float = 100.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Resolve one (plant, protocol) against the precomputed gain atlas.

    Always 200: "no cell for this protocol" is an answer, not an error, and the
    screen renders it as `not precomputed` exactly as an unpublished paper cell
    renders as `not published`. The one 422 left is a genuinely malformed LPF
    value -- `LPF_T_hz=abc`; an EMPTY or `none` LPF is a real no-filter protocol
    and is answered, off-grid, with the field named.

    The index is rebuilt per request while the generator is still writing, so a
    cell finished a minute ago is visible without a restart. That costs one
    directory scan of a few hundred small files.
    """

    atlas_reader.twin_index.cache_clear()
    # An omitted record length means "the published one for this excitation",
    # not 16 s: the atlas holds each excitation at its own duration, and a
    # single default would put five of the six off-grid.
    if record_s is None:
        record_s = paper_record_s(excitation, pct_v)
    protocol = {
        "T_log_ms": T_log_ms, "excitation": excitation, "record_s": record_s,
        "pct_T": pct_T, "pct_v": pct_v,
        "LPF_T_hz": _optional_hz(LPF_T_hz, "LPF_T_hz"),
        "LPF_v_hz": _optional_hz(LPF_v_hz, "LPF_v_hz"),
        "Kp_star": Kp_star, "seed": seed,
    }
    payload = atlas_reader.lookup(plant_id, protocol)
    payload["manifest"] = atlas_reader.load_manifest()
    # How much of the atlas exists right now. With `off_grid_field` null the
    # protocol is on the grid and its cell is simply not written yet, and this
    # is what lets the screen say "not yet, N of M" instead of a dead end.
    payload["cell_counts"] = atlas_reader.cell_counts()
    return payload


@router.post("/retune")
def retune_route(body: RetuneBody) -> dict[str, Any]:
    """Algorithm 1 Steps 4-7 for the posted RunSpec: K_p* and T_I from the twin.

    Works for any protocol, drift or hand-edited plant -- the gain atlas answers
    only on its grid. A failed identification is a 200 with
    `status: identification_failed` and no gains, the same way `/atlas` answers
    an unconverged cell: no twin is an answer, not a server error.
    """

    run = _spec_or_error(body.run_spec)
    try:
        options = retune_pipeline.RetuneOptions(
            bo_refine=body.bo_refine,
            epsilon_margin=body.epsilon_margin,
            c_target_margin=body.c_target_margin,
            ti_flat_tolerance=body.ti_flat_tolerance,
            cost_tier=body.cost_tier,
            gain_structure=body.gain_structure,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return retune_pipeline.retune(run, options)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/plant/trace")
def trace_route(body: TraceBody) -> dict[str, Any]:
    """The protocol's identification record, played on the plant (and twin).

    This is what the machine screen animates: real tensions and roller speeds,
    decimated to at most 600 samples.
    """

    run = _spec_or_error(body.run_spec)
    try:
        return trace_pipeline.acquisition_trace(run, body.estimates)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"estimates missing {exc}") from exc
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ZoneGainsBody(RunSpecBody):
    """The saved six gains are per plant and cost tier; the drift in run_spec is ignored."""

    cost_tier: str = retune_pipeline.DEFAULT_COST_TIER


def _zone_gains_args(body: ZoneGainsBody):
    from backend.pipeline.retune import TIER_NAMES

    if body.cost_tier not in TIER_NAMES:
        raise HTTPException(status_code=422, detail=f"cost_tier must be one of {TIER_NAMES}")
    return _spec_or_error(body.run_spec), body.cost_tier


@router.post("/zone-gains")
def zone_gains_route(body: ZoneGainsBody) -> dict[str, Any]:
    """The six per-zone gains saved for this plant (stage 1), or null."""

    run, tier = _zone_gains_args(body)
    return {"saved": zone_gains.load(run, tier), "cost_tier": tier}


@router.post("/zone-gains/delete")
def zone_gains_delete_route(body: ZoneGainsBody) -> dict[str, Any]:
    """Forget the saved six gains: the next retune of this plant searches all six again."""

    run, tier = _zone_gains_args(body)
    return {"deleted": zone_gains.delete(run, tier)}


@router.post("/plant/baseline")
def baseline_route(body: BaselineBody) -> dict[str, Any]:
    """The line as it runs today: commissioned gains on the drifted plant, with the cost test."""

    run = _spec_or_error(body.run_spec)
    try:
        return step_pipeline.baseline(run, body.cost_tier, measured=body.measured, seed=body.seed)
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/plant/step")
def step_route(body: StepBody) -> dict[str, Any]:
    """The +20 % cost-test step on the drifted plant, or on the twin from θ̂.

    Feeds the two step-response windows: the same gains on the physical plant
    and on the digital twin, so their difference is the twin error as a
    response, not a number.
    """

    run = _spec_or_error(body.run_spec)
    try:
        return step_pipeline.step_response(run, body.kp_star, body.ti_s, body.estimates,
                                           measured=body.measured, seed=body.seed)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"estimates missing {exc}") from exc
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
