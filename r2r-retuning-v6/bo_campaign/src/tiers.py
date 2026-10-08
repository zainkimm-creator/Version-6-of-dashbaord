"""Four nested cost definitions for R2R tension retuning, T0 -> T3.

T0  the authors' Eq. (12), reproduced bit for bit. The gate in
    tests/test_t0_identity.py pins it to backend.validation.retuning_paper_eval
    at 0 ULP, so every tier below is built on a simulator proven equal to theirs.

T1  REPAIR. Same experiment, same dynamics, three defects in the COST removed:

    (a) Setpoint-relative error. Eq. (12) divides RMSE by an absolute 3 N. The six
        retuning plants run at T_ref = 12, 64.8, 270, 229.5, 360, 360 N, so the
        paper's cost silently demands 25 % of setpoint from P001 and 0.83 % from
        P186 -- a 30x difference in strictness that has nothing to do with control.
        T1 divides each span's RMSE by that span's own setpoint.

    (b) Continuous settling. Eq. (12) uses the LAST sample outside the +/-2 % band,
        which jumps by a whole oscillation when a marginal overshoot crosses the
        band. Measured on P001/D07: a 0.14 % gain change moves t_s 1.062 -> 1.460 s
        and the cost +67 %, while the twin's surface is smooth and improving there.
        T1 uses TOTAL time outside the band, softened by a sigmoid -- continuous in
        the gains, and a better spec besides (two excursions score worse than one,
        which last-sample-outside cannot see).

    (c) One aggregator, not three. Eq. (12) combines the spans with MEAN for RMSE,
        MAX for overshoot and ANY for settling -- three different notions of "the
        three spans" inside one scalar. With per-roller gains the mean lets the
        search sacrifice one span to flatter two. T1 forms a complete per-span cost
        and aggregates once, with a p-norm (p=4) that approximates the worst span
        while staying differentiable.

T2  = T1 + TRUE ACTUATION. Torque saturation, slew-rate limit, conditional-
    integration anti-windup, measurement noise on the feedback with the first-order
    filter the dashboard already applies, and a control-effort term. The paper's
    experiment has none of these and does not clamp the integrator, which is why its
    optimum parks on the T_I ceiling: with no cost on effort or windup, the integral
    time has nothing pushing it anywhere and the valley is flat.

T3  = T2 + TRUE R2R OPERATION. A roll is not a fixed radius. The unwinder's radius
    collapses and the rewinder's grows through a run, carrying inertia with them
    (J ~ R^4), so gains commissioned at one radius are not the gains that hold at
    the other end of the roll. T3 scores the SAME gains at three roll positions and
    reports the worst -- gains must work full-roll or they are not commissioned.

Every tier returns the same row so the tiers can be compared directly:
    [S, e_rel, OS_pct, t_out_s, effort, sat_frac]

Nothing outside V6_multiloop_retuning_2026-09-29/multiloop/ is modified.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.validation.retuning_jax import pack_plant                      # noqa: E402
from backend.validation.retuning_paper_eval import (                        # noqa: E402
    BAND_FRACTION, DT_S, N_STEPS, S_E_N, S_OS_PCT, S_T_S,
    STEP_FRACTION, STEP_INDEX, W_OS,
)

TIERS = ("T0", "T1", "T2", "T3")
OUTPUTS = ("S", "err_metric", "OS_percent", "t_out_s", "effort", "sat_frac")
N_ROLLERS = 3

# The three TENSION-controlled zones, in the order every gain vector uses.
# The in-feeder is NOT among them: it is the velocity master and enters the model
# as v_feed, a fixed input with no Kp/T_I. So "tune the middle loop" means the
# OUT-FEEDER nip, and a per-roller gain vector is (UW, out-feeder, RW).
ROLLER_NAMES = ("UW", "OutFeeder", "RW")
ROLLER_LABELS = ("unwinder", "out-feeder (nip)", "rewinder")

# ---- T1: repaired scales ------------------------------------------------------
S_E_REL = 0.02          # 1 cost unit = RMS error of 2 % of that span's setpoint
S_T_OUT = 3.0           # 1 cost unit = 3 s spent outside the +/-2 % band
SOFT_BAND = 0.15        # sigmoid width, as a fraction of the band half-width
P_NORM = 4.0            # span aggregation: ->max as p->inf, differentiable at p=4

# ---- T2: actuation realism ----------------------------------------------------
SAT_FACTOR = 2.0        # torque limit = SAT_FACTOR x (hold + accelerate-in-T_ACC)
T_ACC_S = 0.5
T_SLEW_S = 0.05         # full-scale torque slew takes T_SLEW_S
NOISE_REL = 0.005       # measurement noise std = 0.5 % of setpoint
LPF_TAU_S = 0.02        # first-order feedback filter
W_EFFORT = 0.5          # weight on the control-effort term
DU_SCALE = 1.0          # effort normalised by the slew limit (see _make_cost)

# ---- T3: roll positions -------------------------------------------------------
# The plant's stated R is taken as mid-roll. A full roll is R_FULL_REL x that, a
# spent roll R_CORE_REL x it, and inertia scales with R^4 about the same nominal.
# The paper gives no roll geometry, so these are stated assumptions, not data.
ROLL_FRACTIONS = (0.1, 0.5, 0.9)
R_FULL_REL, R_CORE_REL = 1.35, 0.50


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def roll_geometry(fraction: float) -> tuple[float, float]:
    """(unwinder radius multiplier, rewinder radius multiplier) at a roll fraction.

    Constant web cross-section, so area leaves the unwinder linearly in time:
    R(phi) = sqrt(R_full^2 - phi (R_full^2 - R_core^2)). The rewinder is the mirror.
    """
    a, b = R_FULL_REL ** 2, R_CORE_REL ** 2
    r_uw = float(np.sqrt(a - fraction * (a - b)))
    r_rw = float(np.sqrt(b + fraction * (a - b)))
    return r_uw, r_rw


def _make_cost(tier: str, n_steps: int, step_index: int, dt: float):
    """Per-candidate cost for one tier. `kp_star`, `ti` are (3,) vectors."""

    jax = _jax()
    import jax.numpy as jnp

    polarity = jnp.array([-1.0, 1.0, 1.0])
    realistic = tier in ("T2", "T3")

    def unpack(p):
        return p[0], p[1:4], p[4:7], p[7:10], p[10:13], p[13]

    def derivatives(x, u, p):
        EA, L, R, J, f, v_feed = unpack(p)
        T, w = x[:3], x[3:]
        v = R * w
        dT1 = (EA / L[0]) * (v_feed - v[0]) - (v_feed / L[0]) * T[0]
        dT2 = (EA / L[1]) * (v[1] - v_feed) + (T[0] * v_feed - T[1] * v[1]) / L[1]
        dT3 = (EA / L[2]) * (v[2] - v[1]) + (T[1] * v[1] - T[2] * v[2]) / L[2]
        dw1 = (u[0] - T[0] * R[0] - f[0] * w[0]) / J[0]
        dw2 = (u[1] + T[1] * R[1] - T[2] * R[1] - f[1] * w[1]) / J[1]
        dw3 = (u[2] + T[2] * R[2] - f[2] * w[2]) / J[2]
        return jnp.stack([dT1, dT2, dT3, dw1, dw2, dw3])

    def rk4(x, u, p):
        k1 = derivatives(x, u, p)
        k2 = derivatives(x + 0.5 * dt * k1, u, p)
        k3 = derivatives(x + 0.5 * dt * k2, u, p)
        k4 = derivatives(x + dt * k3, u, p)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def steady_v(p, target):
        EA, L, R, J, f, v_feed = unpack(p)
        v_uw = v_feed * (1.0 - target[0] / EA)
        v_nip = v_feed * (EA - target[0]) / jnp.maximum(EA - target[1], 1e-9)
        v_rw = v_nip * (EA - target[1]) / jnp.maximum(EA - target[2], 1e-9)
        return jnp.stack([v_uw, v_nip, v_rw])

    def cost(p, ref_pre, ref_post, kp_star, ti, noise):
        EA, L, R, J, f, v_feed = unpack(p)
        step = ref_post - ref_pre
        v_post = steady_v(p, ref_post)
        k_vel = 1.4 * J * jnp.sqrt(EA * R * R / (J * L))
        x0 = jnp.concatenate([ref_pre, steady_v(p, ref_pre) / R])

        # torque limit: hold the web + accelerate the roller to speed in T_ACC_S
        w_nom = jnp.abs(steady_v(p, ref_post) / R)
        u_max = SAT_FACTOR * (jnp.abs(ref_post * R) + f * w_nom + J * w_nom / T_ACC_S)
        du_max = u_max * (dt / T_SLEW_S)
        alpha = dt / (LPF_TAU_S + dt)          # first-order feedback filter

        def body(carry, k):
            x, integ, u_prev, t_meas, acc_du, acc_sat = carry
            T, w = x[:3], x[3:]
            target = jnp.where(k >= step_index, ref_post, ref_pre)

            if realistic:
                # the controller sees a filtered, noisy measurement; metrics do not
                t_raw = T + noise[k]
                t_meas = t_meas + alpha * (t_raw - t_meas)
                t_fb = t_meas
            else:
                t_fb = T

            err = target - t_fb
            # The authors integrate BEFORE forming the correction, so the correction
            # sees the NEW integral. Deferring the update by one sample perturbs the
            # trajectory and diverges by ~8 % in the lightly damped regime -- the
            # identity gate caught exactly that, so keep this order.
            integ_new = integ + polarity * err * dt
            correction = (L / EA) * kp_star * (polarity * err + integ_new / ti)
            w_ref = (v_post + correction) / R
            tau_web = R * jnp.stack([-T[0], T[1] - T[2], T[2]])
            u_cmd = k_vel * (w_ref - w) + f * w - tau_web

            if realistic:
                u_cmd = u_prev + jnp.clip(u_cmd - u_prev, -du_max, du_max)   # slew
                u = jnp.clip(u_cmd, -u_max, u_max)                           # torque
                saturated = jnp.abs(u_cmd) >= u_max
                # conditional integration: against a stop, discard this sample's
                # integration rather than winding up behind the saturated actuator
                integ_next = jnp.where(saturated, integ, integ_new)
                acc_du = acc_du + ((u - u_prev) / jnp.maximum(du_max, 1e-12)) ** 2
                acc_sat = acc_sat + jnp.mean(saturated.astype(jnp.float64))
            else:
                u = u_cmd
                integ_next = integ_new

            return (rk4(x, u, p), integ_next, u, t_meas, acc_du, acc_sat), x[:3]

        # The line is ALREADY RUNNING when the step test begins, so the actuator
        # starts at the torque that holds the pre-step operating point -- not at
        # zero. Starting it at zero makes the slew limiter ramp up from nothing and
        # leaves the first ~25 ms effectively open-loop; on the stiffer plants that
        # startup transient alone diverges the run, saturating 20.7 % of the time
        # with a value identical across every gain structure. Gain-independent
        # saturation is the tell: it was the initial condition, not the control.
        w0 = steady_v(p, ref_pre) / R
        tau_web0 = R * jnp.stack([-ref_pre[0], ref_pre[1] - ref_pre[2], ref_pre[2]])
        u0 = f * w0 - tau_web0
        init = (x0, jnp.zeros(3), u0, ref_pre, jnp.zeros(3), jnp.asarray(0.0))
        (x_end, _, _, _, acc_du, acc_sat), states = jax.lax.scan(
            body, init, jnp.arange(n_steps))
        trace = jnp.concatenate([states, x_end[:3][None, :]], axis=0)
        after = trace[step_index + 1:]
        err = after - ref_post
        n_after = after.shape[0]

        rmse_span = jnp.sqrt(jnp.mean(err ** 2, axis=0))                 # (3,)
        # Associate exactly as the authors do -- 100*(M/step), NOT (100*M)/step.
        # The two differ in the last ULP and the gate is an equality, not allclose.
        os_span = 100.0 * (jnp.maximum(jnp.max(err, axis=0), 0.0) / jnp.abs(step))
        band = BAND_FRACTION * jnp.abs(step)                             # (3,)
        effort = jnp.sqrt(jnp.mean(acc_du / n_after))
        sat_frac = acc_sat / n_after

        # --- T1+: relative error, smooth total time outside band, one aggregator
        # (T0 is NOT implemented here -- it delegates to paper_t0, the kernel already
        #  proven equal to the authors' at 0 ULP. A second copy reproduced them only
        #  to 5.7e-14 and would be one more thing to keep honest.)
        err_rel = rmse_span / jnp.abs(ref_post)                          # (3,)
        excess = (jnp.abs(err) - band) / jnp.maximum(band * SOFT_BAND, 1e-12)
        t_out_span = dt * jnp.sum(jax.nn.sigmoid(excess), axis=0)        # (3,)

        j_span = ((err_rel / S_E_REL) ** 2
                  + W_OS * (os_span / S_OS_PCT) ** 2
                  + (t_out_span / S_T_OUT) ** 2)
        S = (jnp.mean(j_span ** P_NORM)) ** (1.0 / P_NORM)
        if realistic:
            S = S + W_EFFORT * effort ** 2

        return jnp.stack([S, jnp.max(err_rel), jnp.max(os_span),
                          jnp.max(t_out_span), effort, sat_frac])

    return cost


@lru_cache(maxsize=16)
def build_kernel(tier: str, n_steps: int = N_STEPS, step_index: int = STEP_INDEX,
                 dt: float = DT_S):
    """One plant, many candidates: (p, pre, post, kp[n,3], ti[n,3], noise) -> (n, 6)."""

    jax = _jax()
    return jax.jit(jax.vmap(_make_cost(tier, n_steps, step_index, dt),
                            in_axes=(None, None, None, 0, 0, None)))


class TieredEvaluator:
    """Eq. (12) and its three repairs, for one plant, with per-roller gains.

    T3 scores the same gains at three roll positions and returns the worst.
    """

    def __init__(self, params, line_speed_m_s: float, tier: str = "T0", *,
                 chunk: int = 4096, seed: int = 0):
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        _jax()
        import jax.numpy as jnp

        self.tier = tier
        self.chunk = int(chunk)

        if tier == "T0":
            # the authors' arithmetic by construction, not by vigilance
            from paper_t0 import PerRollerEvaluator

            self._paper = PerRollerEvaluator(params, line_speed_m_s, chunk=chunk)
            return
        self._paper = None

        base = np.asarray(pack_plant(params, line_speed_m_s), dtype=np.float64)
        pre = np.asarray(params.tension_ref_N, dtype=np.float64)

        # T3 evaluates the same gains on several plants (one per roll position)
        self.plants = [jnp.asarray(base)] if tier != "T3" else [
            jnp.asarray(self._at_roll(base, phi)) for phi in ROLL_FRACTIONS]

        self.ref_pre = jnp.asarray(pre)
        self.ref_post = jnp.asarray(pre * (1.0 + STEP_FRACTION))
        rng = np.random.default_rng(seed)
        noise = (rng.standard_normal((N_STEPS + 1, N_ROLLERS)) * NOISE_REL * pre
                 if tier in ("T2", "T3") else np.zeros((N_STEPS + 1, N_ROLLERS)))
        self.noise = jnp.asarray(noise)
        self._kernel = build_kernel(tier)

    @staticmethod
    def _at_roll(p: np.ndarray, fraction: float) -> np.ndarray:
        """Move the unwinder and rewinder to a roll position. J ~ R^4."""
        q = p.copy()
        r_uw, r_rw = roll_geometry(fraction)
        mult = np.array([r_uw, 1.0, r_rw])          # the nip is a driven roller
        q[4:7] = q[4:7] * mult
        q[7:10] = q[7:10] * (mult ** 4)
        return q

    @staticmethod
    def _as_batch(g) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64)
        if g.ndim == 0:
            g = np.full((1, N_ROLLERS), float(g))
        elif g.ndim == 1:
            g = (np.full((1, N_ROLLERS), g[0]) if g.size == 1 else g.reshape(1, N_ROLLERS))
        return np.ascontiguousarray(g, dtype=np.float64)

    def evaluate(self, kp, ti) -> np.ndarray:
        import jax.numpy as jnp

        if self._paper is not None:
            # [S, RMSE_N, OS_%, t_s_s, integrator_peak] -> the common 6-wide row.
            # Note T0's error column is ABSOLUTE newtons and its time column is
            # last-sample-outside; T1+ report relative error and total-time-outside.
            rows = self._paper.evaluate(kp, ti)
            pad = np.zeros((rows.shape[0], 2))
            return np.concatenate([rows[:, :4], pad], axis=1)

        kp, ti = self._as_batch(kp), self._as_batch(ti)
        if kp.shape != ti.shape:
            raise ValueError(f"kp {kp.shape} and ti {ti.shape} must match")
        worst = None
        for p in self.plants:
            out = []
            for s in range(0, kp.shape[0], self.chunk):
                out.append(np.asarray(self._kernel(
                    p, self.ref_pre, self.ref_post,
                    jnp.asarray(kp[s:s + self.chunk]), jnp.asarray(ti[s:s + self.chunk]),
                    self.noise)))
            rows = np.concatenate(out, axis=0)
            if worst is None:
                worst = rows
            else:   # keep the whole row of whichever roll position scored worst
                take = rows[:, 0] > worst[:, 0]
                worst = np.where(take[:, None], rows, worst)
        return worst

    def costs(self, kp, ti) -> np.ndarray:
        s = self.evaluate(kp, ti)[:, 0]
        return np.where(np.isfinite(s), s, np.inf)

    def cost(self, kp, ti) -> float:
        return float(self.costs(kp, ti)[0])

    def breakdown(self, kp, ti) -> dict[str, float | None]:
        row = self.evaluate(kp, ti)[0]
        return {n: (float(v) if np.isfinite(v) else None) for n, v in zip(OUTPUTS, row)}
