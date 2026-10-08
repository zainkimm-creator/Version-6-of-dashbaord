"""T0: the Eq. (12) retuning cost with PER-ROLLER gains.

Copied VERBATIM (bar this note and the repo-root depth) from the study that
proved it equal to backend.validation.retuning_paper_eval at 0 ULP. It is the
T0 tier: reimplementing it here a second time reproduced the paper only to
5.7e-14, because extra scan carries change how XLA fuses the 30,000-step
loop. Delegating instead of copying makes T0 the authors' arithmetic by
construction rather than by vigilance.

The paper (and the dashboard) search a single normalised pair (K_p*, T_I) shared
by all three tension loops:

    correction[i] = (L[i]/EA) * K_p*      * (e[i] + I[i]/T_I)        <- shared

A real line commissions each loop separately, so here every roller carries its
own pair:

    correction[i] = (L[i]/EA) * K_p*[i]   * (e[i] + I[i]/T_I[i])     <- per roller

Everything else is byte-for-byte the authors' experiment: RK4 at 1 ms over 30 s,
all three references stepped +20 % together at t = 5 s, metrics over the
following 25 s on the true tensions, no noise, no filter, no integrator clamp,
K_vel,i = 1.4 J_i sqrt(EA R_i^2 / (J_i L_i)).

The shared-pair case is the exact special case K_p*[0]=K_p*[1]=K_p*[2] and
T_I[0]=T_I[1]=T_I[2]; `tests/test_equivalence.py` pins that to 0 ULP so the
6-D results are directly comparable to the 2-D campaign.

Nothing in the repository outside per-roller-retuning-2026-09-29/ is modified.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from backend.validation.retuning_jax import pack_plant                      # noqa: E402
from backend.validation.retuning_paper_eval import (                        # noqa: E402
    BAND_FRACTION, DT_S, N_STEPS, OUTPUTS, S_E_N, S_OS_PCT, S_T_S,
    STEP_FRACTION, STEP_INDEX, W_OS,
)

PER_ROLLER_EVAL_KEY = "per-roller-step3-30s-noclamp-ffmeas-v1"
N_GAINS = 3


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def _make_cost(n_steps: int, step_index: int, dt: float):
    """Per-candidate cost. `kp_star` and `ti` are length-3 vectors, one per roller.

    This is the authors' kernel with exactly one change: the two gain scalars
    became 3-vectors. The control expression already multiplied element-wise by
    the per-span length L, so it broadcasts unchanged.
    """

    jax = _jax()
    import jax.numpy as jnp

    polarity = jnp.array([-1.0, 1.0, 1.0])

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

    def cost(p, ref_pre, ref_post, kp_star, ti):
        EA, L, R, J, f, v_feed = unpack(p)
        step = ref_post - ref_pre
        v_post = steady_v(p, ref_post)
        k_vel = 1.4 * J * jnp.sqrt(EA * R * R / (J * L))
        x0 = jnp.concatenate([ref_pre, steady_v(p, ref_pre) / R])

        def body(carry, k):
            x, integ, peak = carry
            T, w = x[:3], x[3:]
            target = jnp.where(k >= step_index, ref_post, ref_pre)
            err = target - T
            integ = integ + polarity * err * dt
            peak = jnp.maximum(peak, jnp.max(jnp.abs(integ)))
            # the only line that differs from the shared-pair kernel
            correction = (L / EA) * kp_star * (polarity * err + integ / ti)
            w_ref = (v_post + correction) / R
            tau_web = R * jnp.stack([-T[0], T[1] - T[2], T[2]])
            u = k_vel * (w_ref - w) + f * w - tau_web
            return (rk4(x, u, p), integ, peak), x[:3]

        (x_end, _, peak), states = jax.lax.scan(
            body, (x0, jnp.zeros(3), jnp.asarray(0.0)), jnp.arange(n_steps))
        trace = jnp.concatenate([states, x_end[:3][None, :]], axis=0)
        after = trace[step_index + 1:]
        err = after - ref_post
        rmse = jnp.mean(jnp.sqrt(jnp.mean(err ** 2, axis=0)))
        overshoot = 100.0 * jnp.max(jnp.maximum(jnp.max(err, axis=0), 0.0) / jnp.abs(step))
        outside = jnp.any(jnp.abs(err) > BAND_FRACTION * jnp.abs(step), axis=1)
        idx = jnp.arange(after.shape[0])
        last = jnp.max(jnp.where(outside, idx, -1))
        settling = jnp.where(last < 0, 0.0, (last + 1) * dt)
        S = (rmse / S_E_N) ** 2 + W_OS * (overshoot / S_OS_PCT) ** 2 + (settling / S_T_S) ** 2
        return jnp.stack([S, rmse, overshoot, settling, peak])

    return cost


@lru_cache(maxsize=4)
def build_kernel(n_steps: int = N_STEPS, step_index: int = STEP_INDEX, dt: float = DT_S):
    """One plant, many candidates: (p, pre, post, kp[n,3], ti[n,3]) -> (n, 5)."""

    jax = _jax()
    return jax.jit(jax.vmap(_make_cost(n_steps, step_index, dt), in_axes=(None, None, None, 0, 0)))


@lru_cache(maxsize=4)
def build_rows_kernel(n_steps: int = N_STEPS, step_index: int = STEP_INDEX, dt: float = DT_S):
    """Every argument varies per row, so one GPU call can mix plants."""

    jax = _jax()
    return jax.jit(jax.vmap(_make_cost(n_steps, step_index, dt), in_axes=(0, 0, 0, 0, 0)))


class PerRollerEvaluator:
    """Eq. (12) for one plant with a (3,) gain vector per roller."""

    def __init__(self, params, line_speed_m_s: float, *, chunk: int = 2048):
        _jax()
        import jax.numpy as jnp

        self.p = jnp.asarray(pack_plant(params, line_speed_m_s))
        pre = np.asarray(params.tension_ref_N, dtype=np.float64)
        self.ref_pre = jnp.asarray(pre)
        self.ref_post = jnp.asarray(pre * (1.0 + STEP_FRACTION))
        self.chunk = int(chunk)
        self._kernel = build_kernel()

    @staticmethod
    def _as_batch(g) -> np.ndarray:
        """Accept (3,), (n,3) or a scalar broadcast to all three rollers."""
        g = np.asarray(g, dtype=np.float64)
        if g.ndim == 0:
            g = np.full((1, N_GAINS), float(g))
        elif g.ndim == 1:
            g = (np.full((1, N_GAINS), g[0]) if g.size == 1 else g.reshape(1, N_GAINS))
        return np.ascontiguousarray(g, dtype=np.float64)

    def evaluate(self, kp, ti) -> np.ndarray:
        import jax.numpy as jnp

        kp, ti = self._as_batch(kp), self._as_batch(ti)
        if kp.shape != ti.shape:
            raise ValueError(f"kp {kp.shape} and ti {ti.shape} must match")
        out = []
        for s in range(0, kp.shape[0], self.chunk):
            out.append(np.asarray(self._kernel(self.p, self.ref_pre, self.ref_post,
                                               jnp.asarray(kp[s:s + self.chunk]),
                                               jnp.asarray(ti[s:s + self.chunk]))))
        return np.concatenate(out, axis=0)

    def costs(self, kp, ti) -> np.ndarray:
        s = self.evaluate(kp, ti)[:, 0]
        return np.where(np.isfinite(s), s, np.inf)

    def cost(self, kp, ti) -> float:
        return float(self.costs(kp, ti)[0])

    def breakdown(self, kp, ti) -> dict[str, float | None]:
        row = self.evaluate(kp, ti)[0]
        return {n: (float(v) if np.isfinite(v) else None) for n, v in zip(OUTPUTS, row)}
