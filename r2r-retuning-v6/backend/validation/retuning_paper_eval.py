"""The Section 4.2 retuning cost, measured the way the paper's authors measure it.

Source: the authors' reply of 2026-09-15 (`../FE sample/Q1a.pdf`), items Q1a/A1,
D4 and E5. The paper does not print this experiment; the reply does:

* fixed-step RK4 at dt = 1 ms, the cascade controller updated every step;
* a 30 s horizon, the loop initialised at the equilibrium of the pre-step
  reference, and at t = 5 s all three tension references stepped together from
  T_ref to 1.2 T_ref, so |dT_ref,i| = 0.2 T_ref on every channel;
* RMSE_y, OS and t_s formed over the 25 s after the step on the TRUE span
  tensions: RMSE_y = mean of the three per-channel RMS errors; OS = max over
  channels of the peak excursion above the new reference / that channel's step;
  t_s = time from the step to the last sample at which any channel is outside
  its 2 % band (uncapped);
* no sensor noise, no filter, no integrator clamp; only (K_p*, T_I) searched;
  K_vel,i = 1.4 J_i sqrt(EA R_i^2 / (J_i L_i)) from the simulated plant;
* the feed-forward friction term uses the measured (here: true) roller speed,
  and the feed-forward speed reference is held at the POST-step steady state
  for the whole run (the pre-step 5 s absorbs the offset, outside the metrics).

S = (RMSE_y / 3 N)^2 + 2 (OS / 20 %)^2 + (t_s / 3 s)^2   (paper Eq. 12)

The kernel also returns the peak |integrator| so the reply's "a +-200 N s clamp
would be inert (peak 20 N s at the delivered gains)" can be checked.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .retuning_jax import pack_plant

PAPER_EVAL_KEY = "author-step3-30s-noclamp-ffmeas-v1"
DT_S = 0.001
T_SIM_S = 30.0
T_STEP_S = 5.0
STEP_FRACTION = 0.20
S_E_N = 3.0
S_OS_PCT = 20.0
S_T_S = 3.0
W_OS = 2.0
BAND_FRACTION = 0.02
N_STEPS = int(round(T_SIM_S / DT_S))
STEP_INDEX = int(round(T_STEP_S / DT_S))
OUTPUTS = ("S", "RMSE_y_N", "OS_percent", "t_s_s", "integrator_peak_Ns")


def _jax():
    """JAX in double precision: float32 moves S in the 4th significant digit."""

    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def _make_cost(n_steps: int, step_index: int, dt: float):
    """The per-candidate cost function (unjitted)."""

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
            correction = (L / EA) * kp_star * (polarity * err + integ / ti)
            w_ref = (v_post + correction) / R
            tau_web = R * jnp.stack([-T[0], T[1] - T[2], T[2]])
            u = k_vel * (w_ref - w) + f * w - tau_web
            return (rk4(x, u, p), integ, peak), x[:3]

        (x_end, _, peak), states = jax.lax.scan(
            body, (x0, jnp.zeros(3), jnp.asarray(0.0)), jnp.arange(n_steps))
        # states[k] is the tension at time k*dt; append the final state.
        trace = jnp.concatenate([states, x_end[:3][None, :]], axis=0)
        after = trace[step_index + 1:]            # the 25 s after the step
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
    """JIT batch kernel for one plant: (p, ref_pre, ref_post, kp[n], ti[n]) -> (n, 5)."""

    jax = _jax()
    return jax.jit(jax.vmap(_make_cost(n_steps, step_index, dt), in_axes=(None, None, None, 0, 0)))


@lru_cache(maxsize=4)
def build_rows_kernel(n_steps: int = N_STEPS, step_index: int = STEP_INDEX, dt: float = DT_S):
    """JIT kernel where every argument varies per row: (p[n], pre[n], post[n], kp[n], ti[n]) -> (n, 5).

    One GPU call can then score candidates from different plants. The GPU cost
    of this experiment is nearly independent of batch size, so batching pending
    evaluations across a whole campaign is what makes sequential BO affordable.
    """

    jax = _jax()
    return jax.jit(jax.vmap(_make_cost(n_steps, step_index, dt), in_axes=(0, 0, 0, 0, 0)))


class PaperEvaluator:
    """Author-spec Eq. (12) for one plant; T_I in absolute seconds."""

    def __init__(self, params, line_speed_m_s: float, *, chunk: int = 1024):
        _jax()
        import jax.numpy as jnp

        self.p = jnp.asarray(pack_plant(params, line_speed_m_s))
        pre = np.asarray(params.tension_ref_N, dtype=np.float64)
        self.ref_pre = jnp.asarray(pre)
        self.ref_post = jnp.asarray(pre * (1.0 + STEP_FRACTION))
        self.chunk = int(chunk)
        self._kernel = build_kernel()

    def evaluate(self, kp, ti) -> np.ndarray:
        import jax.numpy as jnp

        kp = np.atleast_1d(np.asarray(kp, dtype=np.float64))
        ti = np.atleast_1d(np.asarray(ti, dtype=np.float64))
        out = []
        for start in range(0, kp.size, self.chunk):
            out.append(np.asarray(self._kernel(self.p, self.ref_pre, self.ref_post,
                                               jnp.asarray(kp[start:start + self.chunk]),
                                               jnp.asarray(ti[start:start + self.chunk]))))
        return np.concatenate(out, axis=0)

    def costs(self, kp, ti) -> np.ndarray:
        s = self.evaluate(kp, ti)[:, 0]
        return np.where(np.isfinite(s), s, np.inf)

    def cost(self, kp: float, ti: float) -> float:
        return float(self.costs([kp], [ti])[0])

    def breakdown(self, kp: float, ti: float) -> dict[str, float | None]:
        row = self.evaluate([kp], [ti])[0]
        return {name: (float(v) if np.isfinite(v) else None) for name, v in zip(OUTPUTS, row)}
