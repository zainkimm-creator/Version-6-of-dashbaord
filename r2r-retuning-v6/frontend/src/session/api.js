import { apiGet, apiPost } from '../api.js';

async function apiSend(baseUrl, method, path, body = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

export function sessionApi(baseUrl) {
  return {
    getSession: () => apiGet(baseUrl, '/session'),
    getPresets: () => apiGet(baseUrl, '/session/presets'),
    putPlant: (plant) => apiSend(baseUrl, 'PUT', '/session/plant', plant),
    putDrift: (drift) => apiSend(baseUrl, 'PUT', '/session/drift', drift),
    patchProtocol: (patch) => apiSend(baseUrl, 'PATCH', '/session/protocol', patch),
    putTwin: (body) => apiSend(baseUrl, 'PUT', '/session/twin', body),
    resetSession: () => apiPost(baseUrl, '/session/reset', {}),
    derive: (runSpec, swapWithEstimates = null) =>
      apiPost(baseUrl, '/plant/derive', {
        run_spec: runSpec,
        swap_with_estimates: swapWithEstimates,
      }),
    identify: (runSpec, commit = false) =>
      apiPost(baseUrl, '/identify', { run_spec: runSpec, commit }),
    paperCheck: (runSpec) => apiPost(baseUrl, '/identify/paper-check', { run_spec: runSpec }),
    retune: (runSpec, options = {}) => apiPost(baseUrl, '/retune', { run_spec: runSpec, ...options }),
    // the six per-zone gains saved for a plant (stage 1 of the two-stage retune)
    zoneGains: (runSpec) => apiPost(baseUrl, '/zone-gains', { run_spec: runSpec }),
    forgetZoneGains: (runSpec) => apiPost(baseUrl, '/zone-gains/delete', { run_spec: runSpec }),
    trace: (runSpec, estimates = null) => apiPost(baseUrl, '/plant/trace', { run_spec: runSpec, estimates }),
    // The cost test's +20 % step with given gains, on the plant or (with θ̂) the twin.
    baseline: (runSpec, costTier = null, opts = {}) => apiPost(baseUrl, '/plant/baseline', { run_spec: runSpec, cost_tier: costTier, ...opts }),
    step: (runSpec, gains, estimates = null, opts = {}) =>
      apiPost(baseUrl, '/plant/step', { run_spec: runSpec, kp_star: gains.kp_star, ti_s: gains.ti_s, estimates, ...opts }),
    getAtlas: (plantId, protocol) => {
      const q = new URLSearchParams({
        plant_id: plantId,
        T_log_ms: protocol.T_log_ms,
        excitation: protocol.excitation,
        record_s: protocol.record_s,
        pct_T: protocol.pct_T,
        pct_v: protocol.pct_v,
        // A null LPF is a real protocol -- no filter -- so it is sent as the
        // word 'none' rather than dropped. Omitting the key would let the
        // server apply its 50 Hz default and answer a no-filter protocol with
        // the 50 Hz cell; the empty string used to arrive as a 422.
        LPF_T_hz: protocol.LPF_T_hz ?? 'none',
        LPF_v_hz: protocol.LPF_v_hz ?? 'none',
        Kp_star: protocol.Kp_star,
        seed: protocol.seed,
      });
      return apiGet(baseUrl, `/atlas?${q.toString()}`);
    },
  };
}
