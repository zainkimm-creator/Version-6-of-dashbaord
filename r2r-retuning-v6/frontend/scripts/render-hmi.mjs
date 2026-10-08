// SSR smoke test for the machine screen's canvas against a LIVE /plant/trace.
//   node scripts/render-hmi.mjs      (needs the backend on 127.0.0.1:8024)
import { createServer } from 'vite';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const MachineCanvas = (await server.ssrLoadModule('/src/tabs/MachineCanvas.jsx')).default;
const run = (excitation, record_s, extra = {}) => ({ run_spec: {
  plant: { source: 'preset', preset_id: 'P01' },
  drift: { EA_pct: 0, J_UW_pct: 0, J_Nip_pct: 0, J_RW_pct: 0, f_pct: 0 },
  protocol: { T_log_ms: 5, excitation, record_s, pct_T: 0.003, pct_v: 0.003,
              LPF_T_hz: 50, LPF_v_hz: 50, Kp_star: 100, seed: 0 } }, ...extra });
const post = async (body) => {
  const r = await fetch('http://127.0.0.1:8024/plant/trace', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const toggle = await post(run('E_Toggle', 16));
const ev1 = await post(run('EV1', 12));
console.log(`  trace E_Toggle: ${toggle.samples} samples, dt ${toggle.dt_s} s`);
const mid = Math.floor(toggle.samples / 3);
const cases = [
  ['no trace', { trace: null, v0: 0.5 }],
  ['E_Toggle start', { trace: toggle, frame: 0, v0: 0.5 }],
  ['E_Toggle mid, stepped', { trace: toggle, frame: mid, v0: 0.5, fraction: 0.33 }],
  ['held (no motion)', { trace: toggle, frame: mid, playing: false, v0: 0.5 }],
  ['EV1 last frame', { trace: ev1, frame: ev1.samples - 1, v0: 0.5, fraction: 0.99 }],
  ['with twin', { trace: { ...toggle, twin_T_N: toggle.T_N }, frame: mid, v0: 0.5 }],
  ['frame out of range', { trace: toggle, frame: 99999, v0: 3 }],
];
let fail = 0;
for (const [name, props] of cases) {
  try {
    const html = renderToStaticMarkup(React.createElement(MachineCanvas, props));
    if (/NaN|undefined|Infinity/.test(html)) {
      console.log(`  FAIL  ${name}: ${(html.match(/.{0,50}(NaN|undefined|Infinity).{0,50}/) || [''])[0]}`); fail++;
    } else console.log(`  ok    ${name}  (${html.length} chars)`);
  } catch (e) { console.log(`  FAIL  ${name}: ${e.message}`); fail++; }
}
await server.close();
console.log(fail ? `\n${fail} case(s) failed` : '\nall HMI render cases passed');
process.exit(fail ? 1 : 0);
