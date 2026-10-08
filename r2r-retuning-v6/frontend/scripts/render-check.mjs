// Server-side render smoke test for SpanDiagram: proves the component RUNS,
// not merely that it compiles. `npm run build` catches syntax and imports; this
// catches the things that only surface with real data -- a missing array, a
// divide by zero, an unknown excitation -- and fails on any NaN/undefined that
// reaches the markup.
//
//   npm run render-check          (needs the backend up on 127.0.0.1:8024) Uses vite's own SSR pipeline so JSX and the
// project's resolution rules apply exactly as they do in the browser.
import { createServer } from 'vite';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const mod = await server.ssrLoadModule('/src/tabs/SpanDiagram.jsx');
const SpanDiagram = mod.default;

const derive = await (await fetch('http://127.0.0.1:8024/plant/derive', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ run_spec: {
    plant: { source: 'preset', preset_id: 'P01' },
    drift: { EA_pct: 0, J_UW_pct: 0, J_Nip_pct: 0, J_RW_pct: 0, f_pct: 0 },
    protocol: { T_log_ms: 5, excitation: 'E_Toggle', record_s: 16, pct_T: 0.003,
                pct_v: 0.003, LPF_T_hz: 50, LPF_v_hz: 50, Kp_star: 100, seed: 0 } } }),
})).json();

const ident = await (await fetch('http://127.0.0.1:8024/identify', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ commit: false, run_spec: {
    plant: { source: 'preset', preset_id: 'P01' },
    drift: { EA_pct: 0, J_UW_pct: 0, J_Nip_pct: 0, J_RW_pct: 0, f_pct: 0 },
    protocol: { T_log_ms: 5, excitation: 'E_Toggle', record_s: 16, pct_T: 0.003,
                pct_v: 0.003, LPF_T_hz: 50, LPF_v_hz: 50, Kp_star: 100, seed: 0 } } }),
})).json();

const protocol = { T_log_ms: 5, excitation: 'E_Toggle', pct_T: 0.003, pct_v: 0.003,
                   LPF_T_hz: 50, LPF_v_hz: 50 };

const cases = [
  ['no readout at all', { readout: null, twinRows: null, protocol }],
  ['plant only, no twin', { readout: derive, twinRows: null, protocol }],
  ['plant + twin', { readout: derive, twinRows: ident.rows, protocol }],
  ['EV1 (no tension step)', { readout: derive, twinRows: ident.rows,
                              protocol: { ...protocol, excitation: 'EV1' } }],
  ['unknown excitation', { readout: derive, twinRows: ident.rows,
                           protocol: { ...protocol, excitation: 'NOPE' } }],
  ['missing geometry', { readout: { ...derive, R_m: undefined, L_m: undefined,
                                    J_kg_m2: [], f_Nms_per_rad: [] },
                         twinRows: ident.rows, protocol }],
  ['zero noise', { readout: { ...derive, sigma_T_N: 0 }, twinRows: ident.rows,
                   protocol: { ...protocol, pct_T: 0 } }],
  ['not converged', { readout: derive, twinRows: [], protocol }],
];

let fail = 0;
for (const [name, props] of cases) {
  try {
    const html = renderToStaticMarkup(React.createElement(SpanDiagram, props));
    if (/NaN|undefined|Infinity/.test(html)) {
      console.log(`  FAIL  ${name}: markup contains NaN/undefined/Infinity`);
      console.log('        ' + (html.match(/.{0,60}(NaN|undefined|Infinity).{0,60}/) || [''])[0]);
      fail++;
    } else {
      console.log(`  ok    ${name}  (${html.length} chars)`);
    }
  } catch (e) {
    console.log(`  FAIL  ${name}: ${e.message}`);
    fail++;
  }
}
await server.close();
console.log(fail ? `\n${fail} case(s) failed` : '\nall render cases passed');
process.exit(fail ? 1 : 0);
