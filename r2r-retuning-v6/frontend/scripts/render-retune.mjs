// Server-side render smoke test for RetunePanel against a LIVE /retune body.
//   node scripts/render-retune.mjs     (needs the backend on 127.0.0.1:8024)
// Fails on any NaN/undefined/Infinity reaching the markup, or a thrown render.
import { createServer } from 'vite';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const RetunePanel = (await server.ssrLoadModule('/src/tabs/RetunePanel.jsx')).default;

const runSpec = {
  plant: { source: 'preset', preset_id: 'P01' },
  drift: { EA_pct: 0, J_UW_pct: -30, J_Nip_pct: 0, J_RW_pct: 50, f_pct: 0 },
  protocol: { T_log_ms: 5, excitation: 'E_Toggle', record_s: 16, pct_T: 0.003, pct_v: 0.003,
              LPF_T_hz: 50, LPF_v_hz: 50, Kp_star: 100, seed: 0 },
};
const post = async (body) => (await fetch('http://127.0.0.1:8024/retune', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})).json();
const live = await post({ run_spec: runSpec });
const liveBo = await post({ run_spec: runSpec, bo_refine: true });
console.log(`  live retune: status=${live.status} Kp*=${live.gains?.delivered?.kp_star} TI=${live.gains?.delivered?.ti_s}`);

const base = { busy: false, onRun: () => {}, boRefine: false, onToggleBo: () => {}, draftChanged: false };
const cases = [
  ['nothing yet', { ...base, result: null }],
  ['busy', { ...base, result: null, busy: true }],
  ['live result', { ...base, result: live }],
  ['live result, BO refine', { ...base, result: liveBo, boRefine: true }],
  ['stale after a draft change', { ...base, result: live, draftChanged: true }],
  ['identification failed', { ...base, result: { status: 'identification_failed',
                              identification: { converged: false, failure: 'ValueError: diverged' } } }],
  ['no T_I profile', { ...base, result: { ...live, ti_profile: [] } }],
  ['diverging gain set', { ...base, result: { ...live, gains: { ...live.gains,
      simc_reference: { ...live.gains.simc_reference,
                        on_plant: { S: null, RMSE_y_N: null, OS_percent: null, t_s_s: null, finite: false } } } } }],
];

let fail = 0;
if (live.status !== 'ok') { console.log('  FAIL  live /retune did not return ok'); fail++; }
for (const [name, props] of cases) {
  try {
    const html = renderToStaticMarkup(React.createElement(RetunePanel, props));
    if (/NaN|undefined|Infinity/.test(html)) {
      console.log(`  FAIL  ${name}: ${(html.match(/.{0,60}(NaN|undefined|Infinity).{0,60}/) || [''])[0]}`);
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
console.log(fail ? `\n${fail} case(s) failed` : '\nall retune render cases passed');
process.exit(fail ? 1 : 0);
