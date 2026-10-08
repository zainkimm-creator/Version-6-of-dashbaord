// Render the diagram's SVG with real plant data, standalone, so it can be
// rasterised and eyeballed without a browser:
//
//   node scripts/render-svg.mjs P01 /tmp/d.svg && rsvg-convert -w 1600 -b white /tmp/d.svg -o /tmp/d.png
//
// Note rsvg applies CSS far less completely than a browser does, so treat the
// output as a layout check only; for colour and final appearance screenshot the
// real page. Inlines the CSS the SVG depends on, because
// rsvg-convert sees no stylesheet.
import { createServer } from 'vite';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';
import { writeFileSync, readFileSync } from 'node:fs';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
const { default: SpanDiagram } = await server.ssrLoadModule('/src/tabs/SpanDiagram.jsx');

const post = async (path, body) =>
  (await fetch(`http://127.0.0.1:8024${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) })).json();

const plantId = process.argv[2] || 'P01';
const spec = {
  plant: { source: 'preset', preset_id: plantId },
  drift: { EA_pct: 0, J_UW_pct: 0, J_Nip_pct: 0, J_RW_pct: 0, f_pct: 0 },
  protocol: { T_log_ms: 5, excitation: 'E_Toggle', record_s: 16, pct_T: 0.003,
              pct_v: 0.003, LPF_T_hz: 50, LPF_v_hz: 50, Kp_star: 100, seed: 0 },
};
const derive = await post('/plant/derive', { run_spec: spec });
const ident = await post('/identify', { commit: false, run_spec: spec });

const html = renderToStaticMarkup(React.createElement(SpanDiagram, {
  readout: derive, twinRows: ident.rows, protocol: spec.protocol,
}));
const svg = html.match(/<svg[\s\S]*?<\/svg>/)[0];
const css = readFileSync('./src/styles.css', 'utf8');
const marker = '   Three-span diagram (Twin Study)';
const diagCss = css.slice(css.lastIndexOf('/*', css.indexOf(marker)));

const out = svg.replace(
  '<defs>',
  `<style>svg{background:#fff;color:#1b1b1b;font-family:system-ui,sans-serif}${diagCss}</style><defs>`,
);
writeFileSync(process.argv[3] || 'diagram.svg', out);
await server.close();
console.log(`wrote ${process.argv[3] || 'diagram.svg'} for ${plantId}`);
