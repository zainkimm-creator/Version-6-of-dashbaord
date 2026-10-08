// Algorithm 1, Steps 4-7, computed live for the current protocol and drift.
//
// The gain atlas only answers on its grid. This panel asks POST /retune, which
// identifies the (drifted) plant, searches the digital twin with the paper's
// 2,805-point hierarchical grid search, and scores the gains back on the plant.
// Every number here comes off that response; nothing is computed in the browser.

const GAIN_ROWS = [
  ['delivered', 'Delivered — use these'],
  ['recommended', 'HGS-only, T_I valley tie-break'],
  ['hgs_only', 'HGS-only, Eq. (12) optimum as searched'],
  ['hgs_bo', 'HGS + BO(5) on the plant'],
  ['commissioned', 'Commissioned (pre-drift) gains'],
  ['reference_on_plant', 'True-plant optimum — unavailable on a real line'],
  ['simc_reference', 'SIMC T_I — reference only'],
];

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

const ZONE_FALLBACK = ['UW', 'OutFeeder', 'RW'];

// A gain set now carries one pair per TENSION zone (UW, out-feeder nip, RW); the
// in-feeder is the velocity master and has none. When all three share a value the
// backend also sends the plain scalar, so the common case still reads as one
// number and only genuinely per-zone gains expand into three labelled rows.
function perZone(scalar, perZoneValues, zones, digits = 3) {
  if (scalar !== null && scalar !== undefined) return fmt(scalar, digits);
  if (!Array.isArray(perZoneValues)) return '—';
  const names = Array.isArray(zones) && zones.length === perZoneValues.length ? zones : ZONE_FALLBACK;
  return (
    <div className="per-zone">
      {perZoneValues.map((v, i) => (
        <div key={names[i] ?? i}>
          <span className="hint">{names[i] ?? i}</span> {fmt(v, digits)}
        </div>
      ))}
    </div>
  );
}

// `on_bound` is a list of three (one per zone), each null / 'lower' / 'upper'.
// A bare array is always truthy in JS, so it must never be tested directly.
function boundNote(label, onBound, zones) {
  const list = Array.isArray(onBound) ? onBound : [onBound];
  const names = Array.isArray(zones) && zones.length === list.length ? zones : ZONE_FALLBACK;
  const hits = list.map((b, i) => (b ? `${names[i] ?? i} ${b}` : null)).filter(Boolean);
  if (!hits.length) return '';
  const every = hits.length === list.length && new Set(list).size === 1;
  return every ? `${label} on ${list[0]} bound` : `${label} on bound: ${hits.join(', ')}`;
}

function StepChip({ status }) {
  const cls = status === 'pass' ? 'chip chip-pass' : status === 'fail' ? 'chip chip-warn' : 'chip chip-stale';
  return <span className={cls}>{status === 'unknown' ? 'n/a' : status}</span>;
}

// S on the twin against T_I at the optimal K_p*. Log-log: the valley spans
// decades, and the paper's cost is nearly flat across most of it.
function TiProfileChart({ profile, chosenTi, optimumTi }) {
  const points = (profile ?? []).filter((p) => p.S_twin !== null && p.S_twin > 0 && p.ti_s > 0);
  if (points.length < 2) return null;
  const W = 560; const H = 200; const pad = { l: 52, r: 12, t: 12, b: 34 };
  const xs = points.map((p) => Math.log10(p.ti_s));
  const ys = points.map((p) => Math.log10(p.S_twin));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const sy = (v) => H - pad.b - ((v - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b);
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${sx(xs[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ');
  const decades = [];
  for (let d = Math.ceil(x0); d <= Math.floor(x1); d += 1) decades.push(d);
  const marker = (ti, label, cls) => (ti > 0 ? (
    <g className={cls}>
      <line x1={sx(Math.log10(ti))} x2={sx(Math.log10(ti))} y1={pad.t} y2={H - pad.b} />
      {/* Labels in the right third sit left of their line, or the edge clips them. */}
      <text x={sx(Math.log10(ti)) + (sx(Math.log10(ti)) > W * 0.66 ? -4 : 4)} y={pad.t + 12}
            textAnchor={sx(Math.log10(ti)) > W * 0.66 ? 'end' : 'start'}>{label}</text>
    </g>
  ) : null);
  return (
    <div className="chart-frame retune-chart" role="img"
         aria-label="Retuning cost on the twin against integral time at the optimal gain">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        <line className="axis" x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} />
        <line className="axis" x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} />
        {decades.map((d) => (
          <text key={d} className="tick" x={sx(d)} y={H - pad.b + 14} textAnchor="middle">
            {d >= 0 ? `${10 ** d}` : `1e${d}`}
          </text>
        ))}
        <text className="tick" x={pad.l - 6} y={sy(y0)} textAnchor="end">{fmt(10 ** y0, 3)}</text>
        <text className="tick" x={pad.l - 6} y={sy(y1) + 8} textAnchor="end">{fmt(10 ** y1, 2)}</text>
        <text className="tick" x={(W + pad.l) / 2} y={H - 4} textAnchor="middle">T_I (s), log scale</text>
        <path className="series" d={path} />
        {marker(optimumTi, 'Eq. (12) optimum', 'marker-optimum')}
        {marker(chosenTi, 'delivered T_I', 'marker-chosen')}
      </svg>
    </div>
  );
}

export default function RetunePanel({ result, busy, onRun, draftChanged }) {
  const gains = result?.gains;
  const delivered = gains?.delivered;
  const resolution = gains?.recommended?.ti_resolution;

  return (
    <div className="panel retune-panel">
      <h3>Retune — Algorithm 1, computed live</h3>
      <p className="hint">
        Identifies the plant under the protocol above (with any drift), searches the digital twin
        with the paper&rsquo;s 2,805-point hierarchical grid search, and checks the result on the plant.
        Works for any protocol, drift or edited plant — no precomputed cell is needed. First run
        takes roughly 20–60 s; repeats come from cache.
      </p>
      <div className="button-row">
        <button type="button" className="primary" disabled={busy} onClick={onRun}>
          {busy ? 'Retuning…' : 'Recompute'}
        </button>
      </div>
      {draftChanged && result && (
        <div className="chip chip-stale">
          The protocol, plant or drift changed after this retune; press Compute again.
        </div>
      )}

      {result?.status === 'identification_failed' && (
        <p className="chip chip-warn">
          Identification failed ({result.identification?.failure ?? 'the estimator did not converge'}).
          No gains are reported: no twin, no retune.
        </p>
      )}

      {result?.status === 'ok' && (
        <>
          <dl className="readout retune-headline">
            <dt>K_p*</dt><dd><strong>{fmt(delivered?.kp_star, 3)}</strong> 1/s</dd>
            <dt>T_I</dt><dd><strong>{fmt(delivered?.ti_s, 3)}</strong> s</dd>
            <dt>K_i = K_p*/T_I</dt><dd>{fmt(delivered ? delivered.kp_star / delivered.ti_s : null, 5)}</dd>
            <dt>Cost S on plant</dt><dd>{fmt(delivered?.on_plant?.S, 4)}</dd>
            <dt>vs true-plant optimum</dt>
            <dd>{fmt(result.ratio_vs_reference, 3)}<span className="hint"> — 1.00 means as good as knowing the plant</span></dd>
            <dt>Source</dt><dd>{delivered?.source}</dd>
          </dl>

          <h4>Algorithm 1 checklist</h4>
          <ol className="retune-steps">
            {result.steps.map((step) => (
              <li key={step.step}>
                <StepChip status={step.status} /> <strong>Step {step.step}</strong> {step.name}
                <div className="hint">{step.detail}</div>
              </li>
            ))}
          </ol>

          <h4>Gain sets, all scored on the plant (Eq. 12, step-response episode)</h4>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr><th>gain set</th><th>K_p*</th><th>T_I (s)</th><th>S</th><th>RMSE_y (N)</th><th>OS (%)</th><th>t_s (s)</th></tr>
              </thead>
              <tbody>
                {GAIN_ROWS.map(([key, label]) => {
                  const g = gains?.[key];
                  if (!g) return null;
                  const p = g.on_plant ?? {};
                  const bound = [boundNote('K_p*', g.kp_on_bound, g.zones),
                    boundNote('T_I', g.ti_on_bound, g.zones)].filter(Boolean).join('; ');
                  return (
                    <tr key={key} className={key === 'delivered' ? 'row-strong' : ''}>
                      <td>{label}{bound && <div className="hint">{bound}</div>}</td>
                      <td>{perZone(g.kp_star, g.kp_star_per_zone, g.zones, 3)}</td>
                      <td>{perZone(g.ti_s, g.ti_s_per_zone, g.zones, 3)}</td>
                      <td>{fmt(p.S, 4)}</td>
                      <td>{fmt(p.RMSE_y_N, 3)}</td>
                      <td>{fmt(p.OS_percent, 2)}</td>
                      <td>{fmt(p.t_s_s, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="two-column">
            <dl className="readout">
              <dt>Twin MARE_θ</dt><dd>{fmt(result.identification?.MARE_theta_pct, 3)} %</dd>
              <dt>Twin prediction RMSE</dt><dd>{fmt(result.twin_validation?.rmse_N, 4)} N</dd>
              <dt>Twin fit (NRMSE)</dt><dd>{fmt(result.twin_validation?.fit_percent, 1)} %</dd>
              <dt>ε (pre-drift twin)</dt><dd>{fmt(result.twin_validation?.epsilon_N, 4)} N × {fmt(result.twin_validation?.margin, 2)}</dd>
            </dl>
            <dl className="readout">
              <dt>C_target (pre-drift S)</dt><dd>{fmt(result.acceptance?.C_target, 4)} × {fmt(result.acceptance?.margin, 2)}</dd>
              <dt>S delivered</dt><dd>{fmt(result.acceptance?.S_delivered, 4)}</dd>
              <dt>Search</dt>
              <dd>{result.search?.budget} evals · K_p* ∈ [{result.search?.kp_bounds?.join(', ')}] · T_I ∈ [{result.search?.ti_bounds_s?.join(', ')}] s (authors&rsquo; box)</dd>
              <dt>Run</dt><dd>{result.cached ? 'from cache' : `${fmt(result.seconds, 1)} s`}</dd>
            </dl>
          </div>

          <h4>Why this T_I</h4>
          <p className="hint">
            Eq. (12) scores a set-point step, which barely sees integral action, so its cost is nearly
            flat over a wide T_I range. The delivered T_I is the smallest in the continuous valley
            within {fmt(100 * (resolution?.tolerance ?? 0), 1)} % of the best cost
            (valley {fmt(resolution?.valley_ti_s?.[0], 2)}–{fmt(resolution?.valley_ti_s?.[1], 1)} s);
            more integral action rejects load disturbances better at no step-cost penalty.
          </p>
          <TiProfileChart
            profile={result.ti_profile}
            chosenTi={gains?.recommended?.ti_s}
            optimumTi={gains?.hgs_only?.ti_s}
          />

          <details className="retune-assumptions">
            <summary>Assumptions for what the paper does not state ({result.assumptions?.length})</summary>
            <ul>
              {(result.assumptions ?? []).map((a) => (
                <li key={a.id}><strong>{a.id} {a.what}:</strong> {a.value}. <span className="hint">{a.why}</span></li>
              ))}
            </ul>
            <p className="hint">Full reasoning and sources: docs/retuning-assumptions/ASSUMPTIONS-REPORT.md</p>
          </details>
        </>
      )}
    </div>
  );
}
