/**
 * The three-span line, drawn to scale, with the digital twin measured against it.
 *
 * Every acquisition control on the Twin Study screen changes something physical
 * about this picture, and the picture is the only place a reader can see all of
 * them at once. The geometry redraws on every edit (the screen re-derives 150 ms
 * after a keystroke); the twin comparison appears once an identification exists.
 *
 * Topology, from the paper's section 2.1 and the data package's column notes:
 *
 *     [UW] ==T1== [Feeder] ==T2== [Nip] ==T3== [RW]
 *
 * `L1..L3` are UW->Feeder, Feeder->Nip, Nip->RW. The feeder runs at constant
 * velocity and is not actuated, so it carries no identified parameters. Eq. (2)
 * pairs the spans as (0,1) for the unwinder, (2,3) for the nip and (3,4) for the
 * rewinder with `T0 = T4 = 0`, which is why the unwinder is braked by `T1`
 * alone, the nip sees `T2 - T3`, and the rewinder sees `+T3`.
 *
 * HOW THE TWIN DIFFERENCE IS SHOWN. A drawn-to-scale overlay fails here: the
 * identification errors that matter run from a few tenths of a percent to a few
 * hundred, and at true scale everything below ~10 % is invisible while anything
 * above ~100 % leaves the canvas. So each parameter gets a DEVIATION GAUGE - a
 * track with zero at the centre and a marker placed on a symmetric log scale -
 * which resolves 0.3 % and 300 % on the same axis. The numbers sit beside it, so
 * the gauge is the glance and the number is the reading.
 *
 * The one true-scale twin quantity drawn on the line is the implied inertia:
 * `k_t = R^2 / J` and the roller radius is known from commissioning, so an error
 * in k_t IS an error in inertia. The hub disc is sized by J, and the twin's
 * implied J is drawn as a dashed hub beside it. That comparison is honest
 * because both discs are on the same scale.
 *
 * Nothing here computes physics. Every number is read from `/plant/derive` or
 * from the identification result; the component only positions and scales.
 */

const VIEW_W = 1080;
const VIEW_H = 466;
const WEB_Y = 132;          // baseline the web runs along
const PLOT_X0 = 96;
const PLOT_X1 = VIEW_W - 96;
const CARD_Y = 224;         // top of the per-roller comparison cards
const CARD_GAP = 18;
const CARD_W = (VIEW_W - 2 * 16 - 2 * CARD_GAP) / 3;
const CARD_H = 176;

// Which tension channels each excitation actually steps. Source of truth is
// paper_package/excitation_schedules_v5.csv; this mirrors it so the diagram can
// mark the spans a protocol perturbs without another round trip.
const EXCITATION_SPANS = {
  ET1: [1],
  ET3: [1, 2, 3],
  ET6: [1, 2, 3],
  ET3M: [1, 2, 3],
  E_Toggle: [1, 2, 3],
  EV1: [],                  // line-speed step only; no tension channel moves
};

const ROLLERS = [
  { key: 'UW', label: 'Unwinder', index: 0 },
  { key: 'Nip', label: 'Nip', index: 1 },
  { key: 'RW', label: 'Rewinder', index: 2 },
];

/** Colour for a signed relative error, in percent. */
function errorColour(pct) {
  const e = Math.abs(Number(pct));
  if (!Number.isFinite(e)) return '#8a8a8a';
  if (e < 1) return '#1baf7a';
  if (e < 5) return '#6fbf4f';
  if (e < 20) return '#e0a52e';
  if (e < 100) return '#eb6834';
  return '#d0342c';
}

/**
 * Marker position on a deviation gauge, as a fraction in [-1, 1].
 *
 * Symmetric log: 0.1 % sits near the centre, 1 % a third out, 10 % two thirds,
 * 100 % at the end. Errors past 100 % clamp so the marker stays on the track and
 * the number beside it carries the magnitude.
 */
function gaugeFraction(pct) {
  const e = Number(pct);
  if (!Number.isFinite(e) || e === 0) return 0;
  const mag = Math.min(1, Math.log10(Math.abs(e) / 0.1) / 3);
  return Math.sign(e) * Math.max(0, mag);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  const n = Number(value);
  if (n !== 0 && (Math.abs(n) >= 1e5 || Math.abs(n) < 1e-3)) return n.toExponential(2);
  return n.toFixed(digits);
}

function fmtPct(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  const n = Number(value);
  return `${n > 0 ? '+' : ''}${n.toFixed(Math.abs(n) < 10 ? 2 : 1)}%`;
}

/** One parameter's row inside a roller card: name, both values, gauge. */
function GaugeRow({ x, y, name, unit, trueValue, hatValue, errorPct, haveTwin }) {
  const half = 52;                       // half-width of the gauge track
  const cx = x + CARD_W - 74;
  const frac = haveTwin ? gaugeFraction(errorPct) : 0;
  const colour = errorColour(errorPct);
  return (
    <g className="span-gauge-row">
      <text x={x + 14} y={y} className="span-card-key">{name}</text>
      <text x={x + 14} y={y + 15} className="span-card-val">
        {fmt(trueValue, 4)}{unit ? ` ${unit}` : ''}
      </text>
      {haveTwin ? (
        <>
          <text x={x + 14} y={y + 30} className="span-card-hat" fill={colour}>
            twin {fmt(hatValue, 4)}
          </text>
          <line x1={cx - half} y1={y + 4} x2={cx + half} y2={y + 4}
                className="span-gauge-track" />
          <line x1={cx} y1={y - 4} x2={cx} y2={y + 12} className="span-gauge-zero" />
          <line x1={cx} y1={y + 4} x2={cx + frac * half} y2={y + 4}
                className="span-gauge-bar" stroke={colour} />
          <circle cx={cx + frac * half} cy={y + 4} r={4.5}
                  className="span-gauge-dot" fill={colour} />
          <text x={cx} y={y + 26} className="span-gauge-value" fill={colour}>
            {fmtPct(errorPct)}
          </text>
        </>
      ) : (
        <text x={cx} y={y + 8} className="span-card-pending">identify to compare</text>
      )}
    </g>
  );
}

export default function SpanDiagram({
  readout,
  twinRows,
  protocol,
  twinStale = false,
  derivedFromTwin = false,
}) {
  if (!readout) {
    return (
      <div className="span-diagram span-diagram-empty">
        <p className="hint">Select a plant to draw the line.</p>
      </div>
    );
  }

  const R = Array.isArray(readout.R_m) && readout.R_m.length === 3
    ? readout.R_m.map(Number) : [0.15, 0.1, 0.15];
  const L = Array.isArray(readout.L_m) && readout.L_m.length === 3
    ? readout.L_m.map(Number) : [2, 3, 3];
  const J = readout.J_kg_m2 ?? [];
  const F = readout.f_Nms_per_rad ?? [];
  const theta = readout.theta_drifted ?? readout.theta_true ?? {};
  const tRef = Number(readout.T_ref_N) || 0;
  const sigmaT = Number(readout.sigma_T_N) || 0;

  const hat = {};
  const err = {};
  (twinRows ?? []).forEach((row) => {
    hat[row.parameter] = Number(row.theta_hat);
    err[row.parameter] = Number(row.error_pct);
  });
  const haveTwin = Object.keys(hat).length > 0;

  // --- horizontal layout: span widths in proportion to their real lengths ---
  const totalL = L.reduce((a, b) => a + b, 0) || 1;
  const usable = PLOT_X1 - PLOT_X0;
  const nodeX = [PLOT_X0];
  L.forEach((len) => { nodeX.push(nodeX[nodeX.length - 1] + (len / totalL) * usable); });

  // --- roller radii to scale, with a floor so the smallest stays legible ---
  const maxR = Math.max(...R, 1e-6);
  const rPx = (value) => Math.max(12, (value / maxR) * 34);

  // Hubs carry inertia, on a shared scale across the three rollers so the discs
  // are comparable to each other and to the twin's implied values.
  const jValues = ROLLERS.map((r) => Number(J[r.index])).filter(Number.isFinite);
  const maxJ = Math.max(...jValues, 1e-9);
  const hubPx = (value) => {
    const v = Number(value);
    if (!Number.isFinite(v) || v <= 0) return 0;
    return Math.max(3, Math.sqrt(v / maxJ) * 17);
  };

  const nodes = [
    { kind: 'roller', roller: ROLLERS[0], x: nodeX[0], r: rPx(R[0]) },
    { kind: 'feeder', x: nodeX[1], r: 13 },
    { kind: 'roller', roller: ROLLERS[1], x: nodeX[2], r: rPx(R[1]) },
    { kind: 'roller', roller: ROLLERS[2], x: nodeX[3], r: rPx(R[2]) },
  ];

  const steppedSpans = EXCITATION_SPANS[protocol?.excitation] ?? [];
  const stepN = 0.2 * tRef;                     // supplement S1.1: every step is +20 %
  // Noise ribbon height on the same scale as the step arrow above it, so the
  // picture answers "how big is the noise next to the excitation" directly.
  const STEP_PX = 42;
  const noisePx = stepN > 0 ? Math.max(2, (sigmaT / stepN) * STEP_PX) : 0;

  const tickCount = Math.max(2, Math.min(40, Math.round(200 / (Number(protocol?.T_log_ms) || 5))));
  const eaErr = haveTwin ? err.EA : null;

  // Three even columns across the full width. Anchoring each card under its
  // roller instead leaves a wide hole where the (unactuated) feeder sits and
  // pushes the outer two off the canvas, so the cards are evenly spaced and a
  // connector line ties each one back to the roller it describes.
  const cardX = (i) => 16 + i * (CARD_W + CARD_GAP);

  return (
    <div className="span-diagram">
      <div className="span-diagram-head">
        <h3>The line</h3>
        <div className="span-legend">
          <span className="span-legend-item"><i className="swatch swatch-phys" /> physical plant</span>
          <span className="span-legend-item"><i className="swatch swatch-twin" /> digital twin</span>
          {haveTwin && (
            <span className="span-legend-item span-legend-scale">
              error <i className="swatch swatch-ok" />&lt;1%
              <i className="swatch swatch-good" />&lt;5%
              <i className="swatch swatch-warn" />&lt;20%
              <i className="swatch swatch-bad" />&lt;100%
              <i className="swatch swatch-worst" />≥100%
            </span>
          )}
        </div>
      </div>

      <svg
        className={`span-svg${twinStale ? ' is-stale' : ''}`}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label="Three-span roll-to-roll line with the identified digital twin compared against it"
      >

        {/* ---- web axial stiffness, the one parameter shared by all spans ---- */}
        <g className="span-ea">
          <rect x={PLOT_X0} y={16} width={PLOT_X1 - PLOT_X0} height={30} rx={8}
                className="span-ea-band" />
          <text x={PLOT_X0 + 14} y={36} className="span-ea-text">
            web stiffness EA
          </text>
          <text x={PLOT_X0 + 150} y={36} className="span-ea-value">
            {fmt(theta.EA, 1)} N
          </text>
          {haveTwin && (
            <>
              <text x={PLOT_X0 + 280} y={36} className="span-ea-value" fill={errorColour(eaErr)}>
                twin {fmt(hat.EA, 1)} N
              </text>
              <text x={PLOT_X0 + 430} y={36} className="span-ea-value" fill={errorColour(eaErr)}>
                {fmtPct(eaErr)}
              </text>
            </>
          )}
          <text x={PLOT_X1 - 14} y={36} className="span-ea-note">
            v₀ {fmt(readout.v0_m_s, 3)} m/s · {readout.speed_class} ·
            {' '}τ_min {fmt(readout.tau_min_ms, 1)} ms
          </text>
        </g>

        {/* ---- the three web spans ---- */}
        {L.map((len, i) => {
          const x0 = nodeX[i];
          const x1 = nodeX[i + 1];
          const mid = (x0 + x1) / 2;
          const spanNo = i + 1;
          const stepped = steppedSpans.includes(spanNo);
          const thickness = Math.max(2, Math.min(8, 2 + (tRef / 250) * 6));
          return (
            <g key={`span-${spanNo}`} className={`span-web${stepped ? ' is-stepped' : ''}`}>
              {noisePx > 0 && (
                <rect x={x0} y={WEB_Y - noisePx / 2} width={x1 - x0} height={noisePx}
                      className="span-noise-band" />
              )}
              <line x1={x0} y1={WEB_Y} x2={x1} y2={WEB_Y}
                    className="span-web-line" strokeWidth={thickness} />
              {Array.from({ length: tickCount }, (_, k) => {
                const x = x0 + ((k + 0.5) / tickCount) * (x1 - x0);
                return <line key={k} x1={x} y1={WEB_Y - 4} x2={x} y2={WEB_Y + 4}
                             className="span-log-tick" />;
              })}
              {stepped && (
                <g className="span-step">
                  <line x1={mid} y1={WEB_Y - 8} x2={mid} y2={WEB_Y - 6 - STEP_PX}
                        className="span-step-arrow" />
                  <polygon
                    points={`${mid},${WEB_Y - 8 - STEP_PX} ${mid - 4.5},${WEB_Y - 1 - STEP_PX} ${mid + 4.5},${WEB_Y - 1 - STEP_PX}`}
                    className="span-arrowhead"
                  />
                  <text
                    x={mid > VIEW_W * 0.62 ? mid - 8 : mid + 8}
                    y={WEB_Y - 4 - STEP_PX}
                    className="span-step-text"
                    textAnchor={mid > VIEW_W * 0.62 ? 'end' : 'start'}
                  >
                    step +20% = {fmt(stepN, 2)} N
                  </text>
                </g>
              )}
              <text x={mid} y={WEB_Y + 34} className="span-label">
                T{spanNo} {fmt(tRef, 1)} N
              </text>
              <text x={mid} y={WEB_Y + 49} className="span-sub">
                L{spanNo} {fmt(len, 2)} m
              </text>
            </g>
          );
        })}

        {/* noise callout, once, against the first span */}
        {noisePx > 0 && (
          <g className="span-noise-callout">
            <line x1={nodeX[0] + 26} y1={WEB_Y + noisePx / 2} x2={nodeX[0] + 26} y2={WEB_Y + 60}
                  className="span-callout-line" />
            <text x={nodeX[0] + 30} y={WEB_Y + 70} className="span-note">
              sensor noise σ_T = {fmt(sigmaT, 4)} N — {fmt(100 * (sigmaT / (stepN || 1)), 1)}% of the step
            </text>
          </g>
        )}

        {/* ---- rollers and the feeder ---- */}
        {nodes.map((node) => {
          if (node.kind === 'feeder') {
            return (
              <g key="feeder" className="span-node span-feeder">
                <circle cx={node.x} cy={WEB_Y} r={node.r} className="span-feeder-body" />
                <text x={node.x} y={WEB_Y + node.r + 16} className="span-node-name">Feeder</text>
                <text x={node.x} y={WEB_Y + node.r + 29} className="span-sub">constant v</text>
              </g>
            );
          }
          const { key, label, index } = node.roller;
          const jTrue = Number(J[index]);
          // With a twin present the rim carries that roller's k_t error, so the
          // schematic itself says which roller the identification got worst.
          const ktErr = haveTwin ? err[`kt_${key}`] : null;
          return (
            <g key={key} className="span-node">
              <circle
                cx={node.x} cy={WEB_Y} r={node.r} className="span-roller-body"
                stroke={haveTwin ? errorColour(ktErr) : undefined}
              />
              <circle cx={node.x} cy={WEB_Y} r={hubPx(jTrue)} className="span-roller-hub" />
              <text x={node.x} y={WEB_Y - node.r - 22} className="span-node-name">{label}</text>
              <text x={node.x} y={WEB_Y - node.r - 9} className="span-sub">
                R {fmt(R[index], 3)} m · J {fmt(jTrue, 4)}
              </text>
            </g>
          );
        })}

        {/* ---- per-roller comparison cards ---- */}
        {ROLLERS.map((roller, i) => {
          const nodeIdx = i === 0 ? 0 : i + 1;
          const x = cardX(i);
          const rollerX = nodes[nodeIdx].x;
          const ktKey = `kt_${roller.key}`;
          const kfKey = `kf_${roller.key}`;
          const jTrue = Number(J[roller.index]);
          const jHat = haveTwin && Number.isFinite(hat[ktKey]) && hat[ktKey] > 0
            ? (R[roller.index] * R[roller.index]) / hat[ktKey]
            : null;
          return (
            <g key={`card-${roller.key}`} className="span-card">
              <path
                d={`M ${rollerX} ${WEB_Y + 56} L ${rollerX} ${CARD_Y - 14} L ${x + CARD_W / 2} ${CARD_Y - 14} L ${x + CARD_W / 2} ${CARD_Y}`}
                className="span-card-link"
              />
              <rect x={x} y={CARD_Y} width={CARD_W} height={CARD_H} rx={10}
                    className="span-card-box" />
              <text x={x + 14} y={CARD_Y + 22} className="span-card-title">{roller.label}</text>
              <GaugeRow x={x} y={CARD_Y + 46} name="k_t = R²/J" unit=""
                        trueValue={theta[ktKey]} hatValue={hat[ktKey]}
                        errorPct={err[ktKey]} haveTwin={haveTwin} />
              <GaugeRow x={x} y={CARD_Y + 102} name="k_f = f/J" unit=""
                        trueValue={theta[kfKey]} hatValue={hat[kfKey]}
                        errorPct={err[kfKey]} haveTwin={haveTwin} />
              <text x={x + 14} y={CARD_Y + CARD_H - 10} className="span-card-foot">
                J {fmt(jTrue, 4)} kg·m²
                {jHat !== null && ` → twin ${fmt(jHat, 4)}`}
                {'  ·  f '}{fmt(F[roller.index], 3)}
              </text>
            </g>
          );
        })}

        {/* ---- acquisition strip ---- */}
        <g className="span-protocol">
          <line x1={PLOT_X0} y1={VIEW_H - 34} x2={PLOT_X1} y2={VIEW_H - 34}
                className="span-rule" />
          <text x={PLOT_X0} y={VIEW_H - 14} className="span-note">
            {protocol?.excitation ?? '—'} · T_log {fmt(protocol?.T_log_ms, 0)} ms ·
            {' '}σ_T {fmt(sigmaT, 4)} N ({fmt(100 * (Number(protocol?.pct_T) || 0), 2)}% of{' '}
            {fmt(readout.T_max_N, 0)} N full scale) · σ_v {fmt(readout.sigma_v_m_s, 4)} m/s ·
            {' '}LPF {protocol?.LPF_T_hz ?? 'none'}/{protocol?.LPF_v_hz ?? 'none'} Hz
          </text>
          <text x={PLOT_X1} y={VIEW_H - 14} className="span-note span-note-right">
            ζ_CL,min {fmt(readout.zeta_cl_min, 3)} ({readout.regime}) ·
            {' '}K_p* {fmt(readout.Kp_star, 0)} · T_I {fmt(readout.T_I_s, 2)} s
          </text>
        </g>

        {steppedSpans.length === 0 && (
          <text x={(PLOT_X0 + PLOT_X1) / 2} y={WEB_Y - 58} className="span-note span-note-mid">
            {protocol?.excitation} steps the line speed only — no tension channel is perturbed
          </text>
        )}
      </svg>

      <div className="span-errors">
        <div className="span-errors-head">
          <strong>All seven parameters</strong>
          {!haveTwin && <span className="hint"> — identify to compare</span>}
          {haveTwin && derivedFromTwin && <span className="chip chip-warn">columns swapped</span>}
          {haveTwin && twinStale && <span className="chip chip-stale">stale θ̂</span>}
        </div>
        {haveTwin && (
          <ul className="span-error-bars">
            {(twinRows ?? []).map((row) => {
              const e = Number(row.error_pct);
              const frac = gaugeFraction(e);
              const colour = errorColour(e);
              return (
                <li key={row.parameter}>
                  <span className="span-error-name">{row.parameter}</span>
                  <span className="span-error-track">
                    <span className="span-error-zero" />
                    <span
                      className="span-error-fill"
                      style={{
                        background: colour,
                        left: frac >= 0 ? '50%' : `${50 + frac * 50}%`,
                        width: `${Math.abs(frac) * 50}%`,
                      }}
                    />
                  </span>
                  <span className="span-error-value" style={{ color: colour }}>{fmtPct(e)}</span>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
