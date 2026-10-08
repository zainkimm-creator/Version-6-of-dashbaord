import { useEffect, useMemo, useRef, useState } from 'react';
import MachineCanvas from './MachineCanvas.jsx';
import StepResponseChart, { medianMetric } from './StepResponseChart.jsx';
import { Field, SelectField } from '../ui/Field.jsx';
import { useSession } from '../session/SessionContext.jsx';

// The screen is the paper's loop drawn as two twins facing each other:
//
//   PHYSICAL TWIN  --- to the digital twin --->  DIGITAL TWIN
//   the line, its drift and the                  the gains the twin search found,
//   acquisition protocol                         what they cost on the twin and
//   <-- to the physical plant ---                on the plant, OPTIMISE GAIN
//
// and, under each, the cost test's step response: the same gains on the plant
// and on the twin, so their difference is the twin error seen as a response.
// One button runs the whole loop (identify -> search the twin -> apply). With
// "all ten plants" selected it runs every preset in turn and reports medians.

const PRESET_IDS = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10'];
const ALL_PLANTS = 'ALL';
// Two worked cases: a plant plus the drift a real line suffers, chosen so the
// commissioned gains visibly misbehave and the retune visibly fixes it.
const CASES = [
  { id: 'caseA', plant: 'P03', label: 'Case A — P03 PET lab line, reels swapped (J U/W −50 %, J R/W +100 %)',
    drift: { EA_pct: 0, J_UW_pct: -50, J_Nip_pct: 0, J_RW_pct: 100, f_pct: 0 } },
  { id: 'caseB', plant: 'P06', label: 'Case B — P06 aluminium lab line, reels swapped (J U/W −50 %, J R/W +100 %)',
    drift: { EA_pct: 0, J_UW_pct: -50, J_Nip_pct: 0, J_RW_pct: 100, f_pct: 0 } },
];
// The paper's ten drift scenarios (D01-D10), precomputed for every plant x T_log x
// noise in the paper's ranges (backend/pipeline/precompute.py). Picking one is a
// stored answer; editing the drift fields by hand computes live as before.
const PAPER_DRIFTS = [
  ['D01', 'EA +10 %',                        { EA_pct: 10,  J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 0 }],
  ['D02', 'EA −10 %',                        { EA_pct: -10, J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 0 }],
  ['D03', 'EA +30 %',                        { EA_pct: 30,  J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 0 }],
  ['D04', 'EA −30 %',                        { EA_pct: -30, J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 0 }],
  ['D05', 'EA +50 %',                        { EA_pct: 50,  J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 0 }],
  ['D06', 'J UW −30 % / RW +50 %',           { EA_pct: 0,   J_UW_pct: -30, J_Nip_pct: 0, J_RW_pct: 50,  f_pct: 0 }],
  ['D07', 'J UW −50 % / RW +100 %',          { EA_pct: 0,   J_UW_pct: -50, J_Nip_pct: 0, J_RW_pct: 100, f_pct: 0 }],
  ['D08', 'friction +30 %',                  { EA_pct: 0,   J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 30 }],
  ['D09', 'friction −30 %',                  { EA_pct: 0,   J_UW_pct: 0,   J_Nip_pct: 0, J_RW_pct: 0,   f_pct: -30 }],
  ['D10', 'EA +20 %, J UW −20 %, f +15 %',   { EA_pct: 20,  J_UW_pct: -20, J_Nip_pct: 0, J_RW_pct: 0,   f_pct: 15 }],
];
const EXCITATIONS = ['ET1', 'ET3', 'ET6', 'ET3M', 'E_Toggle', 'EV1'];
const TLOG_OPTIONS = [1, 2, 5, 10, 20, 50, 100];
// The paper's noise grid (Fig. 6), as a fraction of full scale.
const NOISE_OPTIONS = [0, 0.0002, 0.0005, 0.001, 0.003, 0.005];
// The anti-alias cutoffs the atlas holds, matched on both channels the way every
// published protocol sets them. 20 Hz is on the list because the paper reports
// it as a feasibility FAILURE. Keep in step with `LPF_AXIS_HZ` in
// backend/atlas/reader.py.
const LPF_OPTIONS = [
  { value: 'none', label: 'none' },
  { value: '20', label: '20 Hz' },
  { value: '50', label: '50 Hz' },
  { value: '100', label: '100 Hz' },
  { value: '200', label: '200 Hz' },
];

// Published record length per excitation (supplement Table S1). Changing the
// excitation or the velocity noise moves the record with it. Keep in step with
// PAPER_RECORD_S and DUAL_CHANNEL_RECORD_S in backend/pipeline/payloads.py.
const RECORD_S = { ET1: 7, ET3: 17, ET6: 32, ET3M: 51, E_Toggle: 16, EV1: 12 };
// Table S1: ET1 is 30 s in the dual-channel campaigns (velocity noise on).
const DUAL_CHANNEL_RECORD_S = { ET1: 30 };

export function publishedRecordS(excitation, pctV) {
  if (Number(pctV) > 0 && DUAL_CHANNEL_RECORD_S[excitation] !== undefined) {
    return DUAL_CHANNEL_RECORD_S[excitation];
  }
  return RECORD_S[excitation];
}

// Drift on the physical plant, as percent change from the commissioned plant.
const DRIFT_FIELDS = [
  ['EA_pct', 'EA'], ['J_UW_pct', 'J U/W'], ['J_Nip_pct', 'J NIP'],
  ['J_RW_pct', 'J R/W'], ['f_pct', 'FRICTION'],
];
const ZONES = ['UW', 'OutFeeder', 'RW'];
const ZONE_LABELS = { UW: 'UW', OutFeeder: 'out-feeder', RW: 'RW' };
const REQUIRED_PROTOCOL_FIELDS = [
  'T_log_ms', 'excitation', 'record_s', 'pct_T', 'pct_v', 'Kp_star', 'seed',
];
const LAYOUT_KEY = 'twin-screen-layout-v1';
const DEFAULT_LAYOUT = { split: 0.5, physical: 720, digital: 720, chartL: 400, chartR: 400 };

function num(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function fmtLcd(value, digits) {
  const v = Number(value);
  return value === null || value === undefined || !Number.isFinite(v) ? '----' : v.toFixed(digits);
}

function median(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function loadLayout() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(LAYOUT_KEY) ?? 'null');
    return saved && typeof saved === 'object' ? { ...DEFAULT_LAYOUT, ...saved } : DEFAULT_LAYOUT;
  } catch { return DEFAULT_LAYOUT; }
}

function Lcd({ label, value, unit, digits = 2, tone = 'pv', wide = false, title }) {
  return (
    <div className={`hmi-lcd-tile${wide ? ' is-wide' : ''}`} title={title}>
      <span className="hmi-lcd-label">{label}</span>
      <span className={`hmi-lcd tone-${tone}`}>{fmtLcd(value, digits)}</span>
      {unit && <span className="hmi-lcd-unit">{unit}</span>}
    </div>
  );
}

function HmiButton({ tone, children, ...props }) {
  return <button type="button" className={`hmi-btn tone-${tone}`} {...props}>{children}</button>;
}

// All three span tensions over the record, with a cursor at the playback time.
const TREND_COLOURS = ['#f2c94c', '#6fe38f', '#7cc4ff'];
const TREND_STROKE = [0, 1, 2].map((ch) => ({ stroke: `var(--hmi-ch${ch + 1}, ${TREND_COLOURS[ch]})` }));
const TREND_FILL = [0, 1, 2].map((ch) => ({ fill: `var(--hmi-ch${ch + 1}, ${TREND_COLOURS[ch]})` }));

function TensionTrend({ trace, frame }) {
  const W = 1000; const H = 130; const pad = { l: 44, r: 10, t: 10, b: 20 };
  if (!trace?.t_s?.length) {
    return <svg className="hmi-trend" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="No tension trend" />;
  }
  const t = trace.t_s;
  const all = [...trace.T_N.flat(), ...trace.T_ref_N.flat()].filter(Number.isFinite);
  const lo = Math.min(...all); const hi = Math.max(...all);
  const span = hi - lo || 1;
  const x = (v) => pad.l + (v / (t[t.length - 1] || 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / span) * (H - pad.t - pad.b);
  const path = (rows, ch) => rows.map((row, k) => `${k ? 'L' : 'M'}${x(t[k]).toFixed(1)},${y(row[ch]).toFixed(1)}`).join(' ');
  const cursor = x(t[Math.min(frame, t.length - 1)]);
  const ticks = [0, 0.5, 1].map((q) => q * (t[t.length - 1] || 0));
  return (
    <svg className="hmi-trend" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Span tensions over the identification record">
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} className="hmi-trend-bg" />
      {[lo, hi].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} className="hmi-trend-grid" />
          <text x={pad.l - 6} y={y(v) + 4} className="hmi-trend-tick" textAnchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      {ticks.map((v) => (
        <text key={v} x={x(v)} y={H - 4} className="hmi-trend-tick" textAnchor="middle">{v.toFixed(0)} s</text>
      ))}
      {[0, 1, 2].map((ch) => (
        <g key={ch}>
          <path d={path(trace.T_ref_N, ch)} className="hmi-trend-sv" stroke={TREND_COLOURS[ch]} />
          <path d={path(trace.T_N, ch)} className="hmi-trend-pv" style={TREND_STROKE[ch]} />
        </g>
      ))}
      <line x1={cursor} x2={cursor} y1={pad.t} y2={H - pad.b} className="hmi-trend-cursor" />
      {['T1', 'T2', 'T3'].map((label, ch) => (
        <text key={label} x={pad.l + 10 + ch * 34} y={pad.t + 13} className="hmi-trend-legend" style={TREND_FILL[ch]}>{label}</text>
      ))}
      <text x={W - pad.r - 8} y={pad.t + 13} className="hmi-trend-legend" textAnchor="end">solid measured · dashed set-point · N</text>
    </svg>
  );
}

// A window with a title bar and a drag grip in its bottom-right corner that
// changes its height. Width comes from the column divider.
function Window({ title, sub, height, onResize, className = '', area, children }) {
  const start = (event) => {
    event.preventDefault();
    const y0 = event.clientY; const h0 = height;
    const move = (e) => onResize(Math.max(160, h0 + (e.clientY - y0)));
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up);
  };
  return (
    <section className={`twin-window ${className}`} style={{ height, gridArea: area }}>
      <header className="twin-window-head">
        <h3>{title}</h3>
        {sub && <span>{sub}</span>}
      </header>
      <div className="twin-window-body">{children}</div>
      <div className="twin-grip" title="drag to resize" onMouseDown={start} />
    </section>
  );
}

function Divider({ onDrag }) {
  const start = (event) => {
    event.preventDefault();
    const move = (e) => onDrag(e.clientX);
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up);
  };
  return <div className="twin-divider" title="drag to change the width share" onMouseDown={start} />;
}

// Gains from a /retune payload as the /plant/step body wants them.
function gainsOf(block) {
  if (!block) return null;
  return { kp_star: block.kp_star_per_zone ?? block.kp_star, ti_s: block.ti_s_per_zone ?? block.ti_s };
}

// What the twin search proposed (before step 7's restore); older payloads lack it.
function proposalOf(retune) {
  return retune.gains.twin_candidate ?? retune.gains.delivered;
}

function estimatesOf(retune) {
  const rows = retune?.identification?.rows ?? [];
  return rows.length ? Object.fromEntries(rows.map((r) => [r.parameter, r.theta_hat])) : null;
}

// The two big arrows between the windows. Drawn as SVG block arrows with the
// text inside, white on a strong fill with a light outline, so they read on
// every skin and the direction is the shape itself.
function LinkArrow({ dir, label, caption, active }) {
  const W = 200; const H = 56;
  const pts = dir === 'forward'
    ? `0,10 ${W - 30},10 ${W - 30},0 ${W},${H / 2} ${W - 30},${H} ${W - 30},${H - 10} 0,${H - 10}`
    : `${W},10 30,10 30,0 0,${H / 2} 30,${H} 30,${H - 10} ${W},${H - 10}`;
  const cx = dir === 'forward' ? (W - 30) / 2 : (W + 30) / 2;
  const [l1, l2] = label;
  return (
    <div className={`twin-arrow is-${dir}${active ? ' is-active' : ''}`}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${l1} ${l2}`}>
        <polygon points={pts} />
        <text x={cx} y={H / 2 - 3} textAnchor="middle" className="twin-arrow-l1">{l1}</text>
        <text x={cx} y={H / 2 + 13} textAnchor="middle" className="twin-arrow-l2">{l2}</text>
      </svg>
      <span>{caption}</span>
    </div>
  );
}

export default function TwinStudy() {
  const {
    draft, session, staleness, error, clearError,
    setDraftPlant, setDraftProtocol, setDraftDrift, derive, fetchTrace, retuneFor, fetchStepFor, baselineFor,
    zoneGainsFor, forgetZoneGainsFor,
  } = useSession();

  const [readout, setReadout] = useState(null);
  const [trace, setTrace] = useState(null);
  const [clock, setClock] = useState(0);
  const timer = useRef(null);
  const requestId = useRef(0);

  // Which plant(s) the loop runs on. 'ALL' runs every preset and reports medians.
  const [scope, setScope] = useState('P01');
  // Two structures (7 Oct 2026). 'outfeeder-only-2D' is the default: UW and RW fixed
  // at the plant's saved six gains, the authors' HGS on the out-feeder only (the first
  // case of a plant runs the six-gain search and saves it). 'full-6D' re-commissions:
  // a free pair per zone, saved for the next case.
  const [gainStructure, setGainStructure] = useState('outfeeder-only-2D');
  const [forgetNote, setForgetNote] = useState(null);
  const [zone, setZone] = useState('UW');
  const [zoomed, setZoomed] = useState(true);
  // The three buttons are the paper's loop in order, and each unlocks the next:
  //   1 RUN PHYSICAL MACHINE  the line with the gains it runs today (commissioned)
  //   2 OPTIMISE GAIN         identify -> twin -> search; the twin's response
  //   3 APPLY TO THE MACHINE  the new gains on the real line
  const [stage, setStage] = useState(0);
  const [stageKey, setStageKey] = useState(null);
  // One entry per plant: { baseline, retune, twinBefore, twinAfter, plantAfter }.
  const [results, setResults] = useState({});
  const [progress, setProgress] = useState(null);   // { step, done, total, current, startedAt, failed: [] }
  const cancel = useRef(false);
  const [layout, setLayoutState] = useState(loadLayout);
  const gridRef = useRef(null);
  const setLayout = (patch) => setLayoutState((l) => {
    const next = { ...l, ...patch };
    try { window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(next)); } catch { /* private window */ }
    return next;
  });

  const specKey = useMemo(() => JSON.stringify([draft?.drift, draft?.protocol, scope, gainStructure]), [draft, scope, gainStructure]);

  // The running line: the identification record on the current plant.
  useEffect(() => {
    if (!draft) return undefined;
    const ready = REQUIRED_PROTOCOL_FIELDS.every((field) => draft.protocol?.[field] !== '')
      && Object.values(draft.drift ?? {}).every((value) => value !== '');
    if (!ready) return undefined;
    clearTimeout(timer.current);
    const id = requestId.current;
    const committedTheta = session?.theta_hat;
    const estimates = committedTheta?.converged && !staleness?.theta_hat?.stale
      ? committedTheta.estimates ?? null : null;
    timer.current = setTimeout(() => {
      derive()
        .then((body) => { if (requestId.current === id) setReadout(body); })
        .catch(() => { if (requestId.current === id) setReadout(null); });
      fetchTrace(estimates)
        .then((body) => { if (requestId.current === id) { setTrace(body); setClock(0); } })
        .catch(() => { if (requestId.current === id) setTrace(null); });
    }, 250);
    return () => { clearTimeout(timer.current); requestId.current += 1; };
  }, [JSON.stringify([draft, session?.twin?.swapped, session?.theta_hat?.run_hash])]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!trace || typeof requestAnimationFrame === 'undefined') return undefined;
    let raf; let last = performance.now(); let pending = 0;
    const loop = (now) => {
      pending += (now - last) / 1000; last = now;
      if (pending >= 0.05) { const dt = pending; pending = 0; setClock((c) => (c + dt) % (trace.record_s || 1)); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [trace]);

  if (!draft) return <p>Loading session…</p>;

  const { plant, protocol } = draft;
  const setProtocol = (key) => (value) => setDraftProtocol({ ...protocol, [key]: value });
  const published = publishedRecordS(protocol.excitation, protocol.pct_v);
  const noiseLevel = Math.max(Number(protocol.pct_T) || 0, Number(protocol.pct_v) || 0);
  const channels = Number(protocol.pct_v) > 0 ? 'dual' : 'tension';
  const setNoise = (level, mode) => {
    const pctV = mode === 'dual' ? level : 0;
    setDraftProtocol({
      ...protocol, pct_T: level, pct_v: pctV,
      record_s: publishedRecordS(protocol.excitation, pctV) ?? protocol.record_s,
    });
  };
  const choosePlant = (value) => {
    const kase = CASES.find((c) => c.id === value);
    if (kase) {
      setScope(kase.plant);
      setDraftPlant({ source: 'preset', preset_id: kase.plant });
      setDraftDrift(kase.drift);
      return;
    }
    setScope(value);
    if (value !== ALL_PLANTS) setDraftPlant({ source: 'preset', preset_id: value });
  };
  const activeCase = CASES.find((c) => c.plant === scope && JSON.stringify(c.drift) === JSON.stringify(draft.drift));
  // Which of the paper's ten drifts the fields currently hold, if any ('' = custom).
  const activeDrift = PAPER_DRIFTS.find(([, , d]) => JSON.stringify(d) === JSON.stringify(draft.drift))?.[0] ?? '';
  const chooseDrift = (code) => {
    const hit = PAPER_DRIFTS.find(([c]) => c === code);
    if (hit) setDraftDrift(hit[2]);
  };
  // Everything the step test is run with: as the line would measure it.
  const stepOpts = { measured: true, seed: Number(protocol.seed) || 0 };

  const plants = scope === ALL_PLANTS ? PRESET_IDS : [scope];
  const busy = Boolean(progress) && progress.done < progress.total;
  const fresh = stageKey === specKey;
  const liveStage = fresh ? stage : 0;
  const specFor = (id) => ({ plant: { source: 'preset', preset_id: id }, drift: draft.drift, protocol });

  // Run `work(id)` for every plant in turn, merging what it returns into that
  // plant's entry. One progress counter, one STOP, per button.
  const forEachPlant = async (step, work) => {
    cancel.current = false;
    setProgress({ step, done: 0, total: plants.length, current: plants[0], startedAt: Date.now(), failed: [] });
    for (const id of plants) {
      if (cancel.current) break;
      try {
        const patch = await work(id);
        setResults((r) => ({ ...r, [id]: { ...(r[id] ?? {}), ...patch } }));
      } catch (cause) {
        setResults((r) => ({ ...r, [id]: { ...(r[id] ?? {}), failure: cause.message } }));
        setProgress((p) => ({ ...p, failed: [...p.failed, id] }));
      }
      setProgress((p) => ({ ...p, done: p.done + 1, current: plants[plants.indexOf(id) + 1] ?? null }));
    }
  };

  // 1  the line as it runs today
  const runPhysical = async () => {
    const key = specKey;
    setResults({}); setStage(0); setStageKey(key);
    let worst = null;
    await forEachPlant(1, async (id) => {
      const baseline = await baselineFor(specFor(id), null, stepOpts);
      // Show the span the line misbehaves on most, so the overshoot is in view.
      baseline.step.metrics.forEach((m, i) => { if (!worst || m.overshoot_pct > worst.os) worst = { os: m.overshoot_pct, zone: ZONES[i] }; });
      return { baseline };
    });
    if (worst) setZone(worst.zone);
    setStage(1);
  };
  // 2  identify, build the twin, search it; the twin's own response before/after
  const optimise = async () => {
    await forEachPlant(2, async (id) => {
      const spec = specFor(id);
      const retune = await retuneFor(spec, { gain_structure: gainStructure });
      if (retune?.status !== 'ok') throw new Error(`${id}: ${retune?.note ?? 'identification failed'}`);
      const est = estimatesOf(retune);
      const [twinAfter, twinBefore] = await Promise.all([
        fetchStepFor(spec, gainsOf(proposalOf(retune)), est, stepOpts),
        fetchStepFor(spec, gainsOf(retune.gains.commissioned), est, stepOpts),
      ]);
      return { retune, twinAfter, twinBefore };
    });
    setStage(2);
  };
  // 3  the new gains back on the real line
  const apply = async () => {
    await forEachPlant(3, async (id) => {
      const r = results[id];
      if (!r?.retune) throw new Error(`${id}: no gains to apply`);
      return { plantAfter: await fetchStepFor(specFor(id), gainsOf(r.retune.gains.delivered), null, stepOpts) };
    });
    setStage(3);
  };

  const entries = fresh ? plants.map((id) => results[id]).filter(Boolean) : [];
  const withBaseline = entries.filter((r) => r.baseline);
  const withRetune = entries.filter((r) => r.retune?.status === 'ok');
  const retunes = withRetune.map((r) => r.retune);
  const med = (list, pick) => median(list.map(pick));
  const multi = plants.length > 1;
  const steps = (k) => entries.map((r) => r[k]).filter(Boolean);
  const baselineSteps = withBaseline.map((r) => r.baseline.step);

  const current = {
    kp: ZONES.map((_, i) => med(withBaseline, (r) => r.baseline.gains.kp_star_per_zone?.[i])),
    ti: ZONES.map((_, i) => med(withBaseline, (r) => r.baseline.gains.ti_s_per_zone?.[i])),
    sPlant: med(withBaseline, (r) => r.baseline.gains.on_plant?.S),
    sCommissioning: med(withBaseline, (r) => r.baseline.gains.at_commissioning?.S),
  };
  // The twin window shows what the twin proposed; APPLY sends what step 7
  // delivers, which is today's gains again when the proposal did not beat them.
  const delivered = {
    kp: ZONES.map((_, i) => med(retunes, (r) => proposalOf(r).kp_star_per_zone?.[i])),
    ti: ZONES.map((_, i) => med(retunes, (r) => proposalOf(r).ti_s_per_zone?.[i])),
    sTwin: med(retunes, (r) => proposalOf(r).S_twin),
    sPlant: med(retunes, (r) => r.gains.delivered.on_plant?.S),
    sBest: med(retunes, (r) => r.gains.reference_on_plant?.on_plant?.S),
    mare: med(retunes, (r) => r.identification?.MARE_theta_pct),
    gap: med(retunes, (r) => (r.ratio_vs_reference != null ? 100 * (r.ratio_vs_reference - 1) : null)),
  };
  const restored = retunes.filter((r) => r.restore?.restored);
  // what step 7 actually sent to the line (the proposal, or today's gains restored)
  const applied = {
    kp: ZONES.map((_, i) => med(retunes, (r) => r.gains.delivered.kp_star_per_zone?.[i])),
    ti: ZONES.map((_, i) => med(retunes, (r) => r.gains.delivered.ti_s_per_zone?.[i])),
  };
  const sTwinToday = med(retunes, (r) => r.gains.commissioned?.S_twin);
  const tier = retunes[0]?.cost_tier ?? withBaseline[0]?.baseline?.cost_tier ?? null;
  const perZone = retunes.map((r) => r.search?.per_zone_gains).filter(Boolean);
  const outfeederSearch = retunes.map((r) => r.search?.outfeeder).filter(Boolean);
  const twoStage = liveStage >= 2 && perZone.length > 0;
  const forgetSaved = async () => {
    const ids = scope === ALL_PLANTS ? plants : [scope];
    let n = 0;
    for (const id of ids) { const r = await forgetZoneGainsFor(specFor(id)); if (r?.deleted) n += 1; }
    setForgetNote(`${n} saved set${n === 1 ? '' : 's'} forgotten: the next optimise searches all six gains again`);
  };
  const showGains = liveStage >= 2 ? delivered : current;
  const perZoneShown = showGains.kp.some((v) => Number.isFinite(v) && Math.abs(v - showGains.kp[0]) > 1e-9)
    || showGains.ti.some((v) => Number.isFinite(v) && Math.abs(v - showGains.ti[0]) > 1e-9);
  const improvement = liveStage >= 3 && current.sPlant && delivered.sPlant ? 100 * (delivered.sPlant / current.sPlant - 1) : null;

  // The difference between the two twins, per span, at every stage so far.
  const diffRows = ZONES.map((z) => {
    const m = (list, key) => medianMetric(list, z, key);
    const osNow = liveStage >= 1 ? m(baselineSteps, 'overshoot_pct') : null;
    const osTwinBefore = liveStage >= 2 ? m(steps('twinBefore'), 'overshoot_pct') : null;
    const osTwin = liveStage >= 2 ? m(steps('twinAfter'), 'overshoot_pct') : null;
    const osAfter = liveStage >= 3 ? m(steps('plantAfter'), 'overshoot_pct') : null;
    const stNow = liveStage >= 1 ? m(baselineSteps, 'settling_time_s') : null;
    const stTwin = liveStage >= 2 ? m(steps('twinAfter'), 'settling_time_s') : null;
    const stAfter = liveStage >= 3 ? m(steps('plantAfter'), 'settling_time_s') : null;
    return { zone: z, osNow, osTwinBefore, osTwin, osAfter, stNow, stTwin, stAfter,
      twinGap: Number.isFinite(osAfter) && Number.isFinite(osTwin) ? osAfter - osTwin : null,
      cut: Number.isFinite(osNow) && Number.isFinite(osAfter) ? osAfter - osNow : null };
  });

  const frame = trace?.t_s?.length
    ? Math.min(trace.t_s.length - 1, Math.max(0, Math.floor(clock / (trace.dt_s || 1))))
    : 0;
  const elapsed = progress ? Math.round((Date.now() - progress.startedAt) / 1000) : 0;
  const notes = [];
  if (error) notes.push([error, ['ACK', clearError]]);
  if (published !== undefined && Number(protocol.record_s) !== published) {
    notes.push([`record ${protocol.record_s} s is not the published ${published} s for ${protocol.excitation}`,
      [`USE ${published} s`, () => setDraftProtocol({ ...protocol, record_s: published })]]);
  }
  if (Number(protocol.LPF_T_hz) === 20) notes.push(['LPF 20 Hz is below the paper’s feasibility gate (Fig. S6)']);
  if (progress?.failed?.length) notes.push([`failed on ${progress.failed.join(', ')}: ${entries.find((r) => r.failure)?.failure ?? 'see console'}`]);
  if (stageKey && !fresh) notes.push(['inputs changed — run the physical machine again']);

  const onDivider = (clientX) => {
    const rect = gridRef.current?.getBoundingClientRect();
    if (!rect) return;
    setLayout({ split: Math.min(0.7, Math.max(0.3, (clientX - rect.left) / rect.width)) });
  };
  const cols = `${(layout.split * 100).toFixed(2)}fr 210px ${((1 - layout.split) * 100).toFixed(2)}fr`;
  const stepLabel = (n) => (progress?.step === n && busy ? ` ${progress.done + 1} / ${progress.total}` : '');
  const progressText = busy
    ? `plant ${progress.current} · ${elapsed} s elapsed`
    : progress ? `${progress.done} / ${progress.total} done in ${elapsed} s` : '';

  return (
    <section className="page hmi">
      <div className="hmi-bezel">
        <header className="hmi-bar twin-bar">
          <div className="hmi-bar-tabs">
            <span className="hmi-tab is-active">MAIN</span>
            <span className="hmi-tab">{scope === ALL_PLANTS ? 'ALL 10 PLANTS' : `PLANT ${plant.preset_id ?? '—'}`}</span>
          </div>
          <div className="twin-notes">
            {notes.length ? notes.map(([text, action], i) => (
              <span key={i} className="twin-note">
                {text}{action && <button type="button" className="hmi-mini" onClick={action[1]}>{action[0]}</button>}
              </span>
            )) : <span className="twin-note is-quiet">running line: live playback of the identification record on {plant.preset_id ?? 'the plant'}</span>}
          </div>
        </header>

        <div className="hmi-canvas-wrap">
          <MachineCanvas trace={trace} frame={frame} playing v0={readout?.v0_m_s} />
          <TensionTrend trace={trace} frame={frame} />
          <div className="hmi-canvas-foot">
            <span>{protocol.excitation} · T_log {protocol.T_log_ms} ms · LPF {protocol.LPF_T_hz ?? 'none'} Hz · noise {(100 * noiseLevel).toFixed(2)} % {channels === 'dual' ? 'dual-channel' : 'tension-only'} · {num(trace?.record_s, 0)} s record</span>
            <span>{trace ? 'live playback of the identification record' : 'no playback for this protocol'}</span>
          </div>
        </div>

        <div className="twin-steps">
          <span className={liveStage >= 1 ? 'is-done' : ''}>1 RUN PHYSICAL MACHINE — the line with the gains it runs today</span>
          <span className={liveStage >= 2 ? 'is-done' : ''}>2 OPTIMISE GAIN — identify, build the twin, search it</span>
          <span className={liveStage >= 3 ? 'is-done' : ''}>3 APPLY TO THE MACHINE — the new gains back on the real line</span>
        </div>

        <div className="twin-grid" ref={gridRef} style={{ gridTemplateColumns: cols, gridTemplateAreas: '"physical links digital" "chartL chartLinks chartR"' }}>
          <Window title="PHYSICAL TWIN" sub="the line, as it drifted, and how it is measured"
                  height={layout.physical} onResize={(h) => setLayout({ physical: h })} className="is-physical" area="physical">
            <SelectField label="Plant / case" value={activeCase ? activeCase.id : scope} onChange={choosePlant}
                         options={[...CASES.map((c) => ({ value: c.id, label: c.label })),
                                   ...PRESET_IDS.map((id) => ({ value: id, label: `${id} (set the drift yourself)` })),
                                   { value: ALL_PLANTS, label: 'all 10 plants → median' }]} />
            <SelectField label="Drift scenario (paper D01–D10, precomputed)" value={activeDrift} onChange={chooseDrift}
                         options={[{ value: '', label: activeDrift ? '—' : 'custom (fields below)' },
                                   ...PAPER_DRIFTS.map(([c, l]) => ({ value: c, label: `${c} — ${l}` }))]} />
            <div className="hmi-pair">
              <SelectField label="Logging rate T_log" value={String(protocol.T_log_ms)}
                           onChange={(value) => setProtocol('T_log_ms')(Number(value))}
                           options={TLOG_OPTIONS.map((ms) => ({ value: String(ms), label: `${ms} ms` }))} />
              <SelectField label="Excitation" value={protocol.excitation}
                           onChange={(value) => setDraftProtocol({
                             ...protocol, excitation: value,
                             record_s: publishedRecordS(value, protocol.pct_v) ?? protocol.record_s,
                           })}
                           options={EXCITATIONS.map((name) => ({ value: name, label: name }))} />
            </div>
            <div className="hmi-pair">
              <SelectField label="Noise (% of full scale)" value={String(noiseLevel)}
                           onChange={(value) => setNoise(Number(value), channels)}
                           options={NOISE_OPTIONS.map((v) => ({ value: String(v), label: v === 0 ? 'noise-free' : `${(100 * v).toFixed(2)} %` }))} />
              <SelectField label="Noisy channels" value={channels}
                           onChange={(value) => setNoise(noiseLevel, value)}
                           options={[{ value: 'tension', label: 'tension only' }, { value: 'dual', label: 'tension + speed (dual)' }]} />
            </div>
            <SelectField label="Logging filter (anti-alias LPF)"
                         value={protocol.LPF_T_hz === null || protocol.LPF_T_hz === undefined ? 'none' : String(protocol.LPF_T_hz)}
                         onChange={(value) => {
                           const hz = value === 'none' ? null : Number(value);
                           setDraftProtocol({ ...protocol, LPF_T_hz: hz, LPF_v_hz: hz });
                         }}
                         options={LPF_OPTIONS} />
            <div className="flow-label">DRIFT SINCE COMMISSIONING (%)</div>
            <div className="twin-drift">
              {DRIFT_FIELDS.map(([key, label]) => (
                <Field key={key} label={label} value={draft.drift?.[key] ?? 0} step="5"
                       onChange={(v) => setDraftDrift({ ...draft.drift, [key]: v === '' ? '' : Number(v) })} />
              ))}
            </div>
            <div className="twin-buttons">
              <HmiButton tone="green" disabled={busy} onClick={runPhysical}>
                {`1 · RUN PHYSICAL MACHINE${stepLabel(1)}`}
              </HmiButton>
              <HmiButton tone="blue" disabled={busy || liveStage < 2} onClick={apply}>
                {`3 · APPLY TO THE MACHINE${stepLabel(3)}`}
              </HmiButton>
              {busy && <button type="button" className="hmi-mini" onClick={() => { cancel.current = true; }}>STOP</button>}
            </div>
            <dl className="flow-out">
              <dt>Gains today</dt><dd>{liveStage >= 1 ? `K_p* ${num(current.kp[0], 2)} · T_I ${num(current.ti[0], 1)} s (commissioned)` : 'run the machine to see'}</dd>
              <dt>Cost today S</dt><dd>{liveStage >= 1 ? `${num(current.sPlant, 4)}  (was ${num(current.sCommissioning, 4)} before the drift)` : '—'}</dd>
              <dt>Cost after retuning</dt><dd>{liveStage >= 3 ? `${num(delivered.sPlant, 4)}  →  ${improvement != null ? `${improvement.toFixed(1)} %` : ''}` : '—'}</dd>
              {liveStage >= 3 && restored.length > 0 && (
                <><dt>Restored</dt><dd>{multi
                  ? `${restored.length} of ${retunes.length} plants kept today's gains: the twin's did not beat them on the line`
                  : `today's gains kept: the twin's gains cost ${num(restored[0].restore.S_candidate, 4)} on the line vs ${num(restored[0].restore.S_commissioned, 4)} today`}</dd></>
              )}
            </dl>
          </Window>

          <div className="twin-links" style={{ gridArea: 'links' }}>
            <Divider onDrag={onDivider} />
            <LinkArrow dir="forward" label={["to the", "DIGITAL TWIN"]} caption="identify θ̂ → build the twin" active={liveStage >= 2} />
            <LinkArrow dir="back" label={["back to the", "PHYSICAL PLANT"]} caption="apply the gains → measure S" active={liveStage >= 3} />
          </div>

          <Window title="DIGITAL TWIN" sub={multi ? `medians over ${withRetune.length || withBaseline.length} of ${plants.length} plants` : (liveStage >= 2 ? 'the gains the twin search found, and what they cost' : 'optimise to search the twin for better gains')}
                  height={layout.digital} onResize={(h) => setLayout({ digital: h })} className="is-digital" area="digital">
            <div className="flow-label">{liveStage >= 2 ? 'OPTIMISED GAINS (FROM THE TWIN)' : 'GAINS THE LINE RUNS TODAY'}</div>
            {perZoneShown ? (
              <div className="zone-gains">
                {ZONES.map((z, i) => (
                  <div className="zone-gain-row" key={z}>
                    <span className="zone-name">{ZONE_LABELS[z]}</span>
                    <Lcd label="Kₚ*" value={showGains.kp[i]} digits={2} tone="out" />
                    <Lcd label="Tᵢ" unit="s" value={showGains.ti[i]} digits={2} tone="out" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="hmi-pair">
                <Lcd label="Kₚ*  (all zones)" value={showGains.kp[0]} digits={2} tone="out" />
                <Lcd label="Tᵢ  (all zones)" unit="s" value={showGains.ti[0]} digits={2} tone="out" />
              </div>
            )}
            <div className="hmi-pair">
              <Lcd label={`COST OF TWIN  S_twin${tier ? ` (${tier})` : ''}`} value={liveStage >= 2 ? delivered.sTwin : null} digits={4} tone="twin" />
              <Lcd label={`COST ON THE PLANT  S${tier ? ` (${tier})` : ''}`} value={liveStage >= 3 ? delivered.sPlant : null} digits={4} tone="out" />
            </div>
            <dl className="flow-out">
              <dt>Twin, gains today</dt><dd>{liveStage >= 2 ? num(sTwinToday, 4) : '—'}</dd>
              <dt>Best possible (same gains fixed)</dt><dd>{liveStage >= 3 ? `${num(delivered.sBest, 4)}${delivered.gap != null ? `  (${delivered.gap >= 0 ? '+' : ''}${delivered.gap.toFixed(1)} % above)` : ''}` : '—'}</dd>
              <dt>Twin error εθ</dt><dd>{liveStage >= 2 ? `${num(delivered.mare, 2)} %` : '—'}</dd>
            </dl>
            <label className="zone-select">
              <span>gain structure</span>
              <select value={gainStructure} disabled={busy} onChange={(e) => setGainStructure(e.target.value)}>
                <option value="outfeeder-only-2D">★ UW + RW fixed at the saved six gains, out-feeder searched</option>
                <option value="full-6D">six gains, one pair per zone — re-commission and save</option>
              </select>
            </label>
            {twoStage && (
              <div className="saved-gains">
                <div className="flow-label">SAVED SIX GAINS (STAGE 1){multi ? ` — ${perZone.filter((p) => p.reused).length} of ${perZone.length} plants reused` : ''}</div>
                {!multi && (
                  <>
                    <table className="data-table saved-gains-table">
                      <thead><tr><th>zone</th><th>K_p*</th><th>T_I (s)</th><th /></tr></thead>
                      <tbody>
                        {ZONES.map((z, i) => (
                          <tr key={z}>
                            <td>{ZONE_LABELS[z]}</td>
                            <td>{num(perZone[0].kp_star_per_zone[i], 2)}</td>
                            <td>{num(perZone[0].ti_s_per_zone[i], 1)}</td>
                            <td>{gainStructure === 'outfeeder-only-2D' ? (i === 1 ? 'searched now (HGS) →' : 'fixed') : 'searched (6-D)'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="saved-gains-note">
                      {perZone[0].reused
                        ? `reused: saved from an earlier case (run ${String(perZone[0].source_run_hash ?? '').slice(0, 8)})`
                        : perZone[0].saved
                          ? `new: searched on this case (${perZone[0].twin_evals.toLocaleString()} twin runs) and saved for the next case`
                          : `searched on this case but NOT saved: ${perZone[0].not_saved_because ?? 'the line did not accept it'}`}
                      {outfeederSearch[0] && ` · out-feeder by the authors' HGS: K_p* ${num(outfeederSearch[0].kp_star, 2)}, T_I ${num(outfeederSearch[0].ti_s, 1)} s (${outfeederSearch[0].evaluations.toLocaleString()} twin runs${outfeederSearch[0].kept_saved_value ? ', saved value kept' : ''})`}
                    </p>
                  </>
                )}
                <button type="button" className="hmi-mini" disabled={busy} onClick={forgetSaved}>FORGET SAVED SIX GAINS</button>
                {forgetNote && <span className="saved-gains-note"> {forgetNote}</span>}
              </div>
            )}
            <div className="twin-buttons">
              <HmiButton tone="amber" disabled={busy || liveStage < 1} onClick={optimise}>
                {`2 · OPTIMISE GAIN${stepLabel(2)}`}
              </HmiButton>
              <span className="twin-progress">{progressText || (liveStage < 1 ? 'run the physical machine first' : 'identify → search the twin, no line experiments')}</span>
            </div>
          </Window>

          <Window title="STEP RESPONSE — PHYSICAL MACHINE" sub={liveStage >= 3 ? 'the cost test on the real line: before (dashed) and after retuning' : 'the cost test on the real line with the gains it runs today'}
                  height={layout.chartL} onResize={(h) => setLayout({ chartL: h })} className="is-chart" area="chartL">
            <StepResponseChart title="physical" zone={zone} normalise={multi} tone="pv"
                               after={liveStage >= 3 ? steps('plantAfter') : baselineSteps}
                               before={liveStage >= 3 ? baselineSteps : []}
                               zoomed={zoomed} afterLabel={liveStage >= 3 ? 'after retuning (real line)' : 'gains today (real line)'}
                               beforeLabel="before retuning"
                               note="press 1 · RUN PHYSICAL MACHINE" />
          </Window>
          <div className="twin-links is-charts" style={{ gridArea: 'chartLinks' }}>
            <Divider onDrag={onDivider} />
            <label className="zone-select">
              <span>span shown</span>
              <select value={zone} onChange={(e) => setZone(e.target.value)}>
                {ZONES.map((z) => <option key={z} value={z}>{ZONE_LABELS[z]} · T{ZONES.indexOf(z) + 1}</option>)}
              </select>
            </label>
            <label className="zone-select twin-zoom">
              <span>view</span>
              <select value={zoomed ? 'zoom' : 'full'} onChange={(e) => setZoomed(e.target.value === 'zoom')}>
                <option value="zoom">zoom on the overshoot</option>
                <option value="full">whole step</option>
              </select>
            </label>
          </div>
          <Window title="STEP RESPONSE — DIGITAL TWIN" sub="the same test on the twin built from θ̂: gains today (dashed) and optimised"
                  height={layout.chartR} onResize={(h) => setLayout({ chartR: h })} className="is-chart" area="chartR">
            <StepResponseChart title="digital" zone={zone} normalise={multi} tone="twin"
                               after={liveStage >= 2 ? steps('twinAfter') : []}
                               before={liveStage >= 2 ? steps('twinBefore') : []}
                               zoomed={zoomed} afterLabel="optimised gains (twin)" beforeLabel="gains today (twin)"
                               note="press 2 · OPTIMISE GAIN" />
          </Window>
        </div>

        <div className="twin-diff">
          <div className="twin-diff-head">
            <h3>PHYSICAL vs DIGITAL — the difference{multi ? ` (medians over ${plants.length} plants)` : ''}</h3>
            <span>per span · overshoot of the +20 % step and the ±2 % settling time · the twin gap is the same gains on the real line minus on the twin</span>
          </div>
          <table className="data-table twin-diff-table twin-gain-table">
            <thead>
              <tr><th>gains</th><th>① today (commissioned)</th><th colSpan={2}>② optimised on the twin</th><th>③ applied to the line</th><th>twin cost S_twin</th><th>line cost S</th></tr>
            </thead>
            <tbody>
              {ZONES.map((z, i) => (
                <tr key={z} className={z === zone ? 'is-current' : ''}>
                  <td>{ZONE_LABELS[z]}</td>
                  <td>{liveStage >= 1 ? `K_p* ${num(current.kp[i], 2)} · T_I ${num(current.ti[i], 1)} s` : '—'}</td>
                  <td colSpan={2}>{liveStage >= 2 ? `K_p* ${num(delivered.kp[i], 2)} · T_I ${num(delivered.ti[i], 1)} s` : '—'}</td>
                  <td>{liveStage >= 3 ? `K_p* ${num(applied.kp[i], 2)} · T_I ${num(applied.ti[i], 1)} s` : '—'}</td>
                  {i === 0 && <td rowSpan={3}>{liveStage >= 2 ? `${num(sTwinToday, 4)} → ${num(delivered.sTwin, 4)}` : '—'}</td>}
                  {i === 0 && <td rowSpan={3}>{liveStage >= 3 ? `${num(current.sPlant, 4)} → ${num(delivered.sPlant, 4)}` : liveStage >= 1 ? `${num(current.sPlant, 4)} today` : '—'}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          <table className="data-table twin-diff-table">
            <thead>
              <tr>
                <th>span</th>
                <th>① real line today<br /><small>overshoot · settling</small></th>
                <th>② twin, gains today<br /><small>overshoot</small></th>
                <th>② twin, optimised<br /><small>overshoot · settling</small></th>
                <th>③ real line, retuned<br /><small>overshoot · settling</small></th>
                <th>overshoot cut<br /><small>③ − ①</small></th>
                <th>twin gap<br /><small>③ − ② optimised</small></th>
              </tr>
            </thead>
            <tbody>
              {diffRows.map((r) => (
                <tr key={r.zone} className={r.zone === zone ? 'is-current' : ''}>
                  <td>{ZONE_LABELS[r.zone]}</td>
                  <td>{num(r.osNow, 1)} % · {num(r.stNow, 2)} s</td>
                  <td>{num(r.osTwinBefore, 1)} %</td>
                  <td>{num(r.osTwin, 1)} % · {num(r.stTwin, 2)} s</td>
                  <td>{num(r.osAfter, 1)} % · {num(r.stAfter, 2)} s</td>
                  <td className={r.cut != null && r.cut < 0 ? 'is-good' : ''}>{r.cut != null ? `${r.cut > 0 ? '+' : ''}${r.cut.toFixed(1)} pp` : '—'}</td>
                  <td>{num(r.twinGap, 1)} pp</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
