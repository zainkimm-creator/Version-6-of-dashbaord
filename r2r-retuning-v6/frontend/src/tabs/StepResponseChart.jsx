import { useEffect, useRef, useState } from 'react';

// The cost test's step response, drawn the way a commissioning engineer reads
// it: the +20 % step, the overshoot, the time constant (63 % rise, dashed red),
// the settling window (amber): the total time the trace spends outside +-2 % of
// the step -- the quantity the cost scores; not 'last sample outside', which jumps
// and the stable region after it (grey). "Before retuning" is the dashed grey
// trace with the commissioned gains; "after" is the solid one.
//
// `runs` is a list of step payloads (one per plant). One run draws in newtons;
// several draw normalised to each plant's own T_ref with the median in bold and
// the individual plants faint behind it -- ten plants with set-points from 12 N
// to 918 N cannot share a newton axis.

const Z = { UW: 0, OutFeeder: 1, RW: 2 };

// Fill the box the window gives us: the SVG takes the container's pixel size as
// its viewBox, so dragging the window bigger shows more, not stretched text.
function useSize(ref) {
  const [size, setSize] = useState({ w: 640, h: 300 });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 50 && height > 50) setSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

function median(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// One plant's curve for a zone, in N or as a fraction of its own T_ref.
function curve(run, zone, normalise) {
  if (!run?.t_s?.length) return null;
  const i = Z[zone] ?? 0;
  const base = run.T_ref_base_N[i];
  const rows = run.T_meas_N ?? run.T_N;
  return { t: run.t_s, y: rows.map((row) => (normalise ? row[i] / base : row[i])) };
}

// Median across plants, sample by sample (every run shares the same clock).
function medianCurve(curves) {
  const valid = curves.filter(Boolean);
  if (!valid.length) return null;
  const n = Math.min(...valid.map((c) => c.t.length));
  return { t: valid[0].t.slice(0, n), y: Array.from({ length: n }, (_, k) => median(valid.map((c) => c.y[k]))) };
}

export function medianMetric(runs, zone, key) {
  const i = Z[zone] ?? 0;
  return median(runs.map((r) => r?.metrics?.[i]?.[key]));
}

export default function StepResponseChart({ title, after = [], before = [], zone = 'UW', normalise = false, tone = 'pv', note, afterLabel = 'after retuning', beforeLabel = 'before (commissioned gains)', zoomed = true }) {
  const box = useRef(null);
  const { w: W, h: H } = useSize(box);
  const pad = { l: 64, r: 16, t: 16, b: 42 };
  const i = Z[zone] ?? 0;
  const runs = after.filter(Boolean);
  const main = runs.length ? medianCurve(runs.map((r) => curve(r, zone, normalise))) : null;
  const prior = before.filter(Boolean).length ? medianCurve(before.filter(Boolean).map((r) => curve(r, zone, normalise))) : null;
  const faint = runs.length > 1 ? runs.map((r) => curve(r, zone, normalise)) : [];
  const step = runs[0]?.step_time_s ?? 5;
  const metrics = {
    overshoot: medianMetric(runs, zone, 'overshoot_pct'),
    tau: medianMetric(runs, zone, 'time_constant_s'),
    settle: medianMetric(runs, zone, 'settling_time_s'),
    peak: medianMetric(runs, zone, 'peak_N'),
    peakT: medianMetric(runs, zone, 'peak_time_s'),
  };
  const base = normalise ? 1 : runs[0]?.T_ref_base_N?.[i];
  const final = normalise ? 1.2 : runs[0]?.T_ref_final_N?.[i];
  const stepSize = final && base ? final - base : null;

  // Zoom on what matters: from just before the step to a little past the
  // longest settling time (never less than 3 s after the step), and a tension
  // axis from the set-point to the peak plus headroom, so the overshoot and
  // the +-2 % band are big enough to read.
  const settleMax = Math.max(...[...runs, ...before.filter(Boolean)].map((r) => r?.metrics?.[i]?.settling_time_s ?? 0), 0);
  const tMin = Math.max(0, step - 0.5);
  const tMax = Math.min(runs[0]?.duration_s ?? 30, step + Math.min(10, Math.max(3, 1.6 * settleMax + 1)));
  const ys = [main, prior, ...faint].filter(Boolean).flatMap((c) => c.y.filter((_, k) => c.t[k] >= tMin && c.t[k] <= tMax));
  let lo = Math.min(...ys, base ?? Infinity); let hi = Math.max(...ys, final ?? -Infinity);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (stepSize != null) {
    if (zoomed) {
      // Zoom on the overshoot: the top third of the step and everything above
      // it. The rising edge enters from below; the peak and the +-2 % band fill
      // the window instead of hiding in a 1 N bump on a 54 N axis.
      const peaks = [...runs, ...before.filter(Boolean)].map((r) => r?.metrics?.[i]?.peak_N ?? final);
      const top = Math.max(...peaks.map((pk) => (normalise ? pk / runs[0].T_ref_base_N[i] : pk)), final);
      lo = final - 0.3 * stepSize;
      hi = Math.max(top + 0.6 * (top - final), final + 0.06 * stepSize);
    } else {
      lo = Math.min(lo, base - 0.05 * stepSize); hi = Math.max(hi, final + 0.12 * stepSize);
    }
  }
  const margin = (hi - lo || 1) * 0.06; lo -= margin; hi += margin;
  const x = (t) => pad.l + ((t - tMin) / (tMax - tMin)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const path = (c) => {
    if (!c) return '';
    let d = ''; let started = false;
    c.t.forEach((t, k) => {
      if (t < tMin || t > tMax || !Number.isFinite(c.y[k])) return;
      d += `${started ? 'L' : 'M'}${x(t).toFixed(1)},${y(c.y[k]).toFixed(1)}`; started = true;
    });
    return d;
  };
  const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
  const unit = normalise ? '× T_ref' : 'N';
  const settleEnd = metrics.settle != null ? step + metrics.settle : null;
  const yTicks = Array.from({ length: 5 }, (_, k) => lo + margin + (k / 4) * (hi - lo - 2 * margin));

  return (
    <div className="step-chart">
      <div className="step-plot" ref={box}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} className="step-bg" />
        {settleEnd != null && (
          <>
            <rect x={x(step)} y={pad.t} width={Math.max(0, x(Math.min(settleEnd, tMax)) - x(step))} height={H - pad.t - pad.b} className="step-settling" />
            {settleEnd < tMax && <rect x={x(settleEnd)} y={pad.t} width={x(tMax) - x(settleEnd)} height={H - pad.t - pad.b} className="step-stable" />}
          </>
        )}
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} className="step-grid" />
            <text x={pad.l - 6} y={y(v) + 4} className="step-tick" textAnchor="end">{fmt(v, normalise ? 3 : 1)}</text>
          </g>
        ))}
        {Array.from({ length: 11 }, (_, k) => step + k).filter((t) => t >= tMin && t <= tMax).map((t) => (
          <text key={t} x={x(t)} y={H - pad.b + 15} className="step-tick" textAnchor="middle">{fmt(t - step, 0)}</text>
        ))}
        <text x={(pad.l + W - pad.r) / 2} y={H - 5} className="step-tick" textAnchor="middle">time after the step (s)</text>
        <text transform={`translate(13 ${(pad.t + H - pad.b) / 2}) rotate(-90)`} className="step-tick" textAnchor="middle">tension ({unit})</text>
        {final != null && stepSize != null && (
          <>
            <line x1={pad.l} x2={W - pad.r} y1={y(final)} y2={y(final)} className="step-ref" />
            <text x={W - pad.r - 4} y={y(final) - 4} className="step-tick" textAnchor="end">set-point {fmt(final, normalise ? 2 : 1)}</text>
            <line x1={pad.l} x2={W - pad.r} y1={y(final + 0.02 * stepSize)} y2={y(final + 0.02 * stepSize)} className="step-band" />
            <line x1={pad.l} x2={W - pad.r} y1={y(final - 0.02 * stepSize)} y2={y(final - 0.02 * stepSize)} className="step-band" />
          </>
        )}
        {metrics.tau != null && (
          <line x1={x(step + metrics.tau)} x2={x(step + metrics.tau)} y1={pad.t} y2={H - pad.b} className="step-tau" />
        )}
        {faint.map((c, k) => <path key={k} d={path(c)} className="step-faint" />)}
        {prior && <path d={path(prior)} className="step-before" />}
        {main && <path d={path(main)} className={`step-after tone-${tone}`} />}
        {main && metrics.peak != null && metrics.overshoot > 0.05 && (() => {
          const px = x(step + metrics.peakT); const py = y(normalise ? metrics.peak / (runs[0].T_ref_base_N[i]) : metrics.peak);
          const lx = Math.min(W - pad.r - 150, px + 70); let ly = Math.max(pad.t + 18, py - 26);
          // The set-point label sits at y(final) - 4 on the right; when the peak is
          // near the top the two labels share a band and overprint. Drop ours below it.
          if (Math.abs((ly - 4) - (y(final) - 4)) < 14) ly = y(final) + 18;
          return (
            <g className="step-os">
              <line x1={lx} y1={ly} x2={px + 2} y2={py - 2} />
              <circle cx={px} cy={py} r={3} />
              <text x={lx + 3} y={ly - 4}>overshoot {fmt(metrics.overshoot, 1)} %</text>
            </g>
          );
        })()}
        {!main && <text x={W / 2} y={H / 2} className="step-empty" textAnchor="middle">{note ?? 'press OPTIMISE GAIN'}</text>}
      </svg>
      </div>
      <div className="step-legend-row">
        <span><i className={`step-key is-line tone-${tone}`} />{afterLabel}</span>
        {prior && <span><i className="step-key is-dashed" />{beforeLabel}</span>}
        <span><i className="step-key is-settling" />settling window</span>
        <span><i className="step-key is-stable" />stable</span>
        <span><i className="step-key is-tau" />time constant (63 %)</span>
        <span><i className="step-key is-band" />±2 % band</span>
        {runs[0]?.measured && <span className="hint">measured through the sensor: noise σ = {fmt(runs[0].noise_sigma_N, 2)} N, LPF {runs[0].lpf_hz ?? 'none'} Hz</span>}
      </div>
      <div className="step-readout">
        <span>overshoot <b>{fmt(metrics.overshoot, 1)} %</b> <small>(true tension)</small></span>
        <span>time constant <b>{fmt(metrics.tau, 3)} s</b></span>
        <span>time outside ±2 % <b>{fmt(metrics.settle, 2)} s</b></span>
        {runs.length > 1 && <span className="hint">median of {runs.length} plants, tension ÷ T_ref</span>}
      </div>
    </div>
  );
}
