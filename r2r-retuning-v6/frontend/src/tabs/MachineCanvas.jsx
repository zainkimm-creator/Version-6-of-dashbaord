import { useEffect, useRef } from 'react';

// The line drawn the way the paper draws it (Fig. 1), in motion.
//
// Topology and every annotation follow Fig. 1 of the paper: unwinder (UW) ->
// in-feeder (velocity master, v0) -> out-feeder (nip roller, tension control) ->
// rewinder (RW), with the web running straight through both roller pairs, the
// span tensions T1..T3 above the web, the span lengths L1..L3 below it, the
// control inputs u_UW / u_Nip / u_RW entering from above and the roller speeds
// omega_UW / omega_Nip / omega_RW leaving below.
//
// What the paper cannot show is movement. Tensions, set-points and roller speeds
// are a real simulated identification record (POST /plant/trace), played in real
// time by the parent. The motion is kinematically consistent: every roller's
// surface speed equals the web speed (no slip, so smaller rollers spin faster)
// and the reels trade material with the wound area conserved -- the unwinder
// shrinks, the rewinder grows, and both spin faster as they shrink. Only the
// on-screen web speed is scaled, so a 0.1 m/s line still visibly moves.
// Animation runs on requestAnimationFrame straight into the SVG attributes,
// never through React state.

const W = 1000;
const H = 364;

const WEB_Y = 150;              // the straight web line, Fig. 1
const UW_X = 140;
const FEED_X = 396;             // in-feeder: velocity master, v0
const NIP_X = 628;              // out-feeder: nip roller, tension control
const RW_X = 872;

// The reel swing is deliberately narrow: the roll must visibly grow and shrink,
// but a reel that halves in size drags its speed arrow far down the canvas and
// leaves a hole under the line.
const CORE_R = 26;              // an empty reel, px
const REEL_MAX = 56;            // a full reel, px
// A reel hangs below the web and always touches it: centre y = WEB_Y + radius.
// That is what keeps the web dead straight, as Fig. 1 draws it, while the reels
// still visibly trade material. The animation loop moves the reel, its speed
// arrow and its label together.
const reelCy = (r) => WEB_Y + r;
const FEED_R = 27;
const NIP_TOP_R = 25;
const NIP_BOT_R = 30;

const REEL_CYCLE_PX = 42000;    // web travel for one full unwind (several minutes on screen)
const START_PHASE = 0.35;       // how much has been unwound when the screen opens

// The three spans, as Fig. 1 divides them: UW->feeder, feeder->nip, nip->RW.
const SPAN_MID = [(UW_X + FEED_X) / 2, (FEED_X + NIP_X) / 2, (NIP_X + RW_X) / 2];

// Screen speed of the web, px/s: monotone in the real line speed, compressed.
export function screenSpeedPx(v0) {
  const v = Math.max(0, Number(v0) || 0);
  return Math.min(240, 36 + 70 * Math.log10(1 + 10 * v));
}

// Rollers the web runs between. Each pair pinches the web, so the pinch point is
// the web line itself. dir: +1 spins clockwise on screen, -1 counter-clockwise.
// The web travels left to right, so the roller above it turns counter-clockwise
// and the roller below it turns clockwise.
const ROLLERS = [
  { id: 'feedTop', x: FEED_X, y: WEB_Y - FEED_R, r: FEED_R, dir: -1, kind: 'idler' },
  { id: 'feedBottom', x: FEED_X, y: WEB_Y + FEED_R, r: FEED_R, dir: 1, kind: 'feed' },
  { id: 'nipTop', x: NIP_X, y: WEB_Y - NIP_TOP_R, r: NIP_TOP_R, dir: -1, kind: 'driven' },
  { id: 'nipBottom', x: NIP_X, y: WEB_Y + NIP_BOT_R, r: NIP_BOT_R, dir: 1, kind: 'driven' },
];

function reelRadii(phase) {
  const area = REEL_MAX ** 2 - CORE_R ** 2;
  return {
    uw: Math.sqrt(REEL_MAX ** 2 - area * phase),
    rw: Math.sqrt(CORE_R ** 2 + area * phase),
  };
}

// The web's contact circles in travel order. A radius of 0 is a pinch between a
// roller pair, which is also where one span ends and the next begins.
function webCircles(phase) {
  const { uw, rw } = reelRadii(phase);
  return [
    { x: UW_X, y: reelCy(uw), r: uw, side: 1 },
    { x: FEED_X, y: WEB_Y, r: 0, side: 1 },
    { x: NIP_X, y: WEB_Y, r: 0, side: 1 },
    { x: RW_X, y: reelCy(rw), r: rw, side: 1 },
  ];
}

// Common tangent from circle a to circle b, with signed radii for the wrap side.
function tangent(a, b) {
  const R1 = a.side * a.r;
  const R2 = b.side * b.r;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const L = Math.hypot(dx, dy) || 1;
  const ux = dx / L; const uy = dy / L;
  const vx = -uy; const vy = ux;
  const k = (R1 - R2) / L;
  const m = -Math.sqrt(Math.max(0, 1 - k * k));
  const nx = k * ux + m * vx; const ny = k * uy + m * vy;
  return [[a.x + R1 * nx, a.y + R1 * ny], [b.x + R2 * nx, b.y + R2 * ny]];
}

// Three span paths with their lengths, so travelling marks stay continuous
// across spans. The middle span is exactly horizontal; the two end spans leave
// the reel surface, so they tilt very slightly as the reels trade material.
export function webGeometry(phase) {
  const c = webCircles(phase);
  const pt = (p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`;
  const spans = [0, 1, 2].map((i) => {
    const [p1, p2] = tangent(c[i], c[i + 1]);
    return { d: `M${pt(p1)} L${pt(p2)}`, length: Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) };
  });
  const starts = [0, spans[0].length, spans[0].length + spans[1].length];
  return { spans, starts };
}

function fmt(value, digits = 2) {
  const v = Number(value);
  return Number.isFinite(v) ? v.toFixed(digits) : '----';
}

// T1..T3 above the web, where Fig. 1 puts them, carrying the live values.
function TensionChip({ x, label, pv, sv, base, twin }) {
  const stepped = Number.isFinite(sv) && Number.isFinite(base) && Math.abs(sv - base) > 1e-9;
  const twoLine = twin !== null && twin !== undefined;
  const h = twoLine ? 40 : 26;
  return (
    <g transform={`translate(${x - 59} ${WEB_Y - 18 - h})`} className={`hmi-chip${stepped ? ' is-stepped' : ''}`}>
      <rect width={118} height={h} rx={3} className="hmi-chip-bg" />
      <text x={9} y={17} className="hmi-chip-label is-tension">{label}</text>
      <text x={109} y={17} className="hmi-chip-value" textAnchor="end">{fmt(pv)} N</text>
      {twoLine && (
        <>
          <text x={9} y={33} className="hmi-chip-label">twin</text>
          <text x={109} y={33} className="hmi-chip-twin" textAnchor="end">{fmt(twin)} N</text>
        </>
      )}
      {stepped && <text x={59} y={-5} className="hmi-chip-step" textAnchor="middle">STEP +20 %</text>}
    </g>
  );
}

// A control input entering from above, as Fig. 1 draws it.
function InputArrow({ x, toY, label }) {
  return (
    <g className="hmi-arrow is-input">
      <text x={x} y={48} textAnchor="middle" className="hmi-arrow-label">{label}</text>
      <line x1={x} y1={56} x2={x} y2={toY} markerEnd="url(#hmi-arrow-u)" />
    </g>
  );
}

// A roller speed leaving below, as Fig. 1 draws it. The two reel arrows move
// with their reel, so they are driven by refs rather than props.
function SpeedArrow({ x, fromY, label, lineRef, textRef }) {
  return (
    <g className="hmi-arrow is-speed">
      <line ref={lineRef} x1={x} y1={fromY} x2={x} y2={fromY + 28} markerEnd="url(#hmi-arrow-w)" />
      <text ref={textRef} x={x} y={fromY + 44} textAnchor="middle" className="hmi-arrow-label">{label}</text>
    </g>
  );
}

// An identity plate -- UW, RW, Nip -- legible over a roller in any skin, which
// bare text on a metal fill is not.
function LabelPlate({ x = 0, y = 0, text, w = 36 }) {
  return (
    <g transform={`translate(${x} ${y})`} className="hmi-plate">
      <rect x={-w / 2} y={-10} width={w} height={20} rx={3} className="hmi-plate-bg" />
      <text y={5} textAnchor="middle" className="hmi-plate-text">{text}</text>
    </g>
  );
}

function Roller({ spec, spinRef }) {
  const { x, y, r, kind } = spec;
  return (
    <g transform={`translate(${x} ${y})`} className={`hmi-roll is-${kind}`}>
      <circle r={r} className="hmi-roll-shell" />
      {kind === 'driven' && <circle r={r * 0.74} className="hmi-roll-rubber" />}
      <g ref={spinRef}>
        <circle r={Math.max(2.2, r * 0.2)} className="hmi-roll-hub" />
        {[0, 120, 240].map((a) => (
          <circle key={a} r={Math.max(1, r * 0.07)} className="hmi-roll-bolt"
                  transform={`rotate(${a}) translate(${(r * 0.55).toFixed(2)} 0)`} />
        ))}
        <line x1={r * 0.28} y1={0} x2={r * 0.88} y2={0} className="hmi-roll-mark" />
      </g>
    </g>
  );
}

// A reel: wound material, the outermost wrap of web around it, and a core that
// turns. The whole group slides down as the reel grows so its top stays on the
// web; the outer radius, the web ring and the group position are all driven by
// the animation loop.
function Reel({ x, r, label, groupRef, outerRef, webRef, spinRef }) {
  return (
    <g ref={groupRef} transform={`translate(${x} ${reelCy(r)})`} className="hmi-reel">
      <circle ref={outerRef} r={r} className="hmi-reel-wound" fill="url(#hmi-wound)" />
      <circle ref={webRef} r={r} className="hmi-reel-web" />
      <g ref={spinRef}>
        <circle r={CORE_R} className="hmi-reel-core" />
        {[0, 90, 180, 270].map((a) => (
          <circle key={a} r={2} className="hmi-roll-bolt"
                  transform={`rotate(${a}) translate(${(CORE_R * 0.72).toFixed(2)} 0)`} />
        ))}
        <line x1={CORE_R} y1={0} x2={CORE_R + 8} y2={0} className="hmi-reel-splice" />
      </g>
      <LabelPlate text={label} />
    </g>
  );
}

export default function MachineCanvas({ trace, frame = 0, playing = true, v0 }) {
  const speed = screenSpeedPx(v0 ?? trace?.line_speed_m_s);
  const T = trace?.T_N?.[frame] ?? [];
  const Tref = trace?.T_ref_N?.[frame] ?? [];
  const twin = trace?.twin_T_N?.[frame] ?? null;
  const base = trace?.T_ref_base_N ?? [];
  const step = Number(trace?.step_N) || 0;

  const speedRef = useRef(speed);
  speedRef.current = speed;
  const motion = useRef({ phase: START_PHASE, offset: 0, angles: {} });
  const spinRefs = useRef({});
  const reelRefs = useRef({});
  const reelWebRefs = useRef({});
  const reelGroupRefs = useRef({});
  const speedRefs = useRef({ uw: {}, rw: {} });
  const webRefs = useRef({ edge: [], film: [], marks: [] });

  // Static geometry for the first paint (and for server rendering); the
  // animation loop owns these attributes from then on.
  const initial = webGeometry(START_PHASE);
  const initialReels = reelRadii(START_PHASE);

  useEffect(() => {
    if (!playing || typeof window === 'undefined' || typeof requestAnimationFrame === 'undefined') return undefined;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return undefined;
    let raf;
    let last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const v = speedRef.current;
      const m = motion.current;
      m.phase = (m.phase + (v * dt) / REEL_CYCLE_PX) % 1;
      m.offset -= v * dt;

      const geo = webGeometry(m.phase);
      geo.spans.forEach((span, i) => {
        webRefs.current.edge[i]?.setAttribute('d', span.d);
        webRefs.current.film[i]?.setAttribute('d', span.d);
        const marks = webRefs.current.marks[i];
        if (marks) {
          marks.setAttribute('d', span.d);
          marks.setAttribute('stroke-dashoffset', (m.offset + geo.starts[i]).toFixed(2));
        }
      });

      const radii = reelRadii(m.phase);
      [['uw', radii.uw, UW_X], ['rw', radii.rw, RW_X]].forEach(([id, r, x]) => {
        const rr = r.toFixed(2);
        reelRefs.current[id]?.setAttribute('r', rr);
        reelWebRefs.current[id]?.setAttribute('r', rr);
        reelGroupRefs.current[id]?.setAttribute('transform', `translate(${x} ${reelCy(r).toFixed(2)})`);
        // the speed arrow hangs below the reel centre, so it clears the rim at
        // every radius while moving only half as far as the rim itself
        const top = WEB_Y + r + REEL_MAX + 8;
        speedRefs.current[id]?.line?.setAttribute('y1', top.toFixed(2));
        speedRefs.current[id]?.line?.setAttribute('y2', (top + 28).toFixed(2));
        speedRefs.current[id]?.text?.setAttribute('y', (top + 44).toFixed(2));
        m.angles[id] = ((m.angles[id] ?? 0) + (v / r) * dt * (180 / Math.PI)) % 360;
        spinRefs.current[id]?.setAttribute('transform', `rotate(${m.angles[id].toFixed(2)})`);
      });
      ROLLERS.forEach((spec) => {
        m.angles[spec.id] = ((m.angles[spec.id] ?? 0) + spec.dir * (v / spec.r) * dt * (180 / Math.PI)) % 360;
        spinRefs.current[spec.id]?.setAttribute('transform', `rotate(${m.angles[spec.id].toFixed(2)})`);
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const width = (i) => {
    const dev = step > 0 && Number.isFinite(T[i]) && Number.isFinite(base[i])
      ? Math.min(1, Math.abs(T[i] - base[i]) / step) : 0;
    return 5 + 2.5 * dev;
  };
  const steppedSpan = (i) => Number.isFinite(Tref[i]) && Number.isFinite(base[i]) && Math.abs(Tref[i] - base[i]) > 1e-9;

  return (
    <svg className="hmi-machine" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Three-span roll-to-roll web tension system in motion: unwinder, in-feeder, out-feeder nip and rewinder, as in Figure 1 of the paper">
      <defs>
        <radialGradient id="hmi-wound" gradientUnits="userSpaceOnUse" cx="0" cy="0" r="3.2" spreadMethod="repeat">
          <stop offset="0" stopColor="#e2d9b3" />
          <stop offset="0.55" stopColor="#f5efd6" />
          <stop offset="1" stopColor="#e2d9b3" />
        </radialGradient>
        <radialGradient id="hmi-metal" cx="0.38" cy="0.35" r="0.75">
          <stop offset="0" stopColor="#fbfbf7" />
          <stop offset="0.6" stopColor="#c9cbc4" />
          <stop offset="1" stopColor="#8e918a" />
        </radialGradient>
        <marker id="hmi-arrow-u" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,1 L9,5 L0,9 z" className="hmi-arrow-head is-input" />
        </marker>
        <marker id="hmi-arrow-w" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,1 L9,5 L0,9 z" className="hmi-arrow-head is-speed" />
        </marker>
        <marker id="hmi-arrow-flow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,1 L9,5 L0,9 z" className="hmi-arrow-head is-flow" />
        </marker>
      </defs>

      <rect x={0} y={0} width={W} height={H} className="hmi-machine-bg" />

      {/* web transport direction, along the top -- Fig. 1 */}
      <text x={W / 2} y={20} textAnchor="middle" className="hmi-flow-label">Web transport direction</text>
      <line x1={90} y1={30} x2={946} y2={30} className="hmi-flow-arrow" markerEnd="url(#hmi-arrow-flow)" />

      {/* control inputs, entering from above */}
      <InputArrow x={UW_X} toY={WEB_Y - 12} label="u_UW" />
      <InputArrow x={NIP_X} toY={WEB_Y - 2 * NIP_TOP_R - 10} label="u_Nip" />
      <InputArrow x={RW_X} toY={WEB_Y - 12} label="u_RW" />

      {/* the web: dark edge, the film itself, and printed marks that travel with it */}
      {initial.spans.map((span, i) => (
        <g key={i} className={`hmi-span${steppedSpan(i) ? ' is-stepped' : ''}`}>
          <path ref={(el) => { webRefs.current.edge[i] = el; }} d={span.d}
                className="hmi-web-edge" strokeWidth={width(i) + 2.2} />
          <path ref={(el) => { webRefs.current.film[i] = el; }} d={span.d}
                className="hmi-web-film" strokeWidth={width(i)} />
          <path ref={(el) => { webRefs.current.marks[i] = el; }} d={span.d}
                className="hmi-web-marks" strokeWidth={width(i) * 0.75}
                strokeDasharray="1.6 17.4" strokeDashoffset={initial.starts[i]} />
        </g>
      ))}

      <Reel x={UW_X} r={initialReels.uw} label="UW"
            groupRef={(el) => { reelGroupRefs.current.uw = el; }}
            outerRef={(el) => { reelRefs.current.uw = el; }}
            webRef={(el) => { reelWebRefs.current.uw = el; }}
            spinRef={(el) => { spinRefs.current.uw = el; }} />
      <Reel x={RW_X} r={initialReels.rw} label="RW"
            groupRef={(el) => { reelGroupRefs.current.rw = el; }}
            outerRef={(el) => { reelRefs.current.rw = el; }}
            webRef={(el) => { reelWebRefs.current.rw = el; }}
            spinRef={(el) => { spinRefs.current.rw = el; }} />
      {ROLLERS.map((spec) => (
        <Roller key={spec.id} spec={spec} spinRef={(el) => { spinRefs.current[spec.id] = el; }} />
      ))}

      {/* the two driven stations name themselves, as in Fig. 1 */}
      <LabelPlate x={FEED_X} y={WEB_Y - FEED_R} text="Feeder" w={54} />
      <LabelPlate x={NIP_X} y={WEB_Y - NIP_TOP_R} text="Nip" w={34} />

      {/* span tensions above the web, span lengths below -- Fig. 1 */}
      <TensionChip x={SPAN_MID[0]} label="T₁" pv={T[0]} sv={Tref[0]} base={base[0]} twin={twin?.[0]} />
      <TensionChip x={SPAN_MID[1]} label="T₂" pv={T[1]} sv={Tref[1]} base={base[1]} twin={twin?.[1]} />
      <TensionChip x={SPAN_MID[2]} label="T₃" pv={T[2]} sv={Tref[2]} base={base[2]} twin={twin?.[2]} />
      {['L₁', 'L₂', 'L₃'].map((label, i) => (
        <text key={label} x={SPAN_MID[i]} y={WEB_Y + 28} textAnchor="middle" className="hmi-span-length">{label}</text>
      ))}

      {/* the in-feeder is the velocity master: v0 is prescribed, not controlled */}
      <text x={FEED_X} y={WEB_Y + 2 * FEED_R + 22} textAnchor="middle" className="hmi-bc">v₀ prescribed (fixed BC)</text>

      {/* roller speeds, leaving below */}
      <SpeedArrow x={UW_X} fromY={WEB_Y + initialReels.uw + REEL_MAX + 8} label="ω_UW"
                  lineRef={(el) => { speedRefs.current.uw.line = el; }}
                  textRef={(el) => { speedRefs.current.uw.text = el; }} />
      <SpeedArrow x={NIP_X} fromY={WEB_Y + 2 * NIP_BOT_R + 12} label="ω_Nip" />
      <SpeedArrow x={RW_X} fromY={WEB_Y + initialReels.rw + REEL_MAX + 8} label="ω_RW"
                  lineRef={(el) => { speedRefs.current.rw.line = el; }}
                  textRef={(el) => { speedRefs.current.rw.text = el; }} />

      {/* the four roles, in the paper's words */}
      {[[UW_X, 'unwinder', ''], [FEED_X, 'in-feeder', '(velocity master)'],
        [NIP_X, 'out-feeder', '(tension control)'], [RW_X, 'rewinder', '']].map(([x, role, note]) => (
        <g key={role}>
          <text x={x} y={336} textAnchor="middle" className="hmi-role">{role}</text>
          {note && <text x={x} y={351} textAnchor="middle" className="hmi-role is-note">{note}</text>}
        </g>
      ))}
    </svg>
  );
}
