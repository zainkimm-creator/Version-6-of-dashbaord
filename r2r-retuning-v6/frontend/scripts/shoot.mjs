// Screenshot the dashboard the way a user sees it: a real browser, a real
// click on OPTIMISE GAIN, real values in every window.
//
//   node scripts/shoot.mjs <outDir> [skin ...]
//
// Needs the dashboard up (vite on 5298, backend on 8024) and `google-chrome` on
// PATH. Drives Chrome over the DevTools protocol with node's built-in WebSocket,
// so there is nothing to install.
//
// Written because the alternative -- `--screenshot`, which cannot click -- only
// ever captured the empty screen, and the screenshots in docs/ui-options/ are
// supposed to show the loop having run.
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const OUT = process.argv[2] || '.';
const SKINS = process.argv.slice(3);
const URL_BASE = process.env.DASH_URL || 'http://127.0.0.1:5298';
const PORT = Number(process.env.CDP_PORT || 9333);
const WIDTH = Number(process.env.SHOT_W || 1500);
const HEIGHT = Number(process.env.SHOT_H || 1500);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForChrome() {
  for (let i = 0; i < 60; i += 1) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (r.ok) return await r.json();
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error('Chrome did not open its debugging port');
}

// One CDP session over one target, with the request/response plumbing.
function session(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const pending = new Map();
  const waiters = [];
  let next = 1;
  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method) {
      for (let i = waiters.length - 1; i >= 0; i -= 1) {
        if (waiters[i].method === msg.method) { waiters[i].resolve(msg.params); waiters.splice(i, 1); }
      }
    }
  });
  const open = new Promise((resolve) => ws.addEventListener('open', resolve));
  return {
    open,
    send: (method, params = {}) => new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    }),
    once: (method, timeoutMs = 30000) => new Promise((resolve, reject) => {
      waiters.push({ method, resolve });
      setTimeout(() => reject(new Error(`timed out waiting for ${method}`)), timeoutMs);
    }),
    close: () => ws.close(),
  };
}

// A private profile directory: Chrome refuses to open a second debugging
// session against a profile another Chrome still holds, and the failure is
// silent unless its stderr is left connected.
const profile = mkdtempSync(join(tmpdir(), 'r2r-shot-'));
const chrome = spawn('google-chrome', [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
  '--disable-dev-shm-usage', `--user-data-dir=${profile}`,
  `--remote-debugging-port=${PORT}`,
  `--window-size=${WIDTH},${HEIGHT}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'inherit'] });

try {
  await waitForChrome();
  mkdirSync(OUT, { recursive: true });

  for (const spec of SKINS) {
    // "slate" or "slate:look-2-slate" or "slate:machine:400" (a shorter frame)
    const [skin, name, heightOverride] = spec.split(':');
    const file = `${name || skin}.png`;
    const height = Number(heightOverride) || HEIGHT;

    const target = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
    const s = session(target.webSocketDebuggerUrl);
    await s.open;
    await s.send('Page.enable');
    await s.send('Runtime.enable');
    await s.send('Emulation.setDeviceMetricsOverride',
                 { width: WIDTH, height, deviceScaleFactor: 1, mobile: false });

    const loaded = s.once('Page.loadEventFired');
    await s.send('Page.navigate', { url: `${URL_BASE}/?skin=${skin}` });
    await loaded;

    // Wait for the app to reach the backend before touching anything. A fixed
    // sleep is not enough: after a backend restart the first requests are slow
    // (JAX import, atlas load) and the screen sits on OFFLINE / "Loading
    // session..." for ten seconds or more. Clicking then silently captures an
    // error screen instead of the dashboard.
    let online = false;
    for (let i = 0; i < 60; i += 1) {
      const ready = await s.send('Runtime.evaluate', {
        expression: `(() => {
          const b = [...document.querySelectorAll('button')]
            .find((el) => el.textContent.trim().toUpperCase().startsWith('1 · RUN PHYSICAL MACHINE'));
          return !!b && !b.disabled && !/OFFLINE/.test(document.body.innerText);
        })()`,
        returnByValue: true,
      });
      if (ready.result.value) { online = true; break; }
      await sleep(1000);
    }
    if (!online) console.log(`  ${file}: backend never came online — capturing anyway`);

    // PICK_CASE=caseA selects a worked case in the plant selector first.
    if (process.env.PICK_CASE) {
      await s.send('Runtime.evaluate', {
        expression: `(() => {
          const sel = [...document.querySelectorAll('select')].find((el) => [...el.options].some((o) => o.value === ${JSON.stringify(process.env.PICK_CASE)}));
          if (!sel) return false;
          const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value').set;
          setter.call(sel, ${JSON.stringify(process.env.PICK_CASE)});
          sel.dispatchEvent(new Event('change', { bubbles: true }));
          return true;
        })()`,
        returnByValue: true,
      });
      await sleep(1500);
    }
    // The loop is three buttons in order; click each and wait for its
    // "n / n done in N s" line before the next.
    let ran = true;
    for (const label of ['1 · RUN PHYSICAL MACHINE', '2 · OPTIMISE GAIN', '3 · APPLY TO THE MACHINE']) {
      const clicked = await s.send('Runtime.evaluate', {
        expression: `(() => {
          const b = [...document.querySelectorAll('button')]
            .find((el) => el.textContent.trim().toUpperCase().startsWith(${JSON.stringify(label.toUpperCase())}));
          if (!b || b.disabled) return false;
          b.click(); return true;
        })()`,
        returnByValue: true,
      });
      if (!clicked.result.value) { console.log(`  ${file}: could not click ${label}`); ran = false; break; }
      let finished = false;
      for (let i = 0; i < 240; i += 1) {   // identify + retune ~80 s per plant
        await sleep(500);
        const probe = await s.send('Runtime.evaluate', {
          expression: `(() => {
            const b = [...document.querySelectorAll('button')]
              .find((el) => el.textContent.trim().toUpperCase().startsWith(${JSON.stringify(label.toUpperCase())}));
            return !!b && !b.disabled && /\\d+ \\/ \\d+ done in \\d+ s/.test(document.body.innerText);
          })()`,
          returnByValue: true,
        });
        if (probe.result.value) { finished = true; break; }
      }
      if (!finished) { console.log(`  ${file}: ${label} did not finish`); ran = false; break; }
    }
    await sleep(1500);   // let the animation settle into a representative frame

    const shot = await s.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    writeFileSync(join(OUT, file), Buffer.from(shot.data, 'base64'));
    console.log(`  ${file}  ${ran ? 'loop ran' : 'NO LOOP RESULT'}  (${skin})`);
    await s.send('Page.close').catch(() => {});
    s.close();
  }
} finally {
  chrome.kill();
  await sleep(500);   // Chrome flushes its profile on the way out
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* a temp dir; leave it */ }
}
