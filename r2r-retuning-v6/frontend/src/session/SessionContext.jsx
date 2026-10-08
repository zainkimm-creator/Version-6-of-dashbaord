import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { sessionApi } from './api.js';

const SessionCtx = createContext(null);

/**
 * Two objects, deliberately.
 *
 * `session` is server truth. `draft` is what the controls show. Dragging a
 * slider mutates the draft only, so exploring is free and never clobbers
 * state; Adopt / Apply drift / Load into twin are the only things that write
 * to the server. `dirty` is what the header's "unadopted changes" chip reads.
 */
export function SessionProvider({ baseUrl, children }) {
  const api = useMemo(() => sessionApi(baseUrl), [baseUrl]);
  const [session, setSession] = useState(null);
  const [draft, setDraft] = useState(null);
  const [presets, setPresets] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const absorb = useCallback((body) => {
    setSession(body);
    setDraft({ plant: body.plant, drift: body.drift, protocol: body.protocol });
    return body;
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([api.getSession(), api.getPresets()])
      .then(([sessionBody, presetBody]) => {
        if (cancelled) return;
        absorb(sessionBody);
        setPresets(presetBody.presets ?? {});
        setError(null);
      })
      .catch((cause) => !cancelled && setError(cause.message))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [api, absorb]);

  const guard = useCallback(async (work) => {
    try {
      const body = await work();
      setError(null);
      return body;
    } catch (cause) {
      setError(cause.message);
      throw cause;
    }
  }, []);

  const runSpec = useMemo(
    () => (draft ? { plant: draft.plant, drift: draft.drift, protocol: draft.protocol } : null),
    [draft],
  );

  const dirty = useMemo(() => {
    if (!session || !draft) return false;
    return JSON.stringify({ plant: session.plant, drift: session.drift, protocol: session.protocol })
      !== JSON.stringify(runSpec);
  }, [session, draft, runSpec]);

  // Memoized so unrelated App re-renders (e.g. activePage) don't invalidate
  // every useSession() consumer — the whole point of separating draft from
  // session is that exploring the controls stays cheap. guard/absorb are
  // stable useCallback refs (empty deps) so they're safe to omit here.
  const value = useMemo(() => ({
    session,
    draft,
    runSpec,
    presets,
    loading,
    error,
    // The tabs render `error` themselves; this is how a dismissed line goes
    // away, and how the next successful call is not confused with a stale
    // failure the user already read.
    clearError: () => setError(null),
    dirty,
    staleness: session?.staleness ?? {},
    setDraftPlant: (plant) => setDraft((d) => ({ ...d, plant })),
    setDraftDrift: (drift) => setDraft((d) => ({ ...d, drift })),
    setDraftProtocol: (protocol) => setDraft((d) => ({ ...d, protocol })),
    revertDraft: () => session && absorb(session),
    applyPlant: () => guard(() => api.putPlant(draft.plant).then(absorb)),
    applyDrift: () => guard(() => api.putDrift(draft.drift).then(absorb)),
    adoptProtocol: (patch) => guard(() => api.patchProtocol(patch).then(absorb)),
    // Swap is an input to /plant/derive, never a session lookup: the endpoint
    // is pure, and the theta-hat inversion belongs on the server.
    derive: () => guard(() => api.derive(
      runSpec,
      session?.twin?.swapped ? session?.theta_hat?.estimates ?? null : null,
    )),
    runIdentify: (commit = false) =>
      guard(async () => {
        const result = await api.identify(runSpec, commit);
        if (commit) absorb(await api.getSession());
        return result;
      }),
    paperCheck: () => guard(() => api.paperCheck(runSpec)),
    // Algorithm 1 Steps 4-7 on the draft: pure, never commits to the session.
    runRetune: (options = {}) => guard(() => api.retune(runSpec, options)),
    // Playback data for the machine screen. Not guarded: a record that cannot
    // be played (a diverging protocol) blanks the animation, it is not an alarm.
    fetchTrace: (estimates = null) => api.trace(runSpec, estimates),
    // Explicit-spec variants for the "all ten plants" loop: the draft's drift and
    // protocol with another preset plant swapped in. Unguarded like fetchTrace --
    // the caller shows progress and failures per plant.
    retuneFor: (spec, options = {}) => api.retune(spec, options),
    baselineFor: (spec, costTier = null, opts = {}) => api.baseline(spec, costTier, opts),
    fetchStepFor: (spec, gains, estimates = null, opts = {}) => api.step(spec, gains, estimates, opts),
    zoneGainsFor: (spec) => api.zoneGains(spec),
    forgetZoneGainsFor: (spec) => api.forgetZoneGains(spec),
    atlas: () => guard(() => api.getAtlas(
      draft.plant.preset_id ?? '', draft.protocol,
    )),
    loadTwin: (runHash) => guard(() => api.putTwin({ action: 'load', run_hash: runHash }).then(absorb)),
    swapTwin: () => guard(() => api.putTwin({ action: 'swap' }).then(absorb)),
    resetSession: () => guard(() => api.resetSession().then(absorb)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [session, draft, runSpec, presets, loading, error, dirty, api]);

  return <SessionCtx.Provider value={value}>{children}</SessionCtx.Provider>;
}

export function useSession() {
  const value = useContext(SessionCtx);
  if (!value) throw new Error('useSession must be used inside a SessionProvider');
  return value;
}
