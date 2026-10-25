import { useCallback, useEffect, useRef, useState } from "react";
import { createReplay, replayDashboardUrl, type Snapshot } from "./api";
import { historyUpTo, loadOfflineRace, raceDuration, snapshotAtIndex, tickIndexAt, type UsePoint } from "./offlineReplay";

export type ReplayPhase = "idle" | "connecting" | "replaying" | "paused" | "finished" | "offline";
/** "backend": ticks streamed by the API. "offline": the recorded copy in public/replays, played in the browser. */
export type ReplaySource = "backend" | "offline";
export type { UsePoint };

export type ReplayState = {
  phase: ReplayPhase;
  snapshot: Snapshot | null;
  error: string | null;
  source: ReplaySource | null;
  playing: boolean;
  speed: number;
  /** Race seconds shown now, and the race length (header.budget_s). */
  t: number;
  duration: number;
  /** Machines training per received tick (both sides), for the machine-use sparkline. */
  history: UsePoint[];
};

export type ReplayControls = {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  restart: () => void;
  seek: (raceS: number) => void;
  setSpeed: (speed: number) => void;
};

type Cmd = { id: number; from: number; speed: number; play: boolean };

const IDLE = (speed: number): ReplayState => ({
  phase: "idle",
  snapshot: null,
  error: null,
  source: null,
  playing: false,
  speed,
  t: 0,
  duration: 0,
  history: [],
});

/**
 * Replays a logged race. First choice: the backend (POST /api/races/{prefix}/replay, then the
 * /ws/replays/{id}/dashboard stream, with start_s for seeks). If the backend is unreachable, or the
 * stream stops, it plays the recorded offline copy (public/replays/{prefix}.json) in the browser
 * from the same race second, at the same speed. Both give the same Snapshot shape.
 * Every number on screen comes from these snapshots. Nothing is estimated here.
 */
export function useReplay(racePrefix: string | null, initialSpeed = 90): ReplayState & ReplayControls {
  const [cmd, setCmd] = useState<Cmd>({ id: 0, from: 0, speed: initialSpeed, play: true });
  const [state, setState] = useState<ReplayState>(() => IDLE(initialSpeed));
  const tRef = useRef(0);
  const finishedRef = useRef(false);
  const hasSnapRef = useRef(false);
  const offlineRef = useRef(false);
  const prefixRef = useRef<string | null>(null);
  const historyRef = useRef<UsePoint[]>([]);

  useEffect(() => {
    if (!racePrefix) {
      prefixRef.current = null;
      setState(IDLE(cmd.speed));
      return;
    }
    let from = cmd.from;
    let play = cmd.play;
    if (prefixRef.current !== racePrefix) {
      prefixRef.current = racePrefix;
      offlineRef.current = false;
      hasSnapRef.current = false;
      finishedRef.current = false;
      historyRef.current = [];
      tRef.current = 0;
      from = 0;
      play = true;
      setState({ ...IDLE(cmd.speed), phase: "connecting" });
    }
    const speed = cmd.speed;
    let disposed = false;
    const cleanups: (() => void)[] = [];

    // Pause on the frame already shown: nothing to fetch.
    if (!play && hasSnapRef.current && Math.abs(from - tRef.current) < 0.01) {
      setState((s) => ({ ...s, playing: false, phase: s.phase === "finished" ? "finished" : "paused", speed }));
      return;
    }

    const runOffline = (fromS: number, err: string | null) => {
      offlineRef.current = true;
      loadOfflineRace(racePrefix)
        .then((race) => {
          if (disposed) return;
          const dur = raceDuration(race);
          let t = Math.min(Math.max(0, fromS), dur);
          let idx = -1;
          const emit = (playing: boolean): boolean => {
            const i = tickIndexAt(race, t);
            const finished = race.ticks[i][2] === "finished";
            if (i === idx) return finished;
            idx = i;
            const snap = snapshotAtIndex(race, i, speed);
            tRef.current = t;
            finishedRef.current = finished;
            hasSnapRef.current = true;
            setState({
              phase: finished ? "finished" : playing ? "replaying" : "paused",
              snapshot: snap,
              error: err,
              source: "offline",
              playing: playing && !finished,
              speed,
              t: finished ? dur : t,
              duration: dur,
              history: historyUpTo(race, i),
            });
            return finished;
          };
          if (!play) {
            emit(false);
            return;
          }
          if (emit(true)) return;
          let last = 0;
          let raf = 0;
          const frame = (now: number) => {
            if (disposed) return;
            if (last) t = Math.min(dur, t + ((now - last) / 1000) * speed);
            last = now;
            tRef.current = t;
            if (emit(true)) return;
            raf = requestAnimationFrame(frame);
          };
          raf = requestAnimationFrame(frame);
          cleanups.push(() => cancelAnimationFrame(raf));
        })
        .catch((e: unknown) => {
          if (disposed) return;
          const msg = `${err ? `${err}; ` : ""}${e instanceof Error ? e.message : String(e)}`;
          setState((s) => ({
            ...s,
            phase: s.snapshot ? (s.snapshot.status === "finished" ? "finished" : "paused") : "offline",
            playing: false,
            error: msg,
          }));
        });
    };

    const runBackend = (fromS: number) => {
      const abort = new AbortController();
      let ws: WebSocket | null = null;
      let got = false;
      let finished = false;
      cleanups.push(() => {
        abort.abort();
        if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
      });
      historyRef.current = fromS <= 0 ? [] : historyRef.current.filter((p) => p.t < fromS);
      setState((s) => ({ ...s, phase: s.snapshot ? s.phase : "connecting", playing: play, speed }));

      createReplay(racePrefix, speed, abort.signal)
        .then((created) => {
          if (disposed) return;
          const sock = new WebSocket(replayDashboardUrl(created.replay_id, 0.1, fromS));
          ws = sock;
          sock.onmessage = (ev) => {
            if (disposed) return;
            let m: { type?: string; snapshot?: Snapshot };
            try {
              m = JSON.parse(ev.data as string);
            } catch {
              return;
            }
            if (m.type !== "tick" || !m.snapshot) return;
            got = true;
            const snap = m.snapshot;
            const t = snap.race_elapsed_s;
            finished = snap.status === "finished";
            const h = historyRef.current;
            const left = snap.arms.find((a) => a.side === "left")?.working_now ?? 0;
            const right = snap.arms.find((a) => a.side === "right")?.working_now ?? 0;
            if (!h.length || t > h[h.length - 1].t) h.push({ t, left, right });
            tRef.current = t;
            finishedRef.current = finished;
            hasSnapRef.current = true;
            setState({
              phase: finished ? "finished" : play ? "replaying" : "paused",
              snapshot: snap,
              error: null,
              source: "backend",
              playing: play && !finished,
              speed,
              t,
              duration: snap.header.budget_s || created.race_duration_s,
              history: h.slice(),
            });
            // A paused seek needs one frame only.
            if (!play) sock.close();
          };
          sock.onclose = () => {
            if (disposed) return;
            if (!got) runOffline(fromS, "the replay stream closed before its first frame");
            else if (play && !finished) runOffline(tRef.current, "the replay stream stopped");
          };
        })
        .catch((e: unknown) => {
          if (disposed || abort.signal.aborted) return;
          runOffline(fromS, e instanceof Error ? e.message : String(e));
        });
    };

    if (offlineRef.current) runOffline(from, null);
    else runBackend(from);

    return () => {
      disposed = true;
      cleanups.forEach((f) => f());
    };
  }, [racePrefix, cmd]);

  const play = useCallback(
    () => setCmd((c) => ({ id: c.id + 1, from: finishedRef.current ? 0 : tRef.current, speed: c.speed, play: true })),
    [],
  );
  const pause = useCallback(() => setCmd((c) => ({ id: c.id + 1, from: tRef.current, speed: c.speed, play: false })), []);
  const restart = useCallback(() => setCmd((c) => ({ id: c.id + 1, from: 0, speed: c.speed, play: true })), []);
  const seek = useCallback(
    (s: number) => setCmd((c) => ({ id: c.id + 1, from: Math.max(0, s), speed: c.speed, play: c.play })),
    [],
  );
  const setSpeed = useCallback(
    (speed: number) =>
      setCmd((c) => (c.speed === speed ? c : { id: c.id + 1, from: tRef.current, speed, play: c.play && !finishedRef.current })),
    [],
  );
  const toggle = state.playing ? pause : play;

  return { ...state, play, pause, toggle, restart, seek, setSpeed };
}
