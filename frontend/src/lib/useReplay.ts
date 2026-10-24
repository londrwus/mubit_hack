import { useEffect, useRef, useState } from "react";
import { createReplay, replayDashboardUrl, type Snapshot } from "./api";

export type ReplayPhase = "idle" | "connecting" | "replaying" | "finished" | "offline";

export type ReplayState = {
  phase: ReplayPhase;
  snapshot: Snapshot | null;
  error: string | null;
};

/**
 * Replays a logged race from the backend: POST /api/races/{prefix}/replay, then the
 * /ws/replays/{id}/dashboard stream. Keeps the latest tick snapshot. A new `runToken` restarts it.
 * Every number on screen comes from these snapshots. Nothing is estimated here.
 */
export function useReplay(racePrefix: string | null, speed: number, runToken: number): ReplayState {
  const [state, setState] = useState<ReplayState>({ phase: "idle", snapshot: null, error: null });
  const gotTick = useRef(false);

  useEffect(() => {
    if (!racePrefix) {
      setState({ phase: "idle", snapshot: null, error: null });
      return;
    }
    const abort = new AbortController();
    let ws: WebSocket | null = null;
    let closed = false;
    gotTick.current = false;
    setState({ phase: "connecting", snapshot: null, error: null });

    const fail = (msg: string) => {
      if (closed) return;
      setState((s) => {
        if (!gotTick.current) return { phase: "offline", snapshot: null, error: msg };
        return { ...s, phase: s.snapshot?.status === "finished" ? "finished" : "offline", error: msg };
      });
    };

    createReplay(racePrefix, speed, abort.signal)
      .then((created) => {
        if (closed) return;
        ws = new WebSocket(replayDashboardUrl(created.replay_id));
        ws.onmessage = (ev) => {
          let m: { type?: string; snapshot?: Snapshot };
          try {
            m = JSON.parse(ev.data as string);
          } catch {
            return;
          }
          if (m.type !== "tick" || !m.snapshot) return;
          gotTick.current = true;
          const snap = m.snapshot;
          setState({ phase: snap.status === "finished" ? "finished" : "replaying", snapshot: snap, error: null });
        };
        ws.onerror = () => fail("the replay stream failed");
        ws.onclose = () => {
          if (closed) return;
          if (!gotTick.current) {
            fail("the replay stream closed before its first frame");
            return;
          }
          setState((s) => ({ ...s, phase: s.snapshot?.status === "finished" ? "finished" : "offline" }));
        };
      })
      .catch((e: unknown) => {
        if (abort.signal.aborted) return;
        fail(e instanceof Error ? e.message : String(e));
      });

    return () => {
      closed = true;
      abort.abort();
      if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
    };
  }, [racePrefix, speed, runToken]);

  return state;
}
