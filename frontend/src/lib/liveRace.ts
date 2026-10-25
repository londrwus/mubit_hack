import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, WS_BASE, type ArmSnapshot, type Snapshot } from "@/lib/api";

// The "Get started" live test: POST /api/mubit/check and POST /api/demo/start?preset=demo-live-10s
// (backend/demo.py), streamed from /ws/races/{id}/dashboard (docs/dashboard_api.md).

export const LIVE_PRESET = "demo-live-10s";
export const LIVE_LABEL = "demo run, not a benchmark result";

// ---- key check

export type KeyCheck = {
  mode: "call" | "presence";
  ok: boolean;
  latency_ms: number | null;
  model: string | null;
  request_id: string | null;
  error: string | null;
};

/** With a key: one tiny call through Mubit (the key is sent once, never stored). Without: the server's own keys. */
export async function checkMubitKey(apiKey?: string): Promise<KeyCheck> {
  const key = apiKey?.trim();
  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/mubit/check`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(key ? { api_key: key } : {}),
    });
  } catch {
    throw new Error(`Can't reach the Busy Machines server at ${API_BASE}. Is the backend running? Then try again.`);
  }
  if (!r.ok) throw new Error(`key check: HTTP ${r.status}`);
  return (await r.json()) as KeyCheck;
}

// ---- live race

/** Measured start-up per arm (race seconds after t0, from event timestamps; backend/dashboard.py). */
export type Startup = {
  first_experiment_s: number | null;
  first_by_source_s: Record<string, number>;
  busy_reached_s: number[];
  all_busy_s: number | null;
};

export type LiveArm = ArmSnapshot & {
  startup?: Startup;
  experiments_started?: number;
  llm_calls?: number;
  planner?: { thinking: boolean; calls?: number };
};

export type LiveSnapshot = Omit<Snapshot, "arms"> & { arms: LiveArm[] };

export type DemoStarted = {
  race_id: string;
  ws: string;
  dry_run: boolean;
  budget_s: number;
  gpus_per_arm: number;
  demo_label: string;
  llm: string;
};

export type LivePhase = "idle" | "starting" | "running" | "finished" | "error";

export type LiveState = {
  phase: LivePhase;
  dryRun: boolean;
  race: DemoStarted | null;
  snapshot: LiveSnapshot | null;
  /** The last snapshot while the race was running (its machine grid is kept on screen after the end). */
  lastLive: LiveSnapshot | null;
  error: string | null;
};

async function startDemo(dryRun: boolean): Promise<DemoStarted> {
  const r = await fetch(`${API_BASE}/api/demo/start?preset=${LIVE_PRESET}&dry_run=${dryRun}`, { method: "POST" });
  if (!r.ok) {
    let detail = "";
    try {
      detail = ((await r.json()) as { detail?: string }).detail ?? "";
    } catch {
      /* not JSON */
    }
    throw new Error(detail || `could not start the live test (HTTP ${r.status})`);
  }
  return (await r.json()) as DemoStarted;
}

async function fetchSnapshot(raceId: string): Promise<LiveSnapshot | null> {
  try {
    const r = await fetch(`${API_BASE}/api/races/${raceId}/snapshot`);
    return r.ok ? ((await r.json()) as LiveSnapshot) : null;
  } catch {
    return null;
  }
}

/** Start the 10-second live test and follow it (snapshots every 0.25 s). */
export function useLiveRace() {
  const [state, setState] = useState<LiveState>({ phase: "idle", dryRun: false, race: null, snapshot: null, lastLive: null, error: null });
  const wsRef = useRef<WebSocket | null>(null);
  const timers = useRef<number[]>([]);

  const cleanup = () => {
    wsRef.current?.close();
    wsRef.current = null;
    timers.current.forEach((t) => clearTimeout(t));
    timers.current = [];
  };
  useEffect(() => cleanup, []);

  const start = useCallback(async (dryRun: boolean) => {
    cleanup();
    setState({ phase: "starting", dryRun, race: null, snapshot: null, lastLive: null, error: null });
    let race: DemoStarted;
    try {
      race = await startDemo(dryRun);
    } catch (e) {
      setState((s) => ({ ...s, phase: "error", error: e instanceof Error ? e.message : String(e) }));
      return;
    }
    setState((s) => ({ ...s, phase: "running", race }));
    const url = `${WS_BASE}${race.ws.replace("tick_s=1", "tick_s=0.25")}`;
    const sock = new WebSocket(url);
    wsRef.current = sock;
    let done = false;
    const settle = () => {
      // The stored summary lands just after the last tick: read the settled snapshot once.
      timers.current.push(
        window.setTimeout(async () => {
          const snap = await fetchSnapshot(race.race_id);
          setState((s) => ({ ...s, phase: "finished", snapshot: snap ?? s.snapshot }));
          sock.close();
        }, 1500),
      );
    };
    sock.onmessage = (ev) => {
      let m: { type?: string; snapshot?: LiveSnapshot };
      try {
        m = JSON.parse(ev.data as string);
      } catch {
        return;
      }
      if (m.type !== "tick" || !m.snapshot) return;
      const snap = m.snapshot;
      const running = snap.status === "running" && snap.arms.some((a) => a.slots.some((x) => x.state !== "off"));
      setState((s) => ({ ...s, snapshot: snap, lastLive: running ? snap : s.lastLive }));
      if (snap.status === "finished" && !done) {
        done = true;
        settle();
      }
    };
    sock.onclose = () => {
      if (!done) {
        done = true;
        settle();
      }
    };
  }, []);

  return { state, start };
}

// ---- the result line (measured numbers only)

export const busyPct = (a: LiveArm): number => Math.max(0, 100 - (a.waiting_so_far_pct ?? 0));

export const secs = (s: number | null | undefined): string =>
  s == null ? "—" : s < 1 ? `${s.toFixed(2)} s` : `${s.toFixed(1)} s`;

export function armsOf(snap: LiveSnapshot | null): { std: LiveArm | null; bm: LiveArm | null } {
  const arms = snap?.arms ?? [];
  return {
    std: arms.find((a) => a.kind === "opus") ?? arms.find((a) => a.side === "left") ?? null,
    bm: arms.find((a) => a.kind === "opus_bm") ?? arms.find((a) => a.side === "right") ?? null,
  };
}

/** "Opus agent: first machine started after 9.6 s · Busy Machines: after 0.03 s", from the final snapshot. */
export function startLine(std: LiveArm, bm: LiveArm, budget: number): string {
  const one = (a: LiveArm) => {
    const t = a.startup?.first_experiment_s;
    return t == null ? `no machine started in ${budget} s` : `first machine started after ${secs(t)}`;
  };
  return `Standard Opus 5.5 agent: ${one(std)} · Busy Machines: ${one(bm)}`;
}

export function resultLine(std: LiveArm, bm: LiveArm): string {
  const parts: string[] = [];
  const e1 = std.experiments_finished, e2 = bm.experiments_finished;
  if (e1 > 0 && e2 > e1) parts.push(`${(e2 / e1).toFixed(1)}× more experiments finished (${e2} vs ${e1})`);
  else parts.push(`experiments finished ${e2} vs ${e1}`);
  parts.push(`machines ${Math.round(busyPct(bm))}% vs ${Math.round(busyPct(std))}% busy`);
  return `Busy Machines: ${parts.join(", ")}`;
}
