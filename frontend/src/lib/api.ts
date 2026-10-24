export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8765";
export const WS_BASE: string = API_BASE.replace(/^http/, "ws");

// ---- dashboard snapshot (schema busy-machines.dashboard/1, docs/dashboard_api.md §3) ----

export type SlotState = "busy" | "waiting" | "cancelled" | "off";

export type Slot = {
  slot_id: string;
  state: SlotState;
  since_s: number | null;
  experiment_id: string | null;
  source: string | null;
};

export type ArmSnapshot = {
  kind: string;
  label: string;
  sublabel: string;
  side: "left" | "right" | "secondary";
  n_slots: number;
  slots: Slot[];
  working_now: number;
  waiting_so_far_pct: number;
  ai_bill_usd: number;
  experiments_finished: number;
  finished_by_source?: Record<string, number>;
  fakes_caught: number;
  model_quality: { best: number | null; best_text: string | null };
  planner?: { thinking: boolean };
};

export type Snapshot = {
  schema: string;
  race_id: string;
  race_elapsed_s: number;
  clock: string;
  status: "not_started" | "running" | "finished";
  finished_text: string | null;
  header: {
    task: string;
    budget_s: number;
    gpus_per_arm: number;
    budget_text: string;
    goal_text: string;
    higher_is_better: boolean;
    score_decimals: number;
    machines: { label: string; caption: string; count: number; simulated: boolean; waiting_label: string };
    mode_tag: string;
    is_replay: boolean;
    logged_race: { ref: string; run: number; runs: number } | null;
    ai_cost_note: string | null;
    gpu_cost_note: string | null;
  };
  arms: ArmSnapshot[];
};

export type ReplayCreated = {
  replay_id: string;
  ws: string;
  race_duration_s: number;
  replay_duration_s: number;
};

export async function createReplay(racePrefix: string, speed: number, signal: AbortSignal): Promise<ReplayCreated> {
  const url = `${API_BASE}/api/races/${encodeURIComponent(racePrefix)}/replay?speed=${speed}&metric_stride=0`;
  const r = await fetch(url, { method: "POST", signal });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`replay ${racePrefix}: HTTP ${r.status} ${body}`.trim());
  }
  return (await r.json()) as ReplayCreated;
}

export function replayDashboardUrl(replayId: string, tickS = 0.2): string {
  return `${WS_BASE}/ws/replays/${encodeURIComponent(replayId)}/dashboard?tick_s=${tickS}&raw=false&metrics=false`;
}
