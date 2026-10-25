import type { ArmSnapshot, Slot, SlotState, Snapshot } from "./api";

/**
 * Offline copies of logged races: public/replays/<prefix>.json, compacted from recorded backend
 * dashboard ticks (schema busy-machines.dashboard/1) by scratchpad tools/compact.py. Values are the
 * recorded ones; nothing here is estimated. `snapshotAt` rebuilds the same Snapshot shape the
 * backend sends, so the page renders both sources with the same code.
 */

type ArmTick = [
  slots: string,
  working_now: number,
  waiting_so_far_pct: number,
  ai_bill_usd: number,
  experiments_finished: number,
  fakes_caught: number,
  best: number | null,
  best_text: string | null,
  thinking: 0 | 1,
];
type Tick = [race_elapsed_s: number, clock: string, status: Snapshot["status"], arms: ArmTick[]];

type ArmMeta = {
  kind: string;
  label: string;
  sublabel: string;
  side: "left" | "right";
  n_slots: number;
  slot_ids: string[];
  finished_by_source: Record<string, number> | null;
  series: [number, number][];
  baseline: number | null;
  goal_reached_text: string | null;
};

export type OfflineRace = {
  format: string;
  schema: string;
  race_id: string;
  finished_text: string | null;
  header: Snapshot["header"] & Record<string, unknown>;
  arms: ArmMeta[];
  ticks: Tick[];
};

export const OFFLINE_LABEL = "Replay of a logged sweep (offline copy)";

const cache = new Map<string, Promise<OfflineRace>>();

export function loadOfflineRace(prefix: string): Promise<OfflineRace> {
  let p = cache.get(prefix);
  if (!p) {
    const base = (import.meta.env.BASE_URL as string | undefined) ?? "/";
    p = fetch(`${base.replace(/\/$/, "")}/replays/${encodeURIComponent(prefix)}.json`).then(async (r) => {
      if (!r.ok) throw new Error(`offline copy of ${prefix}: HTTP ${r.status}`);
      return (await r.json()) as OfflineRace;
    });
    p.catch(() => cache.delete(prefix));
    cache.set(prefix, p);
  }
  return p;
}

export function raceDuration(race: OfflineRace): number {
  return race.header.budget_s || race.ticks[race.ticks.length - 1][0];
}

/** Index of the last tick at or before race second `t` (binary search). */
export function tickIndexAt(race: OfflineRace, t: number): number {
  const ts = race.ticks;
  let lo = 0;
  let hi = ts.length - 1;
  if (t >= ts[hi][0]) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ts[mid][0] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

const STATE: Record<string, SlotState> = { b: "busy", w: "waiting", c: "cancelled", o: "off" };

export function snapshotAtIndex(race: OfflineRace, i: number, speed: number): Snapshot {
  const [t, clock, status, armTicks] = race.ticks[i];
  const lr = race.header.logged_race;
  const header: Snapshot["header"] = {
    ...race.header,
    is_replay: true,
    mode_tag: `${OFFLINE_LABEL}${lr ? ` · ${lr.ref} run ${lr.run} of ${lr.runs}` : ""} · ${speed}× speed`,
  };
  const arms: ArmSnapshot[] = race.arms.map((m, k) => {
    const a = armTicks[k];
    const slots: Slot[] = Array.from(a[0], (c, j) => ({
      slot_id: m.slot_ids[j] ?? String(j + 1),
      state: STATE[c] ?? "off",
      since_s: null,
      experiment_id: null,
      source: null,
    }));
    return {
      kind: m.kind,
      label: m.label,
      sublabel: m.sublabel,
      side: m.side,
      n_slots: m.n_slots,
      slots,
      working_now: a[1],
      waiting_so_far_pct: a[2],
      ai_bill_usd: a[3],
      experiments_finished: a[4],
      finished_by_source: status === "finished" ? m.finished_by_source ?? undefined : undefined,
      fakes_caught: a[5],
      model_quality: {
        best: a[6],
        best_text: a[7],
        series: m.series.filter((p) => p[0] <= t),
        baseline: m.baseline,
        goal_reached_text: status === "finished" ? m.goal_reached_text : null,
      },
      planner: { thinking: a[8] === 1 },
    };
  });
  return {
    schema: race.schema,
    race_id: race.race_id,
    race_elapsed_s: t,
    clock,
    status,
    finished_text: status === "finished" ? race.finished_text : null,
    header,
    arms,
  };
}

/** Machines training per recorded tick up to index `i`: the "Machine use over time" history. */
export function historyUpTo(race: OfflineRace, i: number): UsePoint[] {
  const out: UsePoint[] = [];
  const li = race.arms.findIndex((a) => a.side === "left");
  const ri = race.arms.findIndex((a) => a.side === "right");
  for (let k = 0; k <= i; k++) {
    const [t, , , a] = race.ticks[k];
    out.push({ t, left: a[li]?.[1] ?? 0, right: a[ri]?.[1] ?? 0 });
  }
  return out;
}

export type UsePoint = { t: number; left: number; right: number };
