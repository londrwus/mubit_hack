// Numbers for the Overview page. Headline values come from showcase.json (checked against docs/results_log.md);
// the run-1 grid and curves come from the stored race data of race 4ca4ed27 (R78 run 1, logged as R90).
import { metric, side, displayed } from "@/lib/showcase";
import raw from "@/data/overview_r78_run1.json";

export type ArmKey = "standard" | "busy_machines";
export type RunArm = {
  label: string;
  /** One string per machine; one char per 10 s column: T training, W waiting, O off, C cancelled. */
  rows: string[];
  quality_series: [number, number][];
  quality_baseline: number;
  best_text: string;
  waiting_so_far_pct: number;
  ai_bill_usd: number;
  experiments_finished: number;
  fakes_caught: number;
};
export type Run = {
  source: string;
  r_id: string;
  race_id: string;
  budget_s: number;
  column_s: number;
  column_mid_s: number[];
  goal: number;
  arms: Record<ArmKey, RunArm>;
};

export const run = raw as unknown as Run;

const ai = metric("game_playing", "ai_bill_per_race");
const waiting = metric("game_playing", "machines_waiting");
const fakes = metric("game_playing", "fake_improvements_accepted");

/** R78 headline (3-run means). */
export const headline = {
  aiStd: side(ai, "standard"),
  aiBm: side(ai, "busy_machines"),
  /** "75% less" in R78 = 1 − 0.25×. */
  aiSavedPct: Math.round((1 - (ai.comparison as unknown as { ratio_busy_over_standard: number }).ratio_busy_over_standard) * 100),
  /** Machines busy = 100% − logged waiting (R78 means 68.8% / 1.5%; R90 states 31.2% / 98.5%). */
  busyStd: Math.round((100 - displayed(side(waiting, "standard"))) * 10) / 10,
  busyBm: Math.round((100 - displayed(side(waiting, "busy_machines"))) * 10) / 10,
  fakesBm: side(fakes, "busy_machines"),
};

/** R78 row in docs/results_log.md: "experiments 500 vs 1,587" (mean per race). Not in showcase.json. */
export const EXPERIMENTS_R78 = { standard: 500, busy_machines: 1587, r_id: "R78" } as const;

/** R82 / R90: Busy Machines' AI bill split, mean per race (checker = $0.698 − $0.552). */
export const BILL_R82 = {
  r_id: "R82",
  busy_machines: { planner: 0.552, checker: 0.146, total: 0.698 },
  standard: { planner: 2.632, checker: 0.142, total: 2.774 },
} as const;

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");
