// Typed access to data/frontend/showcase.json (copied into src/data by scripts/sync-data.mjs).
// Every number in that file was checked against docs/results_log.md by scripts/build_showcase.py.
// The landing page shows only these numbers; none are retyped by hand.
import raw from "@/data/showcase.json";

export type ShowNumber = {
  value: number;
  unit: string;
  display: string;
  r_id: string;
  n?: number;
  of?: number;
  in_log: boolean;
  log_text?: string | null;
};

export type Metric = {
  id: string;
  label: string;
  hover: string;
  higher_is_better: boolean | null;
  values: Record<string, ShowNumber | null>;
  comparison: (ShowNumber & { kind?: string; formula?: string }) | null;
  verdict?: string;
  note?: string;
};

export type Panel = {
  id: string;
  title: string;
  setup: string;
  r_id: string;
  race_ids: string[];
  sides: string[];
  metrics: Metric[];
};

export type Item = { text: string; r_id: string };

export type Card = {
  id: string;
  order: number;
  kind: string;
  placement: "main" | "secondary";
  title: string;
  headline: string;
  footer: string;
  r_ids: string[];
  panels: Panel[];
  worse_side: { caption: string; items: Item[] };
  notes: Item[];
};

type Showcase = {
  schema_version: number;
  sides: Record<string, { label: string; sublabel: string; hover: string }>;
  cards: Card[];
  checks_against_log: number;
};

export const showcase = raw as unknown as Showcase;

export function card(id: string): Card {
  const c = showcase.cards.find((x) => x.id === id);
  if (!c) throw new Error(`showcase card ${id} missing`);
  return c;
}

export function metric(cardId: string, metricId: string, panelId?: string): Metric {
  const c = card(cardId);
  const panels = panelId ? c.panels.filter((p) => p.id === panelId) : c.panels;
  for (const p of panels) {
    const m = p.metrics.find((x) => x.id === metricId);
    if (m) return m;
  }
  throw new Error(`showcase metric ${cardId}/${metricId} missing`);
}

export function side(m: Metric, sideId: string): ShowNumber {
  const v = m.values[sideId];
  if (!v || !v.in_log) throw new Error(`showcase value ${m.id}/${sideId} missing or not logged`);
  return v;
}

/** The logged display string reduced to its digits ("$3.30" -> "3.30", "−154.5" -> "154.5"). */
export function digits(n: ShowNumber): string {
  return n.display.replace(/[^0-9.,]/g, "");
}

/** The logged display string parsed back to a number, sign dropped. */
export function displayed(n: ShowNumber): number {
  return parseFloat(digits(n).replace(/,/g, ""));
}

/** "4.0× cheaper" -> "4× cheaper" */
export function tidyRatio(s: string): string {
  return s.replace(/(\d+)\.0×/g, "$1×");
}

/** Race id prefix of run 1 of a card's first panel (R78 run 1 = 4ca4ed27, R82 run 1 = c1f4b1d9). */
export function firstRace(cardId: string): string {
  const id = card(cardId).panels[0]?.race_ids[0];
  if (!id) throw new Error(`showcase card ${cardId} has no race`);
  return id;
}
