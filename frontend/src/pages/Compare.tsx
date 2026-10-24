import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Bot,
  Check,
  FlaskConical,
  Gamepad2,
  Hourglass,
  Info,
  Play,
  Receipt,
  RotateCcw,
  ShieldCheck,
  ShieldX,
  Table2,
  Timer,
  TrendingUp,
  WifiOff,
  type LucideIcon,
} from "lucide-react";
import { PageHeader } from "@/components/platform/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { API_BASE, type ArmSnapshot, type Slot, type Snapshot } from "@/lib/api";
import { card, displayed, firstRace, metric, showcase, side, tidyRatio } from "@/lib/showcase";
import { useReplay, type ReplayPhase } from "@/lib/useReplay";
import { cn } from "@/lib/utils";

const SPEED = 90;

// ---------------------------------------------------------------- examples (all from showcase.json)

type ExampleId = "game_playing" | "tabular" | "sapling";

type StatValue = { value: string; hover?: string };
type Fallback = {
  label: string;
  left: { ai: StatValue; waiting: StatValue; experiments: StatValue; quality: StatValue };
  right: { ai: StatValue; waiting: StatValue; experiments: StatValue; quality: StatValue };
};

type RaceExample = {
  id: "game_playing" | "tabular";
  kind: "race";
  title: string;
  icon: LucideIcon;
  points: string[];
  source: string;
  racePrefix: string;
  nSlots: number;
  detail: string;
  sourceRight: string;
  bmTag: string;
  aiChip: string;
  qualityChip: string;
  worse: { text: string; items: string[] };
  fallback: Fallback;
};

type VerifierExample = {
  id: "sapling";
  kind: "verifier";
  title: string;
  icon: LucideIcon;
  points: string[];
  source: string;
  worse: { text: string; items: string[] };
};

type Example = RaceExample | VerifierExample;

function buildExamples(): Example[] {
  const gp = card("game_playing");
  const gpRuns = gp.panels[0].race_ids.length;
  const gpWaiting = metric("game_playing", "machines_waiting");
  const gpAi = metric("game_playing", "ai_bill_per_race");
  const gpQuality = metric("game_playing", "model_quality");
  const busyBm = Math.floor(100 - displayed(side(gpWaiting, "busy_machines")));

  const tb = card("tabular");
  const tbAi = metric("tabular", "ai_bill_per_race");
  const tbQuality = metric("tabular", "model_quality");
  const tbWaiting = metric("tabular", "machines_waiting");

  const sp = card("sapling");
  const caught = metric("sapling", "fakes_caught");
  const noAnswer = metric("sapling", "no_answer");
  const pubCaught = side(caught, "published_sapling");
  const oursCaught = side(caught, "ours_v2_looser");
  const pubNoAnswer = side(noAnswer, "published_sapling");

  const tbMean = (m: typeof tbAi, s: string) => ({ value: side(m, s).display, hover: `${m.hover} (mean of 3 runs, ${side(m, s).r_id})` });

  return [
    {
      id: "game_playing",
      kind: "race",
      title: "Game-playing sweep",
      icon: Gamepad2,
      points: [
        gpQuality.comparison?.kind === "tie" ? "Same model" : gpQuality.comparison?.display ?? "",
        `${tidyRatio(gpAi.comparison?.display ?? "")} AI bill`,
        `Machines ${busyBm}% busy`,
      ],
      source: `${gpAi.values.busy_machines?.r_id} · ${gpRuns} runs`,
      racePrefix: firstRace("game_playing"),
      nSlots: 30,
      detail: "game-playing task, run 1 of 3 · 30 machines each · 15 min",
      sourceRight: "Source: results log R78 run 1 (R90)",
      bmTag: "Lower bill · same quality",
      aiChip: `${tidyRatio(gpAi.comparison?.display ?? "")}, mean of ${gpRuns} runs`,
      qualityChip: "Same quality",
      worse: { text: gp.worse_side.caption, items: gp.worse_side.items.map((i) => i.text) },
      // R90: run-1 end-of-race values of R78 (race 4ca4ed27), shown only when the backend is offline.
      fallback: {
        label: "Backend offline: logged end-of-sweep values, run 1 (R90)",
        left: { ai: { value: "$3.30" }, waiting: { value: "69.0%" }, experiments: { value: "497" }, quality: { value: "−155.0" } },
        right: { ai: { value: "$0.70" }, waiting: { value: "1.4%" }, experiments: { value: "1,594" }, quality: { value: "−154.9" } },
      },
    },
    {
      id: "tabular",
      kind: "race",
      title: "Tabular tuning",
      icon: Table2,
      points: ["Nearly the same model", `${tidyRatio(tbAi.comparison?.display ?? "")} AI bill`],
      source: `${tbAi.values.busy_machines?.r_id} · ${tb.panels[0].race_ids.length} runs`,
      racePrefix: firstRace("tabular"),
      nSlots: 16,
      detail: "tabular task, run 1 of 3 · 16 machines each (simulated GPU time) · 15 min",
      sourceRight: `Source: stored race ${firstRace("tabular")} (R82 run 1)`,
      bmTag: "Lower bill · within margin",
      aiChip: `${tidyRatio(tbAi.comparison?.display ?? "")}, mean of ${tb.panels[0].race_ids.length} runs`,
      qualityChip: "Within the set margin",
      worse: { text: tb.worse_side.caption, items: tb.worse_side.items.map((i) => i.text) },
      fallback: {
        label: "Backend offline: 3-run means (R82)",
        left: {
          ai: tbMean(tbAi, "standard"),
          waiting: tbMean(tbWaiting, "standard"),
          experiments: { value: "—" },
          quality: tbMean(tbQuality, "standard"),
        },
        right: {
          ai: tbMean(tbAi, "busy_machines"),
          waiting: tbMean(tbWaiting, "busy_machines"),
          experiments: { value: "—" },
          quality: tbMean(tbQuality, "busy_machines"),
        },
      },
    },
    {
      id: "sapling",
      kind: "verifier",
      title: "Catching hallucinated wins",
      icon: ShieldCheck,
      points: [
        `Ours caught ${oursCaught.display} fake wins`,
        `Mubit's published model: ${pubCaught.value}`,
        ...(pubNoAnswer.value === pubNoAnswer.of ? ["It answered unsure on every claim"] : []),
      ],
      source: `${oursCaught.r_id} · ${pubNoAnswer.of} checked claims`,
      // The verifier card's own worse side leads with Opus 5.5 beating our release.
      worse: { text: sp.worse_side.items[0]?.text ?? sp.worse_side.caption, items: sp.worse_side.items.map((i) => i.text) },
    },
  ];
}

// ---------------------------------------------------------------- small pieces

function ExampleCard({ ex, selected, onSelect }: { ex: Example; selected: boolean; onSelect: () => void }) {
  const Icon = ex.icon;
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "flex min-w-0 flex-1 cursor-pointer flex-col items-start gap-2 rounded-[14px] bg-white px-[18px] py-[14px] text-left",
        selected
          ? "shadow-[0_4px_14px_#1D3B9420] outline outline-2 -outline-offset-1 outline-cobalt-bright"
          : "shadow-[0_1px_2px_#1A2F7A0D] outline outline-1 -outline-offset-[0.5px] outline-app-line hover:outline-app-wash-line",
      )}
    >
      <div className="flex w-full items-center gap-[10px]">
        <span className={cn("grid size-7 place-items-center rounded-[8px]", selected ? "bg-cobalt text-white" : "bg-app-side text-cobalt")}>
          <Icon size={15} strokeWidth={1.8} />
        </span>
        <span className="flex-1 text-[15px] font-semibold text-ink">{ex.title}</span>
        {selected && (
          <span className="rounded-full bg-app-wash px-2 py-[3px] text-[11px] font-semibold text-cobalt">Showing</span>
        )}
      </div>
      <div className="flex flex-col gap-[5px]">
        {ex.points.map((p) => (
          <span key={p} className="flex items-center gap-2 text-[13px] leading-4 text-app-text-2">
            <Check size={13} strokeWidth={2.2} className="text-app-ok" /> {p}
          </span>
        ))}
      </div>
      <span className="mt-auto font-mono text-[10.5px] text-app-muted">{ex.source}</span>
    </button>
  );
}

function Stat({ icon: Icon, label, note, value, hover, chip, strong }: { icon: LucideIcon; label: string; note?: string; value: string; hover?: string; chip?: string | null; strong?: boolean }) {
  return (
    <div
      className="flex min-h-[100px] min-w-0 flex-1 flex-col justify-center gap-2 rounded-[12px] px-[18px] py-3 outline outline-1 -outline-offset-[0.5px] outline-app-line"
      title={hover}
    >
      <div className="flex items-center gap-[6px]">
        <Icon size={13} strokeWidth={1.8} className="text-app-slate" />
        <span className="text-[13.5px] text-app-slate">{label}</span>
        {note && <span className="text-[11px] text-app-muted">{note}</span>}
      </div>
      <div className="flex flex-col items-start gap-[6px]">
        <span
          className={cn(
            "whitespace-nowrap font-mono leading-[1.15] font-medium tracking-[-0.5px] tabular-nums",
            value.length > 9 ? "text-[28px]" : "text-[34px]",
            strong ? "text-ink" : "text-app-text",
          )}
        >
          {value}
        </span>
        {chip && <Badge>{chip}</Badge>}
      </div>
    </div>
  );
}

function Tile({ i, state, strong, title }: { i: number; state: Slot["state"] | "unknown"; strong?: boolean; title?: string }) {
  const busy = state === "busy";
  const off = state === "off" || state === "unknown";
  return (
    <div
      title={title}
      className={cn(
        "flex h-[40px] min-w-0 flex-1 flex-col items-start justify-between rounded-[7px] px-[6px] py-[5px] outline outline-1 -outline-offset-[0.5px] transition-colors duration-150",
        busy ? (strong ? "bg-cobalt outline-[#16307F]" : "bg-cobalt-busy outline-[#16307F]") : off ? "bg-app-well outline-app-line-2" : "bg-white outline-app-line",
      )}
    >
      <span className={cn("font-mono text-[8.5px] leading-none", busy ? "text-[#C9D4F2]" : off ? "text-app-muted/60" : "text-app-muted")}>
        {String(i + 1).padStart(2, "0")}
      </span>
      <span className={cn("block h-[3px] rounded-[2px]", busy ? "w-full bg-ivory/70" : state === "cancelled" ? "w-2 bg-[#E3C07A]" : "w-2 bg-[#E6E3DB]")} />
    </div>
  );
}

function Grid({ label, count, slots, n, strong, finished }: { label: string; count: string; slots: Slot[] | null; n: number; strong?: boolean; finished?: boolean }) {
  const cols = n <= 16 ? 8 : 10;
  const rows: number[][] = [];
  for (let i = 0; i < n; i += cols) rows.push(Array.from({ length: Math.min(cols, n - i) }, (_, k) => i + k));
  const describe = (s: Slot | undefined) =>
    finished
      ? `Each tile is 1/${n} of the sweep's machine time. Filled = time spent training (100% − machines waiting).`
      : s ? `${s.slot_id}: ${s.state}${s.source ? ` (${s.source === "llm" ? "planned by the AI" : s.source === "sampler" ? "built-in tuner" : s.source})` : ""}` : undefined;
  return (
    <div className="flex w-full flex-col gap-2 rounded-[12px] bg-app-well p-3 outline outline-1 -outline-offset-[0.5px] outline-app-line-2">
      <div className="flex items-center justify-between">
        <span className="text-[11.5px] font-medium text-app-slate">{label}</span>
        <span className={cn("font-mono text-[10.5px]", strong ? "text-cobalt" : "text-app-slate")}>{count}</span>
      </div>
      {rows.map((r, ri) => (
        <div key={ri} className="flex w-full gap-[6px]">
          {r.map((i) => (
            <Tile key={i} i={i} state={slots?.[i]?.state ?? "unknown"} strong={strong} title={describe(slots?.[i])} />
          ))}
          {r.length < cols && Array.from({ length: cols - r.length }, (_, k) => <div key={`pad${k}`} className="flex-1" />)}
        </div>
      ))}
    </div>
  );
}

function ColumnShell({ strong, head, children }: { strong?: boolean; head: ReactNode; children: ReactNode }) {
  return (
    <div
      className={cn(
        "flex min-w-0 flex-1 flex-col gap-3 rounded-[16px] bg-white p-4",
        strong
          ? "shadow-[0_4px_16px_#1D3B9418] outline outline-[1.5px] -outline-offset-[0.75px] outline-app-wash-line"
          : "shadow-[0_1px_2px_#1A2F7A0D] outline outline-1 -outline-offset-[0.5px] outline-app-line",
      )}
    >
      {head}
      {children}
    </div>
  );
}

function ColumnHead({ strong, name, sub, tag, hover, icon }: { strong?: boolean; name: string; sub: string; tag: string; hover?: string; icon?: LucideIcon }) {
  const Icon = icon ?? Bot;
  return (
    <div className="flex w-full items-center gap-[10px]" title={hover}>
      {strong ? (
        <span className="grid size-[30px] place-items-center rounded-[8px] bg-gradient-to-b from-cobalt-bright to-cobalt font-serif text-[20px] leading-none text-ivory">
          B
        </span>
      ) : (
        <span className="grid size-[30px] place-items-center rounded-[8px] bg-[#E9E7E0] text-app-text-2">
          <Icon size={16} strokeWidth={1.8} />
        </span>
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-px">
        <span className={cn("truncate text-[14px] font-semibold", strong ? "text-ink" : "text-app-text")}>{name}</span>
        <span className="truncate text-[11.5px] text-app-slate">{sub}</span>
      </div>
      <Badge variant={strong ? "blue" : "grey"} size="tag">
        <span className={cn("size-[6px] rounded-full", strong ? "bg-cobalt" : "bg-app-muted")} /> {tag}
      </Badge>
    </div>
  );
}

// ---------------------------------------------------------------- race example (live replay)

function fmtUsd(v: number) {
  return `$${v.toFixed(2)}`;
}
function fmtPct(v: number) {
  return `${v.toFixed(1)}%`;
}
function bySource(a: ArmSnapshot) {
  const s = a.finished_by_source ?? {};
  const parts = [s.llm ? `${s.llm} planned by the AI` : "", s.sampler ? `${s.sampler} by the built-in tuner` : "", s.baseline ? `${s.baseline} starting model` : ""];
  return parts.filter(Boolean).join(", ");
}

function armStats(a: ArmSnapshot, snap: Snapshot) {
  const h = snap.header;
  return {
    ai: { value: fmtUsd(a.ai_bill_usd), hover: `ai_bill_usd: ${h.ai_cost_note ?? "AI $ at list price (estimate)"}` },
    waiting: {
      value: fmtPct(a.waiting_so_far_pct),
      hover: `waiting_so_far_pct: idle ÷ paid machine time so far · ${h.machines.caption}`,
    },
    experiments: { value: a.experiments_finished.toLocaleString("en-US"), hover: `experiments_finished${bySource(a) ? `: ${bySource(a)}` : ""}` },
    quality: { value: a.model_quality.best_text ?? "—", hover: "model_quality.best: best held-out score of a counted run" },
  };
}

function RacePanel({ ex, phase, snapshot }: { ex: RaceExample; phase: ReplayPhase; snapshot: Snapshot | null }) {
  const left = snapshot?.arms.find((a) => a.side === "left") ?? null;
  const right = snapshot?.arms.find((a) => a.side === "right") ?? null;
  // A finished race has every slot off. After the finish the grid shows each side's whole sweep
  // instead: tiles filled = share of machine time spent training (100% − machines waiting, the
  // snapshot's own end value), so the picture matches the logged number next to it.
  const finished = snapshot?.status === "finished";
  const timeSlots = (a: ArmSnapshot): Slot[] => {
    const filled = Math.round((a.n_slots * (100 - a.waiting_so_far_pct)) / 100);
    return Array.from({ length: a.n_slots }, (_, i) => ({
      slot_id: `${i + 1}`,
      state: i < filled ? "busy" : "waiting",
      since_s: null,
      experiment_id: null,
      source: null,
    }));
  };
  const gridSlots = (a: ArmSnapshot | null) => (a ? (finished ? timeSlots(a) : a.slots) : null);
  const gridLabel = finished ? "Machine time spent training, whole sweep" : "Machines right now";
  const offline = phase === "offline" && !snapshot;
  const n = snapshot?.header.gpus_per_arm ?? ex.nSlots;
  const done = phase === "finished" || offline;
  const higher = snapshot?.header.higher_is_better ?? true;

  const count = (a: ArmSnapshot | null) => {
    if (!snapshot || !a) return offline ? "backend offline" : "loading…";
    if (finished) return `${fmtPct(100 - a.waiting_so_far_pct)} · ${snapshot.finished_text ?? "Finished"}`;
    if (snapshot.status === "not_started") return "starting";
    return `${a.working_now} of ${a.n_slots} training`;
  };
  const stats = (a: ArmSnapshot | null, fb: Fallback["left"]) =>
    a && snapshot ? armStats(a, snapshot) : offline ? fb : { ai: { value: "—" }, waiting: { value: "—" }, experiments: { value: "—" }, quality: { value: "—" } };

  const L = stats(left, ex.fallback.left);
  const Rt = stats(right, ex.fallback.right);
  const waitingLabel = snapshot?.header.machines.waiting_label ?? "Machines waiting";
  const std = showcase.sides.standard;
  const bm = showcase.sides.busy_machines;

  return (
    <div className="flex w-full flex-1 gap-4">
      <ColumnShell head={<ColumnHead name="Standard Opus 5.5 agent" sub="Waits for the AI between experiments" tag="Baseline" hover={std.hover} />}>
        <Grid label={gridLabel} count={count(left)} slots={gridSlots(left)} n={n} finished={finished} />
        <div className="flex w-full flex-1 flex-col gap-[10px]">
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={Receipt} label="AI bill" {...L.ai} />
            <Stat icon={Hourglass} label={waitingLabel} {...L.waiting} />
          </div>
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={FlaskConical} label="Experiments" {...L.experiments} />
            <Stat icon={TrendingUp} label="Model quality" note={higher ? "higher is better" : "lower is better"} {...L.quality} />
          </div>
        </div>
      </ColumnShell>
      <ColumnShell
        strong
        head={<ColumnHead strong name="Busy Machines (same Opus 5.5)" sub="Keeps every machine training while the AI thinks" tag={ex.bmTag} hover={bm.hover} />}
      >
        <Grid label={gridLabel} count={count(right)} slots={gridSlots(right)} n={n} strong finished={finished} />
        <div className="flex w-full flex-1 flex-col gap-[10px]">
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={Receipt} label="AI bill" strong chip={done ? ex.aiChip : null} {...Rt.ai} />
            <Stat icon={Hourglass} label={waitingLabel} strong {...Rt.waiting} />
          </div>
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={FlaskConical} label="Experiments" strong {...Rt.experiments} />
            <Stat
              icon={TrendingUp}
              label="Model quality"
              note={higher ? "higher is better" : "lower is better"}
              strong
              chip={done ? ex.qualityChip : null}
              {...Rt.quality}
            />
          </div>
        </div>
      </ColumnShell>
    </div>
  );
}

// ---------------------------------------------------------------- verifier example (static, R81)

function ClaimsGrid({ label, caught, of, strong }: { label: string; caught: number; of: number; strong?: boolean }) {
  return (
    <div className="flex w-full flex-col gap-2 rounded-[12px] bg-app-well p-3 outline outline-1 -outline-offset-[0.5px] outline-app-line-2">
      <div className="flex items-center justify-between">
        <span className="text-[11.5px] font-medium text-app-slate">{label}</span>
        <span className={cn("font-mono text-[10.5px]", strong ? "text-cobalt" : "text-app-slate")}>
          {caught} of {of} caught
        </span>
      </div>
      <div className="grid grid-cols-[repeat(22,minmax(0,1fr))] gap-1" aria-label={`${caught} of ${of} fake wins caught`}>
        {Array.from({ length: of }, (_, i) => (
          <span
            key={i}
            className={cn(
              "block aspect-square rounded-[3px] outline outline-1 -outline-offset-[0.5px]",
              i < caught ? "bg-cobalt outline-[#16307F]" : "bg-white outline-app-line",
            )}
          />
        ))}
      </div>
    </div>
  );
}

function VerifierPanel() {
  const m = (id: string) => metric("sapling", id);
  const pubId = "published_sapling";
  const oursId = "ours_v2_looser";
  const caught = m("fakes_caught");
  const approved = m("fakes_approved");
  const noAnswer = m("no_answer");
  const time = m("time_per_check");
  const v = (mm: typeof caught, s: string) => ({ value: side(mm, s).display, hover: `${mm.hover} (${side(mm, s).r_id})` });
  const pub = showcase.sides[pubId];
  const ours = showcase.sides[oursId];
  const of = side(caught, oursId).of ?? 109;
  return (
    <div className="flex w-full flex-1 gap-4">
      <ColumnShell head={<ColumnHead name="Mubit's published Banyan Sapling" sub="As released by MubitLabs" tag="Baseline" hover={pub.hover} icon={ShieldX} />}>
        <ClaimsGrid label="Fake wins in our race claims" caught={side(caught, pubId).value} of={of} />
        <div className="flex w-full flex-1 flex-col gap-[10px]">
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={ShieldCheck} label="Fakes caught" {...v(caught, pubId)} />
            <Stat icon={ShieldX} label="Fakes approved" {...v(approved, pubId)} />
          </div>
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={Hourglass} label="Answered “unsure”" {...v(noAnswer, pubId)} />
            <Stat icon={Timer} label="Time per check" {...v(time, pubId)} />
          </div>
        </div>
      </ColumnShell>
      <ColumnShell strong head={<ColumnHead strong name="sapling-research (ours)" sub="Same base model, retrained checking layer" tag="Open model" hover={ours.hover} />}>
        <ClaimsGrid label="Fake wins in our race claims" caught={side(caught, oursId).value} of={of} strong />
        <div className="flex w-full flex-1 flex-col gap-[10px]">
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={ShieldCheck} label="Fakes caught" strong {...v(caught, oursId)} />
            <Stat
              icon={ShieldX}
              label="Fakes approved"
              strong
              chip={side(approved, oursId).value === 0 ? "None wrongly approved" : null}
              {...v(approved, oursId)}
            />
          </div>
          <div className="flex w-full flex-1 gap-[10px]">
            <Stat icon={Hourglass} label="Answered “unsure”" strong {...v(noAnswer, oursId)} />
            <Stat icon={Timer} label="Time per check" strong {...v(time, oursId)} />
          </div>
        </div>
      </ColumnShell>
    </div>
  );
}

// ---------------------------------------------------------------- page

/** "/app/compare": standard Opus 5.5 agent vs Busy Machines on logged sweeps (Pen GKsLR). */
export default function Compare() {
  const examples = useMemo(buildExamples, []);
  const [params, setParams] = useSearchParams();
  const initial = (params.get("example") as ExampleId | null) ?? "game_playing";
  const [selected, setSelected] = useState<ExampleId>(examples.some((e) => e.id === initial) ? initial : "game_playing");
  const [runToken, setRunToken] = useState(0);
  const ex = examples.find((e) => e.id === selected)!;
  const race = ex.kind === "race" ? ex : null;
  const { phase, snapshot, error } = useReplay(race?.racePrefix ?? null, SPEED, runToken);

  useEffect(() => {
    document.title = "Compare · Busy Machines";
  }, []);

  const select = (id: ExampleId) => {
    setSelected(id);
    setParams(id === "game_playing" ? {} : { example: id }, { replace: true });
  };

  const progress = snapshot && snapshot.header.budget_s > 0 ? Math.min(1, snapshot.race_elapsed_s / snapshot.header.budget_s) : 0;
  const busy = phase === "connecting" || phase === "replaying";

  let phaseText = "";
  if (race) {
    if (phase === "connecting") phaseText = "loading the logged sweep…";
    else if (phase === "replaying" && snapshot) phaseText = `${snapshot.clock} · ${SPEED}× speed`;
    else if (phase === "finished") phaseText = "end-of-sweep values";
    else if (phase === "offline") phaseText = snapshot ? "stream stopped" : race.fallback.label;
  }

  return (
    <>
      <PageHeader
        eyebrow="Compare"
        title="See how it compares"
        subtitle="Same machines, same budget, same Opus 5.5. Pick an example from our logged sweeps."
        right={
          <div className="flex items-center gap-3">
            {race && phase === "offline" && (
              <Badge variant="next" size="status" title={`${error ?? ""}\nStart the API on ${API_BASE} (see frontend/README.md).`}>
                <WifiOff size={13} /> Backend offline
              </Badge>
            )}
            <Button
              onClick={() => setRunToken((t) => t + 1)}
              disabled={!race || phase === "connecting"}
              title={race ? `Replays the logged sweep from the start at ${SPEED}× (15 min in 10 s)` : "A logged result; there is nothing to replay"}
            >
              {busy ? <span className="animate-bm-pulse size-2 rounded-full bg-white" /> : <Play size={14} />}
              {phase === "replaying" ? `Replaying · ${SPEED}×` : `Replay · ${SPEED}×`}
            </Button>
          </div>
        }
      />

      <div className="flex w-full gap-[14px]">
        {examples.map((e) => (
          <ExampleCard key={e.id} ex={e} selected={e.id === selected} onSelect={() => select(e.id)} />
        ))}
      </div>

      {race ? <RacePanel ex={race} phase={phase} snapshot={snapshot} /> : <VerifierPanel />}

      <div className="flex w-full flex-col gap-[6px]">
        <div className="flex w-full items-center justify-between gap-6">
          <div className="flex min-w-0 items-center gap-2 text-[12.5px]">
            <RotateCcw size={13} className="shrink-0 text-app-text-2" />
            {race ? (
              <>
                <span className="whitespace-nowrap font-semibold text-app-text-2">Replay of a logged sweep</span>
                <span className="truncate text-app-slate">
                  · {race.detail} · {phaseText}
                </span>
                {phase === "replaying" && (
                  <span className="ml-1 block h-1 w-24 shrink-0 overflow-hidden rounded-full bg-app-line" aria-hidden="true">
                    <span className="block h-full bg-cobalt-bright" style={{ width: `${progress * 100}%` }} />
                  </span>
                )}
              </>
            ) : (
              <>
                <span className="whitespace-nowrap font-semibold text-app-text-2">Logged result</span>
                <span className="truncate text-app-slate">· {card("sapling").panels[0].title} · same claims for both checkers</span>
              </>
            )}
          </div>
          <span className="shrink-0 font-mono text-[11px] text-app-muted">{race ? race.sourceRight : "Source: results log R81"}</span>
        </div>
        <div className="flex items-start gap-2 text-[12px] text-app-slate" title={ex.worse.items.join("\n")}>
          <Info size={13} className="mt-px shrink-0 text-app-muted" />
          <span>
            <span className="font-semibold text-app-text-2">Where it falls short:</span> {ex.worse.text}
          </span>
        </div>
      </div>
    </>
  );
}
