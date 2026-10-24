import { ArrowRight } from "lucide-react";
import { RESULTS_LOG_URL } from "@/lib/links";
import { digits, displayed, metric, side, type Metric } from "@/lib/showcase";
import { Eyebrow } from "./Ornament";

const BAR_MAX = 150;

type Bars = { title: string; direction: string; m: Metric; note?: string };

function BarPanel({ title, direction, m, note }: Bars) {
  const std = side(m, "standard");
  const bm = side(m, "busy_machines");
  const a = displayed(std);
  const b = displayed(bm);
  const max = Math.max(a, b);
  const h = (v: number) => Math.max(2, Math.round((BAR_MAX * v) / max));
  return (
    <div className="flex w-[266px] shrink-0 flex-col gap-[6px] border-t border-cobalt/25 pt-4" title={m.hover}>
      <div className="text-[15px] font-medium text-ink">{title}</div>
      <div className="whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-dir">{direction}</div>
      <div className="flex h-[195.5px] items-end gap-[30px] border-b border-cobalt/40 px-10">
        <div className="flex h-full flex-1 flex-col items-center justify-end gap-2">
          <span className="font-mono text-[13px] text-ink-soft">{digits(std)}</span>
          <span className="block w-full bg-bar-grey" style={{ height: h(a) }} />
        </div>
        <div className="flex h-full flex-1 flex-col items-center justify-end gap-2">
          <span className="font-mono text-[13px] font-semibold text-cobalt">{digits(bm)}</span>
          <span className="hatch-engrave block w-full" style={{ height: h(b) }} />
        </div>
      </div>
      <div className="min-h-[18px] pt-[2px] text-center font-mono text-[11px] tracking-[1.2px] text-brass-ink">{note ?? ""}</div>
    </div>
  );
}

const R = 60.9; // ring radius (140 px ring, 18.2 px thick)
const C = 2 * Math.PI * R;

function Donut({ pct, label, strong }: { pct: number; label: string; strong?: boolean }) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative grid size-[140px] place-items-center">
        <svg viewBox="0 0 140 140" className="absolute inset-0 -rotate-90" aria-hidden="true">
          <circle cx="70" cy="70" r={R} fill="none" stroke="#1D3B9414" strokeWidth={18.2} />
          <circle
            cx="70"
            cy="70"
            r={R}
            fill="none"
            stroke={strong ? "#1D3B94" : "#AEB5CA"}
            strokeWidth={18.2}
            strokeDasharray={`${(C * pct) / 100} ${C}`}
          />
        </svg>
        <span className={`relative font-mono text-[17px] ${strong ? "font-semibold text-cobalt" : "text-ink-soft"}`}>{pct.toFixed(1)}%</span>
      </div>
      <span className={`text-[13px] font-medium ${strong ? "text-cobalt" : "text-ink-soft"}`}>{label}</span>
    </div>
  );
}

/** S4 · Benchmarks (Pen oT6Ad): small multiples of R78 (3-run means), R87 and R82. */
export function Benchmarks() {
  const g = (id: string) => metric("game_playing", id);
  const waiting = g("machines_waiting");
  const busyStd = 100 - displayed(side(waiting, "standard"));
  const busyBm = 100 - displayed(side(waiting, "busy_machines"));
  const tabular = metric("tabular", "ai_bill_per_race");
  const quality = g("model_quality");
  return (
    <section id="benchmarks" className="flex flex-col items-center gap-5 border-t border-cobalt/[0.12] bg-cream px-[120px] py-24">
      <Eyebrow>III · BENCHMARKS</Eyebrow>
      <h2 className="m-0 text-center font-serif text-[60px] leading-[63px] font-normal tracking-[-1px] text-ink">Benchmarks</h2>
      <p className="m-0 text-[17px] text-ink-soft">Same Opus 5.5 model, same machines, 3 runs each.</p>

      <figure className="m-0 flex w-[1200px] flex-col gap-7 bg-plate px-8 pt-8 pb-9 outline outline-1 -outline-offset-[0.5px] outline-cobalt/15">
        <div className="flex items-center justify-center gap-7 text-[14px] font-medium">
          <span className="flex items-center gap-[10px] text-ink-soft">
            <span className="block h-3 w-[22px] bg-bar-grey" /> Opus 5.5 lab
          </span>
          <span className="flex items-center gap-[10px] text-cobalt">
            <span className="hatch-engrave block h-3 w-[22px]" /> Busy Machines
          </span>
        </div>
        <div className="flex gap-6">
          <BarPanel title="AI bill per sweep" direction="US$ · LOWER IS BETTER ↓" m={g("ai_bill_per_race")} />
          <BarPanel title="Total cost per sweep" direction="US$ · LOWER IS BETTER ↓" m={g("total_cost_per_race")} />
          <BarPanel title="Time to reach the goal" direction="MINUTES · LOWER IS BETTER ↓" m={g("time_to_goal")} />
          <BarPanel title="Experiments per dollar" direction="COUNT · HIGHER IS BETTER ↑" m={g("experiments_per_dollar")} />
        </div>
        <div className="flex gap-6">
          <BarPanel title="Model quality" direction="PENALTY · LOWER IS BETTER ↓" m={quality} note={quality.comparison?.kind === "tie" ? "SAME" : undefined} />
          <BarPanel
            title="AI bill per sweep, tabular task"
            direction={`US$ · LOWER IS BETTER ↓ · ${side(tabular, "busy_machines").r_id}`}
            m={tabular}
            note="QUALITY WITHIN SET MARGIN"
          />
          <div className="flex min-w-0 flex-1 flex-col gap-[6px] border-t border-cobalt/25 pt-4" title={waiting.hover}>
            <div className="text-[15px] font-medium text-ink">Machines busy</div>
            <div className="whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-dir">SHARE OF TIME WORKING · HIGHER IS BETTER ↑</div>
            <div className="flex h-[196px] items-center justify-center gap-[72px]">
              <Donut pct={busyStd} label="Opus 5.5 lab" />
              <Donut pct={busyBm} label="Busy Machines" strong />
            </div>
            <div className="min-h-[18px] pt-[2px] text-center font-mono text-[11px] tracking-[1.2px] text-brass-ink">
              GAME-PLAYING SWEEP, 30 MACHINES
            </div>
          </div>
        </div>
      </figure>
      <p className="m-0 w-[1200px] text-[13px] leading-5 text-ink-soft">
        Figure 1. Busy Machines vs a standard Opus 5.5 sweep. Game-playing sweep, 30 machines, 15 min, mean of 3 runs (R78,
        R87); tabular sweep (R82). The time panel uses the game task's fixed goal, an easy one. AI cost at list price.
      </p>
      <a href={RESULTS_LOG_URL} target="_blank" rel="noreferrer" className="flex items-center gap-2 py-1 text-[15px] font-medium text-cobalt hover:underline">
        Full results, including every task <ArrowRight size={16} />
      </a>
    </section>
  );
}
