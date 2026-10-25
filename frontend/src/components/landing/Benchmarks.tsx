import { useEffect, useState, type ReactNode } from "react";
import { ArrowRight } from "lucide-react";
import { Reveal } from "@/components/motion/Reveal";
import { useInView } from "@/hooks/useInView";
import { RESULTS_LOG_URL } from "@/lib/links";
import { digits, displayed, metric, side, type Metric } from "@/lib/showcase";
import { Eyebrow } from "./Ornament";
import "./benchmarks.motion.css";

const BAR_MAX = 150;

type Bars = { title: string; direction: string; m: Metric; note?: string; delay?: number };

/** Counts 0 → value once `active` (the panel's own inView, so a number never lags its bar or ring); ends exactly on value. */
function useCount(value: number, active: boolean, duration: number, delay: number) {
  const [n, setN] = useState<number | null>(null);
  useEffect(() => {
    if (!active) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setN(value);
      return;
    }
    let raf = 0;
    let start = 0;
    const tick = (t: number) => {
      if (!start) start = t + delay;
      const p = Math.min(1, Math.max(0, (t - start) / duration));
      setN(p >= 1 ? value : value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, value, duration, delay]);
  return n ?? (active ? value : 0);
}

/** Counts up to a logged display string ("3.30", "1,376", "0.3"), keeping its decimals and grouping; else plain text. */
function Num({ text, delay, active }: { text: string; delay: number; active: boolean }) {
  const value = parseFloat(text.replace(/,/g, ""));
  const dp = text.split(".")[1]?.length ?? 0;
  const format = (n: number) =>
    n.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp, useGrouping: text.includes(",") });
  const ok = Number.isFinite(value) && format(value) === text;
  const n = useCount(ok ? value : 0, active, 1100, delay);
  return <span className="tabular-nums">{ok ? format(n) : text}</span>;
}

/** A panel that fades in once and hands the same inView flag to its bars, numbers and rings. */
function Panel({ delay, className, children }: { delay: number; className: string; children: (inView: boolean) => ReactNode }) {
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.1, rootMargin: "0px" });
  return (
    <div
      ref={ref}
      className={`reveal reveal-fade ${inView ? "is-in" : ""} ${className}`}
      style={{ transitionDelay: `${delay}ms`, transitionDuration: "600ms" }}
    >
      {children(inView)}
    </div>
  );
}

function BarPanel({ title, direction, m, note, delay = 0 }: Bars) {
  const std = side(m, "standard");
  const bm = side(m, "busy_machines");
  const a = displayed(std);
  const b = displayed(bm);
  const max = Math.max(a, b);
  const h = (v: number) => Math.max(2, Math.round((BAR_MAX * v) / max));
  return (
    <Panel delay={delay} className="flex w-[266px] shrink-0 flex-col gap-[6px] border-t border-cobalt/25 pt-4">
      {(inView) => (
      <div className="contents" title={m.hover}>
        <div className="text-[15px] font-medium text-ink">{title}</div>
        <div className="whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-dir">{direction}</div>
        <div className="flex h-[195.5px] items-end gap-[30px] border-b border-cobalt/40 px-10">
          <div className="flex h-full flex-1 flex-col items-center justify-end gap-2">
            <span className="font-mono text-[13px] text-ink-soft">
              <Num text={digits(std)} delay={delay + 150} active={inView} />
            </span>
            <span className="grow-y block w-full bg-bar-grey" style={{ height: h(a), transitionDelay: `${delay + 150}ms` }} />
          </div>
          <div className="flex h-full flex-1 flex-col items-center justify-end gap-2">
            <span className="font-mono text-[13px] font-semibold text-cobalt">
              <Num text={digits(bm)} delay={delay + 300} active={inView} />
            </span>
            <span className="hatch-engrave grow-y block w-full" style={{ height: h(b), transitionDelay: `${delay + 300}ms` }} />
          </div>
        </div>
        <div className="min-h-[18px] pt-[2px] text-center font-mono text-[11px] tracking-[1.2px] text-brass-ink">{note ?? ""}</div>
      </div>
      )}
    </Panel>
  );
}

const R = 60.9; // ring radius (140 px ring, 18.2 px thick)
const C = 2 * Math.PI * R;

function Donut({ pct, label, strong, delay = 0, inView }: { pct: number; label: string; strong?: boolean; delay?: number; inView: boolean }) {
  const n = useCount(pct, inView, 1400, delay);
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
            strokeDasharray={`${C} ${C}`}
            strokeDashoffset={inView ? C - (C * pct) / 100 : C}
            className="transition-[stroke-dashoffset] duration-[1400ms] ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none"
            style={{ transitionDelay: `${delay}ms` }}
          />
        </svg>
        <span className={`relative font-mono text-[17px] ${strong ? "font-semibold text-cobalt" : "text-ink-soft"}`}>
          <span className="tabular-nums">{n.toFixed(1)}</span>%
        </span>
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
  const [figRef, figIn] = useInView<HTMLElement>({ threshold: 0.1 });
  return (
    <section id="benchmarks" className="flex flex-col items-center gap-5 border-t border-cobalt/[0.12] bg-cream px-[120px] py-24">
      <Reveal variant="fade">
        <Eyebrow>V · PROOF · BENCHMARKS</Eyebrow>
      </Reveal>
      <Reveal
        as="h2"
        variant="blur"
        delay={100}
        duration={900}
        className="m-0 text-center font-serif text-[60px] leading-[63px] font-normal tracking-[-1px] text-ink"
      >
        Benchmarks
      </Reveal>
      <Reveal as="p" delay={220} className="m-0 text-[17px] text-ink-soft">
        Cost, time and model score, measured. Same Opus 5.5 model, same machines, 3 runs each.
      </Reveal>
      <Reveal as="p" variant="fade" delay={320} className="m-0 text-center font-mono text-[12px] tracking-[1.2px] text-dir">
        TESTED ON: GAME-PLAYING · PENDULUM-V1 RL (GYMNAX) &nbsp;·&nbsp; TABULAR · LCBENCH (YAHPO GYM) &nbsp;·&nbsp; IMAGE
        RECOGNITION · CIFAR-10 SPEEDRUN (AIRBENCH)
      </Reveal>

      <figure
        ref={figRef}
        className={`bench-fig ${figIn ? "bench-fig-in" : ""} m-0 flex w-[1200px] flex-col gap-7 bg-plate px-8 pt-8 pb-9 outline outline-1 -outline-offset-[0.5px] outline-cobalt/15`}
      >
        <div className="flex items-center justify-center gap-7 text-[14px] font-medium">
          <span className="flex items-center gap-[10px] text-ink-soft">
            <span className="block h-3 w-[22px] bg-bar-grey" /> Opus 5.5 lab
          </span>
          <span className="flex items-center gap-[10px] text-cobalt">
            <span className="hatch-engrave block h-3 w-[22px]" /> Busy Machines
          </span>
        </div>
        <div className="flex gap-6">
          <BarPanel title="AI bill per sweep" direction="US$ · LOWER IS BETTER ↓" m={g("ai_bill_per_race")} delay={250} />
          <BarPanel title="Total cost per sweep" direction="US$ · LOWER IS BETTER ↓" m={g("total_cost_per_race")} delay={370} />
          <BarPanel title="Time to reach the goal" direction="MINUTES · LOWER IS BETTER ↓" m={g("time_to_goal")} delay={490} />
          <BarPanel title="Experiments per dollar" direction="COUNT · HIGHER IS BETTER ↑" m={g("experiments_per_dollar")} delay={610} />
        </div>
        <div className="flex gap-6">
          <BarPanel title="Model quality" direction="PENALTY · LOWER IS BETTER ↓" m={quality} note={quality.comparison?.kind === "tie" ? "SAME" : undefined} delay={100} />
          <BarPanel
            title="AI bill per sweep, tabular task"
            direction={`US$ · LOWER IS BETTER ↓ · ${side(tabular, "busy_machines").r_id}`}
            m={tabular}
            note="QUALITY WITHIN SET MARGIN"
            delay={220}
          />
          <Panel delay={340} className="flex min-w-0 flex-1 flex-col gap-[6px] border-t border-cobalt/25 pt-4">
            {(inView) => (
            <div className="contents" title={waiting.hover}>
              <div className="text-[15px] font-medium text-ink">Machines busy</div>
              <div className="whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-dir">SHARE OF TIME WORKING · HIGHER IS BETTER ↑</div>
              <div className="flex h-[196px] items-center justify-center gap-[72px]">
                <Donut pct={busyStd} label="Opus 5.5 lab" delay={400} inView={inView} />
                <Donut pct={busyBm} label="Busy Machines" strong delay={550} inView={inView} />
              </div>
              <div className="min-h-[18px] pt-[2px] text-center font-mono text-[11px] tracking-[1.2px] text-brass-ink">
                GAME-PLAYING SWEEP, 30 MACHINES
              </div>
            </div>
            )}
          </Panel>
        </div>
      </figure>
      <Reveal as="p" variant="fade" delay={200} className="m-0 w-[1200px] text-[13px] leading-5 text-ink-soft">
        Figure 1. Busy Machines vs a standard Opus 5.5 sweep. Game-playing sweep (Pendulum-v1 RL), 30 machines, 15 min, mean
        of 3 runs (R78, R87); tabular sweep (LCBench, R82). The time panel uses the game task's fixed goal, an easy one. AI cost at list price.
      </Reveal>
      <a href={RESULTS_LOG_URL} target="_blank" rel="noreferrer" className="group flex items-center gap-2 py-1 text-[15px] font-medium text-cobalt hover:underline">
        Full results, including every task <ArrowRight size={16} className="transition-transform duration-300 group-hover:translate-x-0.5" />
      </a>
    </section>
  );
}
