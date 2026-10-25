import { ArrowRight } from "lucide-react";
import { Reveal } from "@/components/motion/Reveal";
import { CountUp } from "@/components/motion/CountUp";
import { useInView } from "@/hooks/useInView";
import timeline from "@/data/race_r78_run1_timeline.json";
import { FieldNote } from "./Ornament";

type Fmt = (n: number) => string;
type Step = {
  label: string;
  single?: { value: number; format: Fmt };
  std?: { value: number; format: Fmt };
  bm?: { value: number; format: Fmt };
  text: string;
  tag: string;
};

const count: Fmt = (n) => Math.round(n).toLocaleString("en-US");
const mins: Fmt = (n) => `${n.toFixed(1)} min`;

// Logged values: R78 (3-run means), R87 (time to the easy goal). The busy/waiting share is shown once, above (Engine).
const STEPS: Step[] = [
  {
    label: "1 · SAME BUDGET",
    single: { value: 15, format: (n) => `${Math.round(n)} min` },
    text: "30 machines on each side, same Opus 5.5 model, same starting code.",
    tag: "R78 SETUP",
  },
  {
    label: "2 · EXPERIMENTS RUN",
    std: { value: 500, format: count },
    bm: { value: 1587, format: count },
    text: "About three times as many tries in the same 15 minutes.",
    tag: "R78 · MEAN OF 3 RUNS",
  },
  {
    label: "3 · TIME TO THE GOAL",
    std: { value: 1.0, format: mins },
    bm: { value: 0.3, format: mins },
    text: "Reached the goal score sooner.",
    tag: "R87 · EASY GOAL, SEE NOTE",
  },
];

function StepCard({ s, i }: { s: Step; i: number }) {
  const d = 250 + i * 220;
  return (
    <Reveal delay={d} className="flex min-w-0 flex-1 flex-col">
      <div className="font-mono text-[11px] tracking-[2px] text-brass-ink">{s.label}</div>
      {s.single ? (
        <CountUp value={s.single.value} format={s.single.format} delay={d + 200} className="mt-4 font-serif text-[46px] leading-[50px] text-ink" />
      ) : (
        <div className="mt-4 flex flex-col gap-2">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[15px] text-ink-soft">Standard agent</span>
            <CountUp value={s.std!.value} format={s.std!.format} delay={d + 200} className="font-serif text-[28px] leading-[32px] tabular-nums text-ink-soft" />
          </div>
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[15px] font-medium text-cobalt">Busy Machines</span>
            <CountUp value={s.bm!.value} format={s.bm!.format} delay={d + 380} className="font-serif text-[40px] leading-[44px] italic tabular-nums text-cobalt" />
          </div>
        </div>
      )}
      <p className="m-0 mt-3 text-[15px] leading-[22px] text-ink-soft">{s.text}</p>
      <div className="mt-auto pt-8 font-mono text-[10.5px] tracking-[1.6px] text-ink-soft">{s.tag}</div>
    </Reveal>
  );
}

// ---- Chart: best model score so far, from stored race 4ca4ed27 (R78 run 1) ----
const W = 1080;
const H = 340;
const PAD = { l: 70, r: 10, t: 12, b: 34 };
const Y_TOP = -150;
const Y_BOT = -200;
const T_MAX = timeline.budget_s; // 900 s
const GOAL = timeline.goal; // -170

const x = (t: number) => PAD.l + (Math.min(t, T_MAX) / T_MAX) * (W - PAD.l - PAD.r);
const y = (v: number) => PAD.t + ((Y_TOP - Math.max(Y_BOT, Math.min(Y_TOP, v))) / (Y_TOP - Y_BOT)) * (H - PAD.t - PAD.b);

/** Step line: holds each best until the next improvement, runs to the end of the race. Scores below −200 sit on −200. */
function stepPath(series: number[][]) {
  let d = "";
  series.forEach(([t, v], i) => {
    d += i === 0 ? `M${x(t).toFixed(1)},${y(v).toFixed(1)}` : `H${x(t).toFixed(1)}V${y(v).toFixed(1)}`;
  });
  return d + `H${x(T_MAX).toFixed(1)}`;
}

const STD = timeline.arms.standard;
const BM = timeline.arms.busy_machines;
const stdPath = stepPath(STD.best_series);
const bmPath = stepPath(BM.best_series);
const stdGoal = STD.goal_reached_at_s ?? 0;
const bmGoal = BM.goal_reached_at_s ?? 0;
const scoreAt = (series: number[][], t: number) => series.filter(([s]) => s <= t + 1e-6).at(-1)?.[1] ?? Y_BOT;

function Chart() {
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.3 });
  // Reduced motion: show the finished chart at once (no draw-in, no delayed notes).
  const still = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const line = (delay: number) => ({
    strokeDasharray: 1,
    strokeDashoffset: inView ? 0 : 1,
    transition: still ? "none" : `stroke-dashoffset 2200ms cubic-bezier(0.65,0,0.35,1) ${delay}ms`,
  });
  const late = (delay: number) => ({ opacity: inView ? 1 : 0, transition: still ? "none" : `opacity 600ms ease ${delay}ms` });
  const dot = (t: number, v: number, strong: boolean, delay: number) => (
    <circle cx={x(t)} cy={y(v)} r={5} fill={strong ? "#1D3B94" : "#8B98C4"} stroke="#FFFDF8" strokeWidth={2} style={late(delay)} />
  );
  const ticks = [-150, -160, -170, -180, -190, -200];
  return (
    <div ref={ref} className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full overflow-visible" role="img" aria-label="Best model score so far over the 15-minute race, both sides">
        {ticks.map((v) => (
          <g key={v}>
            {v !== GOAL && <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="#1D3B9414" />}
            <text x={PAD.l - 10} y={y(v) + 4} textAnchor="end" className="fill-ink-soft font-mono text-[11px]">
              {v === GOAL ? `goal ${v}` : v}
            </text>
          </g>
        ))}
        <line
          x1={PAD.l}
          x2={W - PAD.r}
          y1={y(GOAL)}
          y2={y(GOAL)}
          stroke="#8F6F3E"
          strokeWidth={1.2}
          pathLength={1}
          style={line(0)}
        />
        {[0, 3, 6, 9, 12, 15].map((m) => (
          <text key={m} x={x(m * 60)} y={H - 8} textAnchor={m === 15 ? "end" : m === 0 ? "start" : "middle"} className="fill-ink-soft font-mono text-[11px]">
            {m} min
          </text>
        ))}
        <path d={stdPath} fill="none" stroke="#8B98C4" strokeWidth={2.2} strokeLinejoin="round" pathLength={1} style={line(200)} />
        <path d={bmPath} fill="none" stroke="#1D3B94" strokeWidth={3} strokeLinejoin="round" pathLength={1} style={line(450)} />
        {dot(bmGoal, scoreAt(BM.best_series, bmGoal), true, 1400)}
        {dot(stdGoal, scoreAt(STD.best_series, stdGoal), false, 1600)}
        {dot(BM.final_best_at_s, BM.final_best, true, 2400)}
        {dot(STD.final_best_at_s, STD.final_best, false, 2600)}
      </svg>
      {/* Notes, positioned in chart units so they track the data. */}
      <p
        className="absolute m-0 w-[340px] text-[15px] leading-[21px] text-brass-ink"
        style={{ left: `${(x(stdGoal) / W) * 100 + 1.5}%`, top: `${(y(-176) / H) * 100}%`, ...late(1800) }}
      >
        Goal reached at 16.9 s by Busy Machines, at 74.6 s by the standard agent.
      </p>
      <p
        className="absolute m-0 w-[230px] text-[15px] leading-[21px] text-ink"
        style={{ left: `${(x(BM.final_best_at_s) / W) * 100 - 11}%`, top: `${(y(-162) / H) * 100}%`, ...late(2600) }}
      >
        8:05 · Busy Machines passes the standard agent's final score.
      </p>
      <p
        className="absolute m-0 w-[250px] text-[15px] leading-[21px] text-ink"
        style={{ left: `${(x(STD.final_best_at_s) / W) * 100 - 22}%`, top: `${(y(-162) / H) * 100}%`, ...late(2800) }}
      >
        13:46 · the standard agent reaches its final score.
      </p>
    </div>
  );
}

/** III · Time, part B (Pen T3KBM): what less waiting turns into. Rendered inside the Engine section, so the
 *  waiting/busy share is not repeated here: same budget → more experiments → goal sooner, then the best-score chart. */
export function TimeToResult() {
  return (
    <div id="time-result" className="mt-[120px] flex w-full flex-col items-center border-t border-cobalt/15 pt-[100px]">
      <Reveal variant="fade" className="font-mono text-[12px] tracking-[2.4px] text-brass-ink">
        WHAT IT TURNS INTO
      </Reveal>
      <Reveal as="h3" variant="blur" delay={120} duration={1000} className="m-0 mt-[18px] text-center font-serif text-[46px] leading-[50px] font-normal tracking-[-1px] text-ink">
        More experiments. The goal, sooner.
      </Reveal>
      <Reveal as="p" delay={260} className="m-0 mt-[20px] w-[720px] max-w-full text-center text-[19px] leading-[29px] text-ink-soft">
        Same 15 minutes, same 30 machines: about three times the experiments, so the sweep finds good models sooner.
      </Reveal>

      <Reveal variant="scale" duration={900} className="mt-12 w-full max-w-[1200px]">
        <div className="flex items-stretch gap-6 bg-plate px-7 pt-9 pb-8 outline outline-1 -outline-offset-[0.5px] outline-cobalt/15">
          {STEPS.map((s, i) => (
            <div key={s.label} className="contents">
              {i > 0 && (
                <Reveal variant="left" delay={150 + i * 220} className="flex shrink-0 items-center self-center">
                  <ArrowRight size={20} strokeWidth={1.4} className="text-brass-ink" />
                </Reveal>
              )}
              <StepCard s={s} i={i} />
            </div>
          ))}
        </div>
      </Reveal>

      <Reveal duration={900} delay={100} className="mt-6 w-full max-w-[1200px]">
        <div className="bg-plate px-11 pt-9 pb-8 outline outline-1 -outline-offset-[0.5px] outline-cobalt/15">
          <div className="flex items-start justify-between gap-6">
            <div>
              <div className="font-serif text-[30px] leading-[34px] text-ink">Best model score so far</div>
              <div className="mt-2 font-mono text-[10.5px] tracking-[1.6px] text-ink-soft">
                GAME-PLAYING SWEEP · HIGHER IS BETTER · R78 RUN 1, FROM THE STORED RACE (4CA4ED27)
              </div>
            </div>
            <div className="flex items-center gap-6 pt-2 font-mono text-[11px] tracking-[1.6px]">
              <span className="flex items-center gap-2 text-ink-soft">
                <span className="block h-[3px] w-5 bg-[#8B98C4]" /> STANDARD AGENT
              </span>
              <span className="flex items-center gap-2 text-cobalt">
                <span className="block h-[3px] w-5 bg-cobalt" /> BUSY MACHINES
              </span>
            </div>
          </div>
          <div className="mt-6">
            <Chart />
          </div>
        </div>
      </Reveal>

      <Reveal variant="fade" delay={200} className="mt-6 w-full max-w-[1100px]">
        <FieldNote>One run of three. Over 3 runs the final scores tie (−154.5 vs −154.3, R78). The goal is easy: the no-AI tuner reached it in 0.4 min (R87).</FieldNote>
      </Reveal>
    </div>
  );
}
