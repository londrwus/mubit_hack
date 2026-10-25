import { Brain, CircleAlert, FlaskConical, Gauge, LoaderCircle, Timer, Trophy, Wallet, Zap } from "lucide-react";
import { CountUp } from "@/components/motion/CountUp";
import { Reveal } from "@/components/motion/Reveal";
import { Badge } from "@/components/ui/badge";
import { armsOf, busyPct, LIVE_LABEL, resultLine, secs, startLine, type LiveArm, type LiveState } from "@/lib/liveRace";
import { cn } from "@/lib/utils";

function MachineGrid({ arm, n }: { arm: LiveArm | null; n: number }) {
  const slots = arm?.slots ?? [];
  const cells = Array.from({ length: n }, (_, i) => slots[i] ?? null);
  return (
    <div className="grid grid-cols-8 gap-[5px]" aria-label="Machines">
      {cells.map((s, i) => {
        const st = s?.state ?? "off";
        const ai = s?.source === "llm";
        return (
          <span
            key={s?.slot_id ?? i}
            title={
              st === "busy"
                ? ai
                  ? "Training a run the AI planned"
                  : s?.source === "sampler"
                    ? "Training a run picked while the AI thinks"
                    : "Training"
                : st === "waiting"
                  ? "Waiting for the AI"
                  : st === "cancelled"
                    ? "Stopped at the time limit"
                    : "Not started"
            }
            className={cn(
              "h-7 rounded-[5px] transition-colors duration-300 ease-out",
              st === "busy" && (ai ? "bg-cobalt-bright" : "bg-[#6F8CF0]"),
              st === "busy" && "shadow-[inset_0_1px_0_#FFFFFF40]",
              st === "waiting" && "bg-[#E9E6DE] outline outline-1 -outline-offset-1 outline-dashed outline-[#C9C4B8]",
              st === "cancelled" && "bg-[#D9D4C8]",
              st === "off" && "bg-app-well outline outline-1 -outline-offset-1 outline-app-line",
            )}
          />
        );
      })}
    </div>
  );
}

function Stat({ icon: Icon, label, value, hint }: { icon: typeof Gauge; label: string; value: string; hint?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-[2px]" title={hint}>
      <span className="flex items-center gap-[5px] text-[11.5px] text-app-slate">
        <Icon size={12} strokeWidth={1.8} /> {label}
      </span>
      <span className="font-mono text-[19px] font-semibold tabular-nums text-app-text">{value}</span>
    </div>
  );
}

function ArmCard({ arm, gridArm, title, sub, n, strong, decimals, frozen }: { arm: LiveArm | null; gridArm: LiveArm | null; title: string; sub: string; n: number; strong: boolean; decimals: number; frozen: boolean }) {
  const first = arm?.startup?.first_experiment_s ?? null;
  const thinking = arm?.planner?.thinking;
  return (
    <div
      className={cn(
        "flex min-w-0 flex-1 flex-col gap-4 rounded-[14px] bg-white p-5",
        strong
          ? "shadow-[0_4px_16px_#1D3B9418] outline outline-[1.5px] -outline-offset-[0.75px] outline-app-wash-line"
          : "outline outline-1 -outline-offset-[0.5px] outline-app-line",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col">
          <span className={cn("text-[17px] font-semibold", strong ? "text-cobalt" : "text-app-text")}>{title}</span>
          <span className="text-[12px] text-app-slate">{sub}</span>
        </div>
        {thinking && (
          <Badge variant="next" size="tag" className="animate-pop-in">
            <Brain size={12} className="animate-bm-pulse" /> AI is planning…
          </Badge>
        )}
      </div>

      <div className="flex flex-col gap-1 rounded-[10px] bg-app-well px-4 py-3">
        <span className="flex items-center gap-[5px] text-[12px] font-medium text-app-slate">
          <Zap size={12} /> Time to first experiment running
        </span>
        <span
          key={first == null ? "none" : "set"}
          className={cn("animate-pop-in font-serif text-[34px] leading-tight tabular-nums", strong ? "text-cobalt" : "text-app-text")}
        >
          {first == null ? <span className="text-app-muted">not yet</span> : secs(first)}
        </span>
        <span className="text-[11.5px] text-app-slate">
          All {n} machines busy: {arm?.startup?.all_busy_s != null ? `after ${secs(arm.startup.all_busy_s)}` : "not yet"}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <MachineGrid arm={gridArm ?? arm} n={n} />
        {frozen && <span className="text-[11px] text-app-muted">Machines as they were just before the time limit</span>}
      </div>

      <div className="grid grid-cols-3 gap-x-4 gap-y-3">
        <Stat icon={Gauge} label="Machines busy now" value={`${arm?.working_now ?? 0} / ${n}`} />
        <Stat
          icon={Gauge}
          label="Busy so far"
          value={arm ? `${Math.round(busyPct(arm))}%` : "—"}
          hint="Share of paid machine time spent training (100% minus time spent waiting)"
        />
        <Stat icon={FlaskConical} label="Experiments finished" value={String(arm?.experiments_finished ?? 0)} />
        <Stat icon={Wallet} label="AI bill" value={arm ? `$${arm.ai_bill_usd.toFixed(3)}` : "—"} hint="LLM cost at list price, through Mubit" />
        <Stat
          icon={Trophy}
          label="Best score"
          value={arm?.model_quality.best != null ? arm.model_quality.best.toFixed(Math.min(4, decimals || 4)) : "—"}
          hint="Model quality of the best counted run (higher is better)"
        />
        <Stat icon={Brain} label="AI calls" value={String(arm?.llm_calls ?? 0)} />
      </div>
    </div>
  );
}

/** The inline live panel of the 10-second test: both sides, measured start-up, and the result line. */
export function LiveRacePanel({ state }: { state: LiveState }) {
  const { phase, snapshot, race, dryRun, error } = state;
  if (phase === "idle") return null;
  if (phase === "error")
    return (
      <div className="animate-enter flex items-center gap-2 rounded-[12px] bg-[#FBEDEA] px-4 py-3 text-[13px] text-[#A23B2A]">
        <CircleAlert size={15} /> {error}
      </div>
    );

  const budget = snapshot?.header.budget_s ?? race?.budget_s ?? 10;
  const n = snapshot?.header.gpus_per_arm ?? race?.gpus_per_arm ?? 16;
  const elapsed = snapshot?.race_elapsed_s ?? 0;
  const left = Math.max(0, budget - elapsed);
  const finished = phase === "finished";
  const { std, bm } = armsOf(snapshot);
  const grid = armsOf(finished ? state.lastLive : snapshot);
  const decimals = snapshot?.header.score_decimals ?? 4;

  return (
    <div className="animate-enter flex w-full flex-col gap-4 rounded-[16px] bg-app-side p-5 outline outline-1 -outline-offset-[0.5px] outline-app-line">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[15px] font-semibold text-app-text">
          {dryRun ? "Rehearsal with a mock AI" : "Live test: real Opus 5.5 through Mubit"}
        </span>
        <Badge variant="grey" size="tag">{dryRun ? "mock AI, $0 · not a result" : `${LIVE_LABEL} · n = 1`}</Badge>
        <span className="flex-1" />
        <span className="flex items-center gap-2 font-mono text-[22px] font-semibold tabular-nums text-cobalt" aria-live="polite">
          {phase === "starting" ? (
            <>
              <LoaderCircle size={18} className="animate-spin" /> starting…
            </>
          ) : finished ? (
            "done"
          ) : (
            <>
              <Timer size={18} /> {left.toFixed(1)} s
            </>
          )}
        </span>
      </div>
      <div className="h-[5px] w-full overflow-hidden rounded-full bg-app-line">
        <div
          className="h-full rounded-full bg-gradient-to-r from-cobalt-bright to-cobalt transition-[width] duration-300 ease-linear"
          style={{ width: `${Math.min(100, (elapsed / budget) * 100)}%` }}
        />
      </div>

      <div className="flex w-full flex-col gap-4 md:flex-row">
        <ArmCard arm={std} gridArm={grid.std} frozen={finished && !!state.lastLive} title="Standard Opus 5.5 agent" sub="Machines wait while the AI plans" n={n} strong={false} decimals={decimals} />
        <ArmCard arm={bm} gridArm={grid.bm} frozen={finished && !!state.lastLive} title="Busy Machines" sub="Same Opus 5.5; machines start before the AI answers" n={n} strong decimals={decimals} />
      </div>

      <div className="flex flex-wrap items-center gap-4 text-[11.5px] text-app-slate">
        <span className="flex items-center gap-[6px]"><span className="size-3 rounded-[3px] bg-cobalt-bright" /> training a run the AI planned</span>
        <span className="flex items-center gap-[6px]"><span className="size-3 rounded-[3px] bg-[#6F8CF0]" /> training a run picked while the AI thinks</span>
        <span className="flex items-center gap-[6px]"><span className="size-3 rounded-[3px] bg-[#E9E6DE] outline outline-1 outline-dashed outline-[#C9C4B8]" /> waiting</span>
        <span>{n} simulated machines per side · same task, budget and time limit</span>
      </div>

      {finished && std && bm && (
        <Reveal variant="up" className="flex flex-col gap-2 rounded-[12px] bg-white px-5 py-4 outline outline-1 -outline-offset-[0.5px] outline-app-wash-line">
          <span className="text-[16px] font-semibold text-cobalt">{startLine(std, bm, budget)}</span>
          <span className="text-[14px] text-app-text">{resultLine(std, bm)}</span>
          <span className="flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-app-slate">
            <span>
              Experiments: <CountUp value={bm.experiments_finished} className="font-mono font-semibold text-app-text" /> vs{" "}
              <CountUp value={std.experiments_finished} className="font-mono font-semibold text-app-text" />
            </span>
            <span>
              AI bill: ${bm.ai_bill_usd.toFixed(3)} vs ${std.ai_bill_usd.toFixed(3)}
            </span>
            <span>
              Best score: {bm.model_quality.best?.toFixed(4) ?? "—"} vs {std.model_quality.best?.toFixed(4) ?? "—"}
            </span>
          </span>
          <span className="text-[11.5px] text-app-muted">
            {dryRun
              ? "Mock AI with paced replies: rehearsal numbers, not a result."
              : `Measured in this run (race ${race?.race_id.slice(0, 8)}): ${LIVE_LABEL}, one run.`}
          </span>
        </Reveal>
      )}
    </div>
  );
}
