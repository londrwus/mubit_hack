import { useId } from "react";
import { TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

export type QualityDomain = { lo: number; hi: number };

/**
 * Shared y range for both sides: a window around the goal (±15 %), widened to include each side's
 * best so far. Worse early scores fall below the window and enter from the bottom edge.
 */
export function qualityDomain(goal: number | null | undefined, bests: (number | null | undefined)[], higher: boolean): QualityDomain | null {
  const vals = bests.filter((b): b is number => b != null && Number.isFinite(b));
  const g = goal ?? (vals.length ? vals[0] : null);
  if (g == null) return null;
  const half = Math.max(Math.abs(g) * 0.15, 1e-6);
  let lo = g - half;
  let hi = g + half;
  for (const b of vals) {
    const pad = half * 0.12;
    if (higher && b + pad > hi) hi = b + pad;
    if (!higher && b - pad < lo) lo = b - pad;
  }
  return { lo, hi };
}

const W = 600;
const H = 150;

/** "Model quality over time": best score so far (step line) against race minutes, with the goal line. */
export function QualityChart({
  series,
  now,
  duration,
  domain,
  goal,
  goalText,
  higher,
  bestText,
  note,
  strong,
}: {
  series: [number, number][];
  now: number;
  duration: number;
  domain: QualityDomain | null;
  goal: number | null | undefined;
  goalText: string | null | undefined;
  higher: boolean;
  bestText: string | null;
  note: string;
  strong?: boolean;
}) {
  const gid = useId().replace(/:/g, "");
  const dur = duration > 0 ? duration : 900;
  const x = (t: number) => (Math.min(Math.max(t, 0), dur) / dur) * W;
  const y = (v: number) => {
    if (!domain) return H;
    const f = (v - domain.lo) / (domain.hi - domain.lo);
    const ff = higher ? f : 1 - f;
    return Math.min(H, Math.max(0, H - ff * H));
  };
  let line = "";
  if (series.length) {
    line = `M${x(series[0][0]).toFixed(1)},${y(series[0][1]).toFixed(1)}`;
    for (let i = 1; i < series.length; i++) line += `H${x(series[i][0]).toFixed(1)}V${y(series[i][1]).toFixed(1)}`;
    line += `H${x(Math.max(now, series[series.length - 1][0])).toFixed(1)}`;
  }
  const area = line ? `${line}V${H}H${x(series[0][0]).toFixed(1)}Z` : "";
  const goalY = goal != null && domain ? y(goal) : null;
  const stroke = strong ? "var(--color-cobalt)" : "#8E95AE";
  const minutes = Math.round(dur / 60);
  const ticks = [0, Math.round(minutes / 3), Math.round((2 * minutes) / 3), minutes];

  return (
    <div className="flex w-full flex-col gap-2 rounded-[12px] px-4 pt-[10px] pb-2 outline outline-1 -outline-offset-[0.5px] outline-app-line">
      <div className="flex items-center gap-[6px]">
        <TrendingUp size={13} strokeWidth={1.8} className="text-app-slate" />
        <span className="text-[13px] text-app-slate">Model quality over time</span>
        <span className="truncate text-[11px] text-app-muted">{note}</span>
        <span
          key={bestText ?? "none"}
          className={cn("animate-pop-in ml-auto font-mono text-[17px] font-medium tabular-nums", strong ? "text-ink" : "text-app-text")}
          title="model_quality.best: best held-out score of a counted run"
        >
          {bestText ?? "—"}
        </span>
      </div>
      <div className="relative h-[72px] w-full">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full overflow-visible" aria-label={`Best model quality so far: ${bestText ?? "none yet"}`}>
          <defs>
            <linearGradient id={`qa${gid}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={strong ? "#2F5BEA" : "#8E95AE"} stopOpacity={strong ? 0.16 : 0.12} />
              <stop offset="100%" stopColor={strong ? "#2F5BEA" : "#8E95AE"} stopOpacity={0} />
            </linearGradient>
          </defs>
          {[0.25, 0.5, 0.75].map((f) => (
            <line key={f} x1={0} x2={W} y1={H * f} y2={H * f} stroke="var(--color-app-line-2)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          ))}
          <line x1={0} x2={W} y1={H} y2={H} stroke="var(--color-app-line)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          {goalY != null && (
            <line x1={0} x2={W} y1={goalY} y2={goalY} stroke="var(--color-brass)" strokeWidth={1.2} strokeDasharray="5 4" vectorEffect="non-scaling-stroke" />
          )}
          {area && <path d={area} fill={`url(#qa${gid})`} />}
          {line && <path d={line} fill="none" stroke={stroke} strokeWidth={strong ? 2.2 : 1.8} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />}
          <line x1={x(now)} x2={x(now)} y1={0} y2={H} stroke="var(--color-app-wash-line)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        </svg>
        {goalY != null && goalText && (
          <span
            className="pointer-events-none absolute right-1 -translate-y-full pb-[2px] font-mono text-[10px] text-brass-ink"
            style={{ top: `${(goalY / H) * 100}%` }}
          >
            {goalText}
          </span>
        )}
      </div>
      <div className="flex justify-between font-mono text-[10px] text-app-muted">
        {ticks.map((m, i) => (
          <span key={i}>{i === 0 ? "0 min" : i === ticks.length - 1 ? `${m} min` : m}</span>
        ))}
      </div>
    </div>
  );
}
