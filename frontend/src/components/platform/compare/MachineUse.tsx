import { Activity } from "lucide-react";
import type { UsePoint } from "@/lib/offlineReplay";
import { cn } from "@/lib/utils";

const BUCKETS = 60;

/**
 * "Machine use over time": one bar per 1/60 of the race, height = machines training in the
 * recorded ticks of that stretch (mean) ÷ machines. Stretches not reached yet stay empty.
 */
export function MachineUse({
  history,
  side,
  nSlots,
  duration,
  busySoFar,
  strong,
}: {
  history: UsePoint[];
  side: "left" | "right";
  nSlots: number;
  duration: number;
  busySoFar: string | null;
  strong?: boolean;
}) {
  const dur = duration > 0 ? duration : 900;
  const sum = new Array<number>(BUCKETS).fill(0);
  const cnt = new Array<number>(BUCKETS).fill(0);
  for (const p of history) {
    const b = Math.min(BUCKETS - 1, Math.floor((p.t / dur) * BUCKETS));
    sum[b] += p[side];
    cnt[b] += 1;
  }
  const bucketS = Math.round(dur / BUCKETS);
  return (
    <div
      className="flex min-w-0 flex-1 flex-col gap-2 rounded-[12px] bg-code px-4 py-3 text-ivory-dim"
      title={`From the logged sweep. Top right: share of machine time spent training so far (100% − machines waiting). Bars: machines training (working_now) in the recorded ticks, averaged per ${bucketS} s of race time, ÷ ${nSlots} machines`}
    >
      <div className="flex items-center gap-[6px]">
        <Activity size={13} strokeWidth={1.8} className={strong ? "text-[#8FA8F5]" : "text-[#8E95AE]"} />
        <span className="truncate text-[12.5px] text-ivory">Machine use over time</span>
        {busySoFar && <span className="ml-auto shrink-0 font-mono text-[10.5px] text-ivory">{busySoFar}</span>}
      </div>
      <div className="flex h-[46px] items-end gap-[2px]" aria-hidden="true">
        {sum.map((s, i) => {
          const h = cnt[i] ? s / cnt[i] / Math.max(1, nSlots) : null;
          return (
            <span key={i} className="flex h-full min-w-0 flex-1 items-end">
              <span
                className={cn(
                  "block w-full rounded-t-[2px] transition-[height,background-color] duration-300 ease-out",
                  h == null ? "bg-transparent" : strong ? "bg-cobalt-bright" : "bg-[#5D6585]",
                )}
                style={{ height: h == null ? "0%" : `${Math.max(3, h * 100)}%` }}
              />
            </span>
          );
        })}
      </div>
      <div className="flex justify-between gap-2 whitespace-nowrap font-mono text-[9.5px] text-[#7D84A0]">
        <span>0 min</span>
        <span className="truncate">{bucketS} s per bar</span>
        <span>{Math.round(dur / 60)} min</span>
      </div>
    </div>
  );
}
