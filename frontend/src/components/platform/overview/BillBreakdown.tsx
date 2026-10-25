import { useState } from "react";
import { Sigma } from "lucide-react";
import { CountUp } from "@/components/motion/CountUp";
import { useInView } from "@/hooks/useInView";
import { cn } from "@/lib/utils";
import { Panel } from "./Panel";
import { BILL_R82 } from "./data";

type Part = "planner" | "checker";
const PARTS: { key: Part; label: string; cls: string; hover: string }[] = [
  { key: "planner", label: "Planning the next experiments", cls: "bg-cobalt", hover: "AI calls that propose the next batch of experiments" },
  { key: "checker", label: "Checking claimed wins", cls: "bg-brass", hover: "AI calls that check whether a new best result is real" },
];

const usd = (n: number) => `$${n.toFixed(3)}`;

/** R82: where Busy Machines' AI bill goes, vs the standard agent on the same scale. */
export function BillBreakdown() {
  const [hot, setHot] = useState<Part | null>(null);
  const [ref, inView] = useInView<HTMLDivElement>();
  const bm = BILL_R82.busy_machines;
  const std = BILL_R82.standard;
  const max = std.total;
  const pct = (p: Part) => Math.round((bm[p] / bm.total) * 100);

  const bar = (vals: { planner: number; checker: number; total: number }, label: string, delay: number) => (
    <div className="flex items-center gap-3">
      <span className="w-[112px] shrink-0 text-[12px] text-app-slate">{label}</span>
      <div className="relative h-[26px] flex-1 overflow-hidden rounded-[6px] bg-app-well">
        <div
          className={cn("grow-x flex h-full", inView && "is-in")}
          style={{ width: `${(vals.total / max) * 100}%`, transitionDelay: `${delay}ms` }}
        >
          {PARTS.map((p) => (
            <div
              key={p.key}
              onMouseEnter={() => setHot(p.key)}
              onMouseLeave={() => setHot(null)}
              title={`${p.label}: ${usd(vals[p.key])} per sweep (R82)`}
              className={cn(
                "h-full transition-opacity duration-300 [background-image:repeating-linear-gradient(90deg,transparent_0_7px,rgba(255,255,255,0.55)_7px_9px)]",
                p.cls,
                hot && hot !== p.key && "opacity-30",
              )}
              style={{ width: `${(vals[p.key] / vals.total) * 100}%` }}
            />
          ))}
        </div>
      </div>
      <span className="w-[58px] shrink-0 text-right font-mono text-[12px] text-ink tabular-nums">{usd(vals.total)}</span>
    </div>
  );

  return (
    <Panel delay={180}>
      <div className="text-[14px] text-app-text-2">Where the AI bill goes</div>
      <div className="mt-1 flex items-baseline gap-3">
        <CountUp value={bm.total} format={usd} className="font-mono text-[28px] font-medium text-ink tabular-nums" />
        <span className="text-[13px] text-app-slate">
          <span className="mr-1.5 inline-block size-[6px] rounded-full bg-cobalt-bright align-middle" />
          per sweep · standard agent {usd(std.total)}
        </span>
      </div>

      <div ref={ref} className="mt-4 flex flex-col gap-2">
        {bar(bm, "Busy Machines", 0)}
        {bar(std, "Standard agent", 150)}
      </div>

      <div className="mt-3 flex flex-col gap-1.5">
        {PARTS.map((p) => (
          <div
            key={p.key}
            title={p.hover}
            onMouseEnter={() => setHot(p.key)}
            onMouseLeave={() => setHot(null)}
            className={cn(
              "flex items-center gap-3 rounded-[9px] px-3 py-[7px] text-[13.5px] text-app-text transition-colors duration-300",
              hot === p.key ? "bg-app-wash" : "bg-app-well",
            )}
          >
            <span className={cn("size-[10px] rounded-[3px]", p.cls)} />
            <span className="flex-1">{p.label}</span>
            <span className="font-mono text-[11.5px] text-app-slate">{pct(p.key)}%</span>
            <span className="font-mono text-[11.5px] text-app-muted">std {usd(std[p.key])}</span>
            <span className="w-[56px] text-right font-mono text-[13px] text-ink tabular-nums">{usd(bm[p.key])}</span>
          </div>
        ))}
        <div className="flex items-center gap-3 rounded-[9px] px-3 py-[7px] text-[13.5px] font-semibold text-ink outline outline-1 -outline-offset-1 outline-app-line">
          <Sigma size={13} className="text-app-slate" />
          <span className="flex-1">Total per sweep</span>
          <span className="font-mono text-[11.5px] font-normal text-app-muted">std {usd(std.total)}</span>
          <span className="w-[56px] text-right font-mono text-[13px] tabular-nums">{usd(bm.total)}</span>
        </div>
      </div>
      <div className="mt-auto pt-3 font-mono text-[10.5px] text-app-muted" title="Mean per sweep over 3 tabular sweeps. AI bill is a list-price estimate.">
        Tabular tuning sweep · mean of 3 · R82
      </div>
    </Panel>
  );
}
