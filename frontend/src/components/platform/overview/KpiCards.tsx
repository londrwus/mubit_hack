import type { ReactNode } from "react";
import { Check, Cpu, FlaskConical, PiggyBank, ShieldCheck, type LucideIcon } from "lucide-react";
import { CountUp } from "@/components/motion/CountUp";
import { Reveal } from "@/components/motion/Reveal";
import { cn } from "@/lib/utils";
import { EXPERIMENTS_R78, fmtInt, headline } from "./data";

type Kpi = {
  title: string;
  icon: LucideIcon;
  iconClass: string;
  value: ReactNode;
  foot: string;
  hover: string;
};

const kpis: Kpi[] = [
  {
    title: "AI bill saved",
    icon: PiggyBank,
    iconClass: "bg-gradient-to-b from-cobalt-bright to-cobalt",
    value: <CountUp value={headline.aiSavedPct} format={(n) => `${Math.round(n)}%`} />,
    foot: `${headline.aiBm.display} vs ${headline.aiStd.display} per sweep`,
    hover: `R78 · mean of 3 game-playing sweeps, 15 min, 30 machines per side. AI bill is a list-price estimate.`,
  },
  {
    title: "Machines busy",
    icon: Cpu,
    iconClass: "bg-cobalt-deep",
    value: <CountUp value={headline.busyBm} format={(n) => `${n.toFixed(1)}%`} />,
    foot: `vs ${headline.busyStd.toFixed(1)}% for the standard agent`,
    hover: "R78 / R90 · 100% minus the logged share of machine time spent waiting, mean of 3 sweeps.",
  },
  {
    title: "Experiments run",
    icon: FlaskConical,
    iconClass: "bg-brass",
    value: <CountUp value={EXPERIMENTS_R78.busy_machines} format={fmtInt} />,
    foot: `vs ${fmtInt(EXPERIMENTS_R78.standard)} in the same 15 min`,
    hover: "R78 · experiments finished per sweep, mean of 3 sweeps, same machines and clock.",
  },
  {
    title: "Hallucinated wins blocked",
    icon: ShieldCheck,
    iconClass: "bg-ink",
    value: (
      <>
        <CountUp value={0} /> shipped
      </>
    ),
    foot: `${headline.fakesBm.display} accepted`,
    hover: "R78 · planted fake improvements accepted as the best model: 0 of 847 trap runs.",
  },
];

export function KpiCards() {
  return (
    <div className="grid grid-cols-4 gap-4">
      {kpis.map((k, i) => (
        <Reveal key={k.title} delay={i * 80}>
          <div
            title={k.hover}
            className="group rounded-[16px] bg-app-well p-[5px] outline outline-1 -outline-offset-1 outline-app-line transition-shadow duration-300 hover:shadow-[0_6px_20px_#1A2F7A14]"
          >
            <div className="flex flex-col gap-4 rounded-[12px] bg-white px-5 pt-4 pb-5 shadow-[0_1px_3px_#1A2F7A12] outline outline-1 -outline-offset-1 outline-app-line">
              <div className="flex items-center gap-[10px]">
                <span className={cn("grid size-7 place-items-center rounded-[8px] text-white transition-transform duration-300 group-hover:scale-110", k.iconClass)}>
                  <k.icon size={15} strokeWidth={2} />
                </span>
                <span className="text-[14px] font-semibold whitespace-nowrap text-ink">{k.title}</span>
              </div>
              <div className="font-mono text-[32px] leading-none font-medium tracking-tight text-ink tabular-nums">{k.value}</div>
            </div>
            <div className="flex items-center gap-2 px-4 pt-[9px] pb-[7px] text-[12px] whitespace-nowrap text-app-slate">
              <Check size={12} strokeWidth={2.4} className="shrink-0 text-cobalt-bright" />
              <span className="flex-1">{k.foot}</span>
              <span className="font-mono text-[10.5px] text-app-muted opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                {k.title === "Machines busy" ? "R78 · R90" : "R78"}
              </span>
            </div>
          </div>
        </Reveal>
      ))}
    </div>
  );
}
