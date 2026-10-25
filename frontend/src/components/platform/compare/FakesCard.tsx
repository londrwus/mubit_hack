import { useEffect, useRef, useState } from "react";
import { RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Tween } from "./Tween";

/** "Hallucinated wins blocked": fakes_caught from the snapshot, with the check that blocks them. */
export function FakesCard({ count, strong }: { count: number | null; strong?: boolean }) {
  // Flash the last step each time the count goes up.
  const prev = useRef(count ?? 0);
  const [flash, setFlash] = useState(0);
  useEffect(() => {
    if (count != null && count > prev.current) setFlash((f) => f + 1);
    prev.current = count ?? 0;
  }, [count]);
  const steps = [
    { icon: Sparkles, text: "A run claims a new best" },
    { icon: RefreshCw, text: "Checked again on fresh data" },
    { icon: ShieldCheck, text: "Not a real gain: blocked" },
  ];
  const hit = (count ?? 0) > 0;
  return (
    <div
      className={cn(
        "flex min-w-0 flex-1 flex-col gap-2 rounded-[12px] px-4 py-3 outline outline-1 -outline-offset-[0.5px]",
        strong && hit ? "bg-[#FBF7EE] outline-[#EBDDBF]" : "bg-white outline-app-line",
      )}
      title="fakes_caught: claimed improvements that the check rejected (a known shortcut or noise), so they never became the best model"
    >
      <div className="flex items-center gap-[6px]">
        <ShieldCheck size={13} strokeWidth={1.8} className={strong ? "text-brass-ink" : "text-app-slate"} />
        <span className="text-[12.5px] text-app-slate">Hallucinated wins blocked</span>
        <span className={cn("ml-auto font-mono text-[22px] leading-none font-medium tabular-nums", strong ? "text-ink" : "text-app-text")}>
          {count == null ? "—" : <Tween value={count} format={(n) => String(Math.round(n))} />}
        </span>
      </div>
      <div className="flex flex-col gap-[4px]">
        {steps.map((s, i) => {
          const Icon = s.icon;
          const last = i === steps.length - 1;
          return (
            <span
              key={last ? `last${flash}` : i}
              className={cn(
                "flex items-center gap-2 rounded-[7px] px-2 py-[3px] text-[11.5px] outline outline-1 -outline-offset-[0.5px] transition-colors duration-500",
                last && hit
                  ? cn("font-semibold", strong ? "bg-[#F6EBD3] text-brass-ink outline-[#E8D3A6]" : "bg-app-well text-app-text-2 outline-app-line-2", flash ? "animate-pop-in" : "")
                  : "bg-white text-app-slate outline-app-line-2",
              )}
            >
              <Icon size={12} strokeWidth={1.8} /> {s.text}
            </span>
          );
        })}
      </div>
    </div>
  );
}
