import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Panel, Segmented } from "./Panel";
import { run, type ArmKey } from "./data";

const STATE: Record<string, { label: string; cls: string }> = {
  T: { label: "Training", cls: "bg-cobalt" },
  W: { label: "Waiting for the AI", cls: "bg-[#E4E1D9]" },
  O: { label: "Off (sweep ending)", cls: "bg-transparent outline outline-1 -outline-offset-1 outline-app-line" },
  C: { label: "Cancelled", cls: "bg-brass/60" },
};

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** 30 machines x 90 ten-second columns of real slot states from race 4ca4ed27 (R78 run 1). */
export function MachinesHeatmap() {
  const [arm, setArm] = useState<ArmKey>("busy_machines");
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const a = run.arms[arm];
  const other = run.arms[arm === "busy_machines" ? "standard" : "busy_machines"];
  const nCols = run.column_mid_s.length;
  const busy = (x: typeof a) => Math.round((100 - x.waiting_so_far_pct) * 10) / 10;

  const counts = useMemo(() => {
    const t = new Array(nCols).fill(0);
    a.rows.forEach((row) => [...row].forEach((ch, c) => ch === "T" && t[c]++));
    return t;
  }, [a, nCols]);

  const hs = hover ? a.rows[hover.r][hover.c] : null;
  const hT = hover ? run.column_mid_s[hover.c] : 0;

  return (
    <Panel delay={120}>
      <div className="flex items-start justify-between gap-6">
        <div className="flex flex-col gap-1">
          <div className="text-[14px] text-app-text-2">Machines over time</div>
          <div className="flex items-baseline gap-3">
            <span key={arm} className="animate-enter font-mono text-[28px] font-medium whitespace-nowrap text-ink tabular-nums">
              {busy(a).toFixed(1)}% busy
            </span>
            <span className="whitespace-nowrap text-[13px] text-app-slate">
              <span className="mr-1.5 inline-block size-[6px] rounded-full bg-cobalt-bright align-middle" />
              {other.label === "Busy Machines" ? "Busy Machines" : "standard agent"}: {busy(other).toFixed(1)}% busy, same 30 machines ·{" "}
              <span title="Run 1 of 3 (race 4ca4ed27). Per-run values logged in R90; the 3-run means are R78.">R90 · R78 run 1</span>
            </span>
          </div>
        </div>
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-4 text-[13px] text-app-text-2">
            <span className="flex items-center gap-2">
              <span className="size-[10px] rounded-[3px] bg-cobalt" /> Training
            </span>
            <span className="flex items-center gap-2">
              <span className="size-[10px] rounded-[3px] bg-[#E4E1D9]" /> Waiting
            </span>
          </div>
          <Segmented
            value={arm}
            onChange={setArm}
            options={[
              { value: "standard", label: "Standard agent" },
              { value: "busy_machines", label: "Busy Machines" },
            ]}
          />
        </div>
      </div>

      <div className="relative mt-4 flex gap-3" onMouseLeave={() => setHover(null)}>
        <div className="flex w-[78px] shrink-0 flex-col justify-between py-[1px] font-mono text-[10.5px] text-app-muted">
          {[1, 10, 20, 30].map((m) => (
            <span key={m}>Machine {m}</span>
          ))}
        </div>
        <div className="min-w-0 flex-1">
          <div
            className="grid gap-[2px]"
            style={{ gridTemplateColumns: `repeat(${nCols}, minmax(0, 1fr))` }}
            onMouseOver={(e) => {
              const d = (e.target as HTMLElement).dataset;
              if (d.r !== undefined && d.c !== undefined) setHover({ r: +d.r, c: +d.c });
            }}
          >
            {a.rows.map((row, r) =>
              [...row].map((ch, c) => (
                <div
                  key={`${r}-${c}`}
                  data-r={r}
                  data-c={c}
                  className={cn(
                    "h-[9px] rounded-[2px] transition-[background-color,transform,outline-color] duration-500",
                    STATE[ch]?.cls ?? "bg-app-line",
                    hover && hover.r === r && hover.c === c && "scale-[1.6] outline outline-2 outline-cobalt-bright",
                    hover && (hover.r === r || hover.c === c) && !(hover.r === r && hover.c === c) && "brightness-110",
                  )}
                  style={{ transitionDelay: `${c * 5}ms` }}
                />
              )),
            )}
          </div>
          <div className="mt-2 flex justify-between font-mono text-[10.5px] text-app-muted">
            {[0, 3, 6, 9, 12, 15].map((m) => (
              <span key={m}>{m === 0 || m === 15 ? `${m} min` : m}</span>
            ))}
          </div>
        </div>

        {hover && hs && (
          <div
            className="pointer-events-none absolute z-20 -translate-x-1/2 rounded-[10px] bg-ink px-3 py-2 text-[12px] text-white shadow-[0_6px_20px_#1A2F7A40]"
            style={{
              left: `calc(90px + (100% - 90px) * ${(hover.c + 0.5) / nCols})`,
              top: `${(hover.r / a.rows.length) * 100}%`,
              transform: `translate(-50%, ${hover.r > 15 ? "-120%" : "28px"})`,
            }}
          >
            <div className="font-semibold">
              Machine {hover.r + 1} · minute {mmss(hT)}
            </div>
            <div className="mt-0.5 flex items-center gap-2 text-white/80">
              <span className={cn("size-2 rounded-[2px]", hs === "T" ? "bg-cobalt-bright" : "bg-[#E4E1D9]")} />
              {STATE[hs]?.label ?? hs}
            </div>
            <div className="mt-0.5 text-white/60">
              {counts[hover.c]} of {a.rows.length} training at this moment
            </div>
          </div>
        )}
      </div>
      <div className="mt-3 flex items-center gap-2 text-[11.5px] text-app-muted">
        <Badge variant="grey" size="tag">
          Real machine states
        </Badge>
        Each column is the state of every machine sampled every {run.column_s} s from the stored race data (race {run.race_id.slice(0, 8)}).
      </div>
    </Panel>
  );
}
