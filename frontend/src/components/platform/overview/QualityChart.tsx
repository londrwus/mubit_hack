import { useMemo, useRef, useState } from "react";
import { useInView } from "@/hooks/useInView";
import { cn } from "@/lib/utils";
import { Panel } from "./Panel";
import { run, type ArmKey } from "./data";

const W = 640;
const H = 170;
const PAD = { l: 8, r: 14, t: 10, b: 8 };
const Y_MIN = -200;
const Y_MAX = -150;
const T_MAX = run.budget_s;

const x = (s: number) => PAD.l + (s / T_MAX) * (W - PAD.l - PAD.r);
const y = (v: number) => PAD.t + ((Y_MAX - Math.max(Y_MIN, Math.min(Y_MAX, v))) / (Y_MAX - Y_MIN)) * (H - PAD.t - PAD.b);
const minus = (v: number, d = 1) => (v < 0 ? `−${Math.abs(v).toFixed(d)}` : v.toFixed(d));

/** Best-so-far at time s (step function), or null before the first finished run. */
function bestAt(series: [number, number][], s: number): number | null {
  let b: number | null = null;
  for (const [t, v] of series) if (t <= s) b = v;
  return b;
}

function stepPath(series: [number, number][]) {
  let d = "";
  let prev: number | null = null;
  for (const [t, v] of series) {
    d += prev === null ? `M${x(t)},${y(v)}` : `H${x(t)}V${y(v)}`;
    prev = v;
  }
  return d + `H${x(T_MAX)}`;
}

const ARMS: { key: ArmKey; label: string; stroke: string; width: number }[] = [
  { key: "standard", label: "Standard agent", stroke: "#9096A8", width: 2 },
  { key: "busy_machines", label: "Busy Machines", stroke: "#1D3B94", width: 2.6 },
];

/** Model quality (best held-out game score so far) over the 15-minute race 4ca4ed27, both arms. */
export function QualityChart() {
  const [ref, inView] = useInView<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverS, setHoverS] = useState<number | null>(null);
  const paths = useMemo(() => ARMS.map((a) => ({ ...a, d: stepPath(run.arms[a.key].quality_series) })), []);
  const bm = run.arms.busy_machines;
  const std = run.arms.standard;
  const lastBm = bm.quality_series[bm.quality_series.length - 1][1];

  return (
    <Panel delay={240}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="text-[14px] text-app-text-2">Model quality over time</div>
          <div className="flex items-baseline gap-3">
            <span className="font-serif text-[30px] leading-none whitespace-nowrap text-ink">Same quality</span>
            <span className="whitespace-nowrap text-[13px] text-app-slate">
              <span className="mr-1.5 inline-block size-[6px] rounded-full bg-cobalt-bright align-middle" />
              at a 4× lower AI bill · R78
            </span>
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-start gap-1 pt-1 text-[12px] text-app-text-2">
          {ARMS.map((a) => (
            <span key={a.key} className="flex items-center gap-2 whitespace-nowrap">
              <span className="h-[3px] w-4 rounded-full" style={{ background: a.stroke }} />
              {a.label}
            </span>
          ))}
        </div>
      </div>

      <div ref={ref} className="relative mt-3">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block h-auto w-full overflow-visible"
          onMouseMove={(e) => {
            const r = svgRef.current!.getBoundingClientRect();
            const px = ((e.clientX - r.left) / r.width) * W;
            setHoverS(Math.max(0, Math.min(T_MAX, ((px - PAD.l) / (W - PAD.l - PAD.r)) * T_MAX)));
          }}
          onMouseLeave={() => setHoverS(null)}
        >
          <defs>
            <linearGradient id="ov-q-fill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#2F5BEA" stopOpacity="0.12" />
              <stop offset="1" stopColor="#2F5BEA" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[-160, -180, -200].map((v) => (
            <line key={v} x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="#ECE9E2" />
          ))}
          <line x1={PAD.l} x2={W - PAD.r} y1={y(run.goal)} y2={y(run.goal)} stroke="#C9A46A" strokeDasharray="4 4" />
          {inView && (
            <path
              d={`${paths[1].d}V${y(Y_MIN)}H${x(bm.quality_series[0][0])}Z`}
              fill="url(#ov-q-fill)"
              className="animate-enter"
              style={{ animationDelay: "900ms" }}
            />
          )}
          {inView &&
            paths.map((p, i) => (
              <path
                key={p.key}
                d={p.d}
                fill="none"
                stroke={p.stroke}
                strokeWidth={p.width}
                strokeLinejoin="round"
               
                pathLength={1000}
                className="animate-draw"
                style={{ ["--dash" as string]: 1000, animationDelay: `${i * 250}ms` }}
              />
            ))}
          {inView && <circle cx={x(T_MAX)} cy={y(lastBm)} r={4.5} fill="white" stroke="#1D3B94" strokeWidth={2} />}
          {hoverS !== null && (
            <line x1={x(hoverS)} x2={x(hoverS)} y1={PAD.t} y2={H - PAD.b} stroke="#1A2F7A" strokeOpacity={0.35} />
          )}
        </svg>
        <span className="pointer-events-none absolute right-4 font-mono text-[10.5px] text-brass-ink" style={{ top: `${(y(run.goal) / H) * 100}%`, transform: "translateY(4px)" }}>
          Goal {minus(run.goal, 0)}
        </span>
        {hoverS !== null && (
          <div
            className="pointer-events-none absolute top-1 z-10 rounded-[10px] bg-ink px-3 py-2 text-[12px] text-white shadow-[0_6px_20px_#1A2F7A40]"
            style={{ left: `${(x(hoverS) / W) * 100}%`, transform: `translateX(${hoverS > T_MAX * 0.6 ? "calc(-100% - 10px)" : "10px"})` }}
          >
            <div className="font-semibold">Minute {(hoverS / 60).toFixed(1)}</div>
            {ARMS.map((a) => {
              const v = bestAt(run.arms[a.key].quality_series, hoverS);
              return (
                <div key={a.key} className="mt-0.5 flex items-center gap-2 whitespace-nowrap text-white/80">
                  <span className="h-[3px] w-3 rounded-full" style={{ background: a.key === "busy_machines" ? "#7F9BF5" : a.stroke }} />
                  <span className="flex-1">{a.label}</span>
                  <span className="font-mono text-white">{v === null ? "no result yet" : minus(v)}</span>
                </div>
              );
            })}
            {(bestAt(std.quality_series, hoverS) ?? 0) < Y_MIN || (bestAt(bm.quality_series, hoverS) ?? 0) < Y_MIN ? (
              <div className="mt-1 text-[11px] text-white/50">Values below {minus(Y_MIN, 0)} sit on the bottom edge</div>
            ) : null}
          </div>
        )}
        <div className="mt-1 flex justify-between font-mono text-[10.5px] text-app-muted">
          {[0, 5, 10, 15].map((m) => (
            <span key={m}>{m === 0 || m === 15 ? `${m} min` : m}</span>
          ))}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2">
        {[
          { k: "Final", v: `${bm.best_text} vs ${std.best_text}`, t: "Best score on unseen game levels at the end of run 1 (higher is better). R90." },
          { k: "AI bill", v: `$${bm.ai_bill_usd.toFixed(2)} vs $${std.ai_bill_usd.toFixed(2)}`, t: "AI bill of run 1, list-price estimate. R90." },
          { k: "Source", v: "R78 · run 1", t: `Race ${run.race_id.slice(0, 8)}, stored race data; per-run values logged in R90.` },
        ].map((s) => (
          <div key={s.k} title={s.t} className={cn("rounded-[10px] bg-app-well px-3 py-2 outline outline-1 -outline-offset-1 outline-app-line-2")}>
            <div className="text-[11.5px] text-app-slate">{s.k}</div>
            <div className="font-mono text-[13.5px] text-ink tabular-nums">{s.v}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
