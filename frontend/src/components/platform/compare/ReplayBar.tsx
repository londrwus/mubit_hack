import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Pause, Play, RotateCcw, Server, WifiOff } from "lucide-react";
import { OFFLINE_LABEL } from "@/lib/offlineReplay";
import type { ReplayControls, ReplayState } from "@/lib/useReplay";
import { cn } from "@/lib/utils";

export const SPEEDS = [30, 90, 180] as const;

function clock(s: number) {
  const v = Math.max(0, Math.round(s));
  return `${String(Math.floor(v / 60)).padStart(2, "0")}:${String(v % 60).padStart(2, "0")}`;
}

/** Play/pause, restart, speed and a seekable progress bar for a logged-sweep replay. */
export function ReplayBar({ r }: { r: ReplayState & ReplayControls }) {
  const track = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const dur = r.duration || r.snapshot?.header.budget_s || 900;
  const ready = r.snapshot != null;
  const shown = drag ?? r.t;
  const frac = Math.min(1, Math.max(0, shown / dur));
  // Offline the browser holds every tick, so seeking follows the pointer; the server is asked once, on release.
  const live = r.source === "offline";

  const at = (e: PointerEvent<HTMLDivElement>) => {
    const box = track.current!.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)) * dur;
  };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    if (!ready) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const v = at(e);
    setDrag(v);
    if (live) r.seek(v);
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    if (drag == null) return;
    const v = at(e);
    setDrag(v);
    if (live) r.seek(v);
  };
  const up = (e: PointerEvent<HTMLDivElement>) => {
    if (drag == null) return;
    const v = at(e);
    setDrag(null);
    r.seek(v);
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 120 : 30;
    if (e.key === "ArrowRight") r.seek(Math.min(dur, r.t + step));
    else if (e.key === "ArrowLeft") r.seek(Math.max(0, r.t - step));
    else if (e.key === "Home") r.seek(0);
    else if (e.key === "End") r.seek(dur);
    else return;
    e.preventDefault();
  };

  const finished = r.phase === "finished";
  const iconBtn =
    "grid size-8 shrink-0 cursor-pointer place-items-center rounded-[9px] outline outline-1 -outline-offset-[0.5px] transition-colors duration-200 disabled:cursor-default disabled:opacity-50";

  return (
    <div className="animate-enter flex w-full items-center gap-3 rounded-[12px] bg-white px-3 py-2 shadow-[0_1px_2px_#1A2F7A0D] outline outline-1 -outline-offset-[0.5px] outline-app-line" style={{ animationDelay: "260ms" }}>
      <button
        type="button"
        onClick={r.toggle}
        disabled={!ready && r.phase !== "offline"}
        aria-label={r.playing ? "Pause" : finished ? "Play again" : "Play"}
        title={r.playing ? "Pause" : finished ? "Play again from the start" : "Play"}
        className={cn(iconBtn, "bg-cobalt text-white outline-cobalt-deep hover:bg-cobalt-bright")}
      >
        {r.playing ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
      </button>
      <button type="button" onClick={r.restart} aria-label="Restart" title="Restart from 00:00" className={cn(iconBtn, "bg-white text-app-text-2 outline-app-line hover:bg-app-well")}>
        <RotateCcw size={14} />
      </button>
      <div className="flex shrink-0 rounded-[9px] bg-app-well p-[3px] outline outline-1 -outline-offset-[0.5px] outline-app-line-2" role="group" aria-label="Replay speed">
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => r.setSpeed(s)}
            aria-pressed={r.speed === s}
            title={`${s}× speed: 15 min of race in ${Math.round(900 / s)} s`}
            className={cn(
              "cursor-pointer rounded-[7px] px-[9px] py-[3px] font-mono text-[11.5px] transition-colors duration-200",
              r.speed === s ? "bg-white font-semibold text-cobalt shadow-[0_1px_2px_#1A2F7A1A]" : "text-app-slate hover:text-app-text",
            )}
          >
            {s}×
          </button>
        ))}
      </div>
      <span className="w-[104px] shrink-0 text-center font-mono text-[12px] text-app-text-2 tabular-nums">
        {clock(shown)} / {clock(dur)}
      </span>
      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Race time"
        aria-valuemin={0}
        aria-valuemax={Math.round(dur)}
        aria-valuenow={Math.round(shown)}
        aria-valuetext={clock(shown)}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={() => setDrag(null)}
        onKeyDown={key}
        className="group relative flex h-6 min-w-0 flex-1 cursor-pointer touch-none items-center rounded-full focus-visible:outline-2 focus-visible:outline-cobalt-bright"
      >
        <span className="block h-[6px] w-full overflow-hidden rounded-full bg-app-line">
          <span
            className={cn("block h-full rounded-full bg-cobalt-bright", drag == null && "transition-[width] duration-150 ease-linear")}
            style={{ width: `${frac * 100}%` }}
          />
        </span>
        <span
          className={cn(
            "absolute top-1/2 size-[14px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_1px_3px_#1A2F7A40] outline outline-2 outline-cobalt-bright transition-transform duration-150 group-hover:scale-110",
            drag == null && "transition-[left,transform] duration-150 ease-linear",
          )}
          style={{ left: `${frac * 100}%` }}
        />
      </div>
      <span
        className={cn(
          "flex shrink-0 items-center gap-[6px] rounded-full px-[10px] py-[4px] text-[11.5px] font-medium outline outline-1 -outline-offset-1",
          r.source === "offline" ? "bg-app-side text-app-slate outline-app-line" : "bg-app-wash text-cobalt outline-app-wash-line",
        )}
        title={r.source === "offline" ? `The server is not reachable${r.error ? ` (${r.error})` : ""}. Playing the recorded ticks of this logged sweep in the browser.` : "Ticks streamed by the server from the logged race"}
      >
        {r.source === "offline" ? <WifiOff size={12} /> : <Server size={12} />}
        {r.source === "offline" ? OFFLINE_LABEL : r.source === "backend" ? "Replay from the server" : "Loading…"}
      </span>
    </div>
  );
}
