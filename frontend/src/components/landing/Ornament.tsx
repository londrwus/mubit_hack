import { Hourglass } from "lucide-react";

/** Pen "Ornament divider": hairlines, four diamonds and a medallion, on the cobalt plates. */
export function OrnamentDivider({ className = "" }: { className?: string }) {
  const diamond = <span className="block size-2 rotate-45 outline outline-1 -outline-offset-[0.5px] outline-white/35" />;
  return (
    <div
      aria-hidden="true"
      className={`relative flex h-[44px] w-full items-center gap-[14px] border-y border-white/15 ${className}`}
    >
      <span className="h-px flex-1 bg-white/15" />
      <span className="grid size-[11.3px] place-items-center">{diamond}</span>
      <span className="grid size-[11.3px] place-items-center">{diamond}</span>
      <span className="grid size-7 place-items-center rounded-full outline outline-1 -outline-offset-[0.5px] outline-white/35">
        <Hourglass size={13} strokeWidth={1.6} className="text-ivory" />
      </span>
      <span className="grid size-[11.3px] place-items-center">{diamond}</span>
      <span className="grid size-[11.3px] place-items-center">{diamond}</span>
      <span className="h-px flex-1 bg-white/15" />
    </div>
  );
}

/** Small-caps mono eyebrow ("I · THE ENGINE"). */
export function Eyebrow({ children, tone = "brass-ink" }: { children: string; tone?: "brass-ink" | "brass" }) {
  return (
    <div
      className={`whitespace-nowrap font-mono text-[12px] tracking-[2px] ${tone === "brass" ? "text-brass" : "text-brass-ink"}`}
    >
      {children}
    </div>
  );
}

/** "FIELD NOTE —— text" line. */
export function FieldNote({ children, dark = false }: { children: string; dark?: boolean }) {
  return (
    <div className="flex items-center gap-[10px]">
      <span className={`whitespace-nowrap font-mono text-[11px] tracking-[2px] ${dark ? "text-brass" : "text-brass-ink"}`}>
        FIELD NOTE
      </span>
      <span className={`h-px w-[18px] ${dark ? "bg-brass" : "bg-brass-ink"}`} />
      <span className={`font-mono text-[12px] tracking-[0.4px] ${dark ? "text-ivory-dim" : "text-cobalt"}`}>{children}</span>
    </div>
  );
}
