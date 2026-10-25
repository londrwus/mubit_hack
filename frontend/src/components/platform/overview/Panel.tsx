import type { ReactNode } from "react";
import { Reveal } from "@/components/motion/Reveal";
import { cn } from "@/lib/utils";

/** White dashboard card with the frame's outline and soft shadow. */
export function Panel({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <Reveal delay={delay} className={cn("min-w-0", className)}>
      <section className="flex h-full flex-col rounded-[16px] bg-white px-6 pt-5 pb-5 shadow-[0_1px_3px_#1A2F7A12] outline outline-1 -outline-offset-1 outline-app-line">
        {children}
      </section>
    </Reveal>
  );
}

/** Two-way switch (shadcn-style segmented control) with a sliding thumb. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  return (
    <div role="tablist" className="relative grid rounded-[10px] bg-app-well p-[3px] outline outline-1 -outline-offset-1 outline-app-line" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      <span
        aria-hidden
        className="absolute top-[3px] bottom-[3px] left-[3px] rounded-[8px] bg-white shadow-[0_1px_3px_#1A2F7A1F] outline outline-1 -outline-offset-1 outline-app-line transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
        style={{ width: `calc((100% - 6px) / ${options.length})`, transform: `translateX(${idx * 100}%)` }}
      />
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            "relative z-10 flex cursor-pointer items-center justify-center gap-2 rounded-[8px] px-4 py-[6px] text-[13px] whitespace-nowrap transition-colors duration-300",
            o.value === value ? "font-semibold text-ink" : "text-app-slate hover:text-ink",
          )}
        >
          {o.value === value && <span className="size-[6px] shrink-0 rounded-full bg-cobalt-bright" />}
          {o.label}
        </button>
      ))}
    </div>
  );
}
