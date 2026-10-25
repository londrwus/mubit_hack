import { useEffect, useState } from "react";
import { useInView } from "@/hooks/useInView";

/**
 * Counts from 0 to `value` once in view, then shows `format(value)` exactly, so the final text is the logged number.
 * Reduced motion: shows the final value at once.
 */
export function CountUp({
  value,
  format = (n) => String(Math.round(n)),
  duration = 1200,
  delay = 0,
  className,
}: {
  value: number;
  format?: (n: number) => string;
  duration?: number;
  delay?: number;
  className?: string;
}) {
  const [ref, inView] = useInView<HTMLSpanElement>();
  const [n, setN] = useState<number | null>(null);

  useEffect(() => {
    if (!inView) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setN(value);
      return;
    }
    let raf = 0;
    let start = 0;
    const tick = (t: number) => {
      if (!start) start = t + delay;
      const p = Math.min(1, Math.max(0, (t - start) / duration));
      setN(p >= 1 ? value : value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, value, duration, delay]);

  return (
    <span ref={ref} className={className}>
      {format(n ?? (inView ? value : 0))}
    </span>
  );
}
