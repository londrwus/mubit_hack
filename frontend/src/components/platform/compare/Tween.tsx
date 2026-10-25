import { useEffect, useRef, useState } from "react";

const reducedMotion = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Glides from the last shown value to `target` (between two real snapshots, never past the latest); ends exactly on target. */
export function useTween(target: number, ms = 420) {
  const [v, setV] = useState(target);
  const cur = useRef(target);
  useEffect(() => {
    if (cur.current === target || reducedMotion()) {
      cur.current = target;
      setV(target);
      return;
    }
    const from = cur.current;
    let raf = 0;
    let t0 = 0;
    const tick = (t: number) => {
      if (!t0) t0 = t;
      const p = Math.min(1, (t - t0) / ms);
      const x = p >= 1 ? target : from + (target - from) * (1 - Math.pow(1 - p, 3));
      cur.current = x;
      setV(x);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export function Tween({ value, format }: { value: number; format: (n: number) => string }) {
  const v = useTween(value);
  return <>{format(v === value ? value : v)}</>;
}
