import type { CSSProperties, ElementType, ReactNode } from "react";
import { cn } from "@/lib/utils";
import { useInView } from "@/hooks/useInView";

export type RevealVariant = "up" | "fade" | "scale" | "left" | "right" | "blur";

/**
 * Fades/slides its children in once they scroll into view. `delay` (ms) staggers siblings.
 * Styles live in index.css (`.reveal`, `.reveal-*`, `.is-in`); reduced motion shows everything at once.
 */
export function Reveal({
  as: Tag = "div",
  variant = "up",
  delay = 0,
  duration,
  className,
  style,
  children,
}: {
  as?: ElementType;
  variant?: RevealVariant;
  delay?: number;
  duration?: number;
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
}) {
  const [ref, inView] = useInView<HTMLElement>();
  return (
    <Tag
      ref={ref}
      className={cn("reveal", `reveal-${variant}`, inView && "is-in", className)}
      style={{ ...style, transitionDelay: `${delay}ms`, ...(duration ? { transitionDuration: `${duration}ms` } : {}) }}
    >
      {children}
    </Tag>
  );
}
