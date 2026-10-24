import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// shadcn/ui Badge, with the pill styles of the platform frames.
const badgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full font-sans outline outline-1 -outline-offset-1 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        blue: "bg-app-wash text-cobalt outline-app-wash-line",
        grey: "bg-app-well text-app-slate outline-app-line-2",
        ok: "bg-app-ok-bg text-app-ok outline-app-ok-line",
        next: "bg-app-side text-app-slate outline-app-line",
      },
      size: {
        default: "px-2.5 py-1 text-[11.5px] font-semibold",
        tag: "px-[9px] py-[3px] text-[11px] font-medium",
        status: "px-2.5 py-1.5 text-[12px] font-semibold",
      },
    },
    defaultVariants: { variant: "blue", size: "default" },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, size, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant, size }), className)} {...props} />;
}
