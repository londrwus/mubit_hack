import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// shadcn/ui Button, restyled with the platform frames' tokens (P6OjMG / GKsLR).
const buttonVariants = cva(
  "inline-flex shrink-0 cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap font-sans font-semibold transition-[filter,background-color] outline-none focus-visible:ring-2 focus-visible:ring-cobalt-bright/50 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary:
          "rounded-[9px] bg-gradient-to-b from-cobalt-bright to-cobalt text-[13px] text-white outline outline-1 -outline-offset-1 outline-[#16307F] shadow-[0_1px_2px_#1D3B9440,inset_0_1px_0_#FFFFFF33] hover:brightness-110",
        code: "rounded-[6px] text-[11px] text-[#C9CEDD] outline outline-1 -outline-offset-1 outline-[#363D57] hover:bg-white/5",
      },
      size: {
        default: "h-[38px] px-4",
        lg: "h-10 px-4",
        sm: "px-2.5 py-[5px]",
      },
    },
    defaultVariants: { variant: "primary", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, type = "button", ...props }, ref) => (
    <button ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  ),
);
Button.displayName = "Button";
