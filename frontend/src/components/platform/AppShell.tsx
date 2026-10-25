import type { ReactNode } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { BookOpen, Columns2, Ellipsis, LayoutGrid, PanelLeft, Rocket, type LucideIcon } from "lucide-react";
import { DOCS_URL } from "@/lib/links";
import { cn } from "@/lib/utils";
import "./platform.motion.css";

const item =
  "group flex h-[38px] w-full items-center gap-[10px] rounded-[9px] px-3 text-[14px] transition-[background-color,color,box-shadow,outline-color] duration-300 ease-out";
const active = "bg-white font-semibold text-ink shadow-[0_1px_3px_#1A2F7A12] outline outline-1 -outline-offset-[0.5px] outline-app-line";
const idle = "text-app-text-2 outline outline-1 -outline-offset-[0.5px] outline-transparent hover:bg-white/60 hover:text-ink";

function NavItem({ to, icon: Icon, children, end }: { to: string; icon: LucideIcon; children: ReactNode; end?: boolean }) {
  return (
    <NavLink to={to} end={end} className={({ isActive }) => cn(item, isActive ? active : idle)}>
      {({ isActive }) => (
        <>
          <Icon
            size={16}
            strokeWidth={1.8}
            className={cn(
              "transition-[color,transform] duration-300 ease-out",
              isActive ? "text-ink" : "text-app-slate group-hover:translate-x-px group-hover:text-app-text-2",
            )}
          />
          <span className="flex-1">{children}</span>
        </>
      )}
    </NavLink>
  );
}

/** Sidebar + main area of the platform frames (Pen P6OjMG / GKsLR). */
export function AppShell() {
  const { pathname } = useLocation();
  return (
    <div className="flex min-h-screen min-w-[1280px] bg-app-bg font-sans text-app-text">
      <aside className="sticky top-0 flex h-screen w-[264px] shrink-0 flex-col gap-[14px] border-r border-app-line bg-app-side px-[14px] py-4">
        <Link
          to="/"
          title="Back to the landing page"
          className="flex items-center gap-[10px] rounded-[12px] bg-white py-2 pr-[10px] pl-2 shadow-[0_1px_2px_#1A2F7A0D] transition-shadow duration-300 hover:shadow-[0_3px_10px_#1A2F7A18] outline outline-1 -outline-offset-[0.5px] outline-app-line"
        >
          <span className="grid size-8 place-items-center rounded-[9px] bg-gradient-to-b from-cobalt-bright to-cobalt font-serif text-[22px] leading-none text-ivory">
            B
          </span>
          <span className="flex flex-1 flex-col gap-px">
            <span className="text-[13px] font-semibold text-ink">Busy Machines</span>
            <span className="text-[11px] text-app-slate">Acme Lab · 8 members</span>
          </span>
          <PanelLeft size={16} strokeWidth={1.8} className="text-app-slate" />
        </Link>
        <nav className="flex flex-col gap-1 pt-2">
          <NavItem to="/app/overview" icon={LayoutGrid}>
            Overview
          </NavItem>
          <NavItem to="/app" end icon={Rocket}>
            Get started
          </NavItem>
          <NavItem to="/app/compare" icon={Columns2}>
            Compare
          </NavItem>
          <a href={DOCS_URL} target="_blank" rel="noreferrer" className={cn(item, idle)}>
            <BookOpen size={16} strokeWidth={1.8} className="text-app-slate transition-[color,transform] duration-300 ease-out group-hover:translate-x-px group-hover:text-app-text-2" />
            <span className="flex-1">Docs</span>
          </a>
        </nav>
        <div className="flex-1" />
        <div className="flex items-center gap-[10px] px-2 py-[6px]">
          <span className="grid size-8 place-items-center rounded-full bg-[#DCD3BF] text-[11px] font-semibold text-ink outline outline-2 -outline-offset-1 outline-white">
            SR
          </span>
          <span className="flex flex-1 flex-col gap-px">
            <span className="text-[13px] font-semibold text-app-text">Sam Reyes</span>
            <span className="text-[11px] text-app-slate">Plan &amp; billing</span>
          </span>
          <Ellipsis size={16} className="text-app-slate" />
        </div>
      </aside>
      <main className="flex min-w-0 flex-1 flex-col px-10 pt-6 pb-5">
        {/* Keyed by route so each page plays its entrance when you switch pages. */}
        <div key={pathname} className="animate-page-in flex min-w-0 flex-1 flex-col gap-4">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

/** Page header: eyebrow, serif title, subtitle and a right-hand slot. */
export function PageHeader({ eyebrow, title, subtitle, right }: { eyebrow: string; title: string; subtitle: string; right?: ReactNode }) {
  return (
    <div className="flex w-full items-end justify-between gap-6">
      <div className="flex flex-col gap-[6px]">
        <div className="animate-enter text-[12px] font-medium text-app-slate">{eyebrow}</div>
        <h1 className="animate-enter m-0 whitespace-nowrap font-serif text-[44px] leading-[46px] font-normal text-ink" style={{ animationDelay: "70ms" }}>
          {title}
        </h1>
        <div className="animate-enter text-[15px] text-app-slate" style={{ animationDelay: "150ms" }}>
          {subtitle}
        </div>
      </div>
      {right && (
        <div className="animate-enter" style={{ animationDelay: "220ms" }}>
          {right}
        </div>
      )}
    </div>
  );
}
