import { useEffect, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Cog } from "lucide-react";
import { DOCS_URL } from "@/lib/links";
import { OrnamentDivider } from "./Ornament";
import "./hero.motion.css";

type NavLink = { label: string; to: string; external?: boolean; fallback?: string };

const navLinks: NavLink[] = [
  { label: "HOW IT WORKS", to: "#how" },
  { label: "COST", to: "#cost" },
  { label: "TIME", to: "#time", fallback: "#engine" },
  { label: "MODEL SCORE", to: "#score" },
  { label: "BENCHMARK", to: "#benchmarks" },
  { label: "DOCS", to: DOCS_URL, external: true },
];

const NAV_H = 72;
const SOLID_AFTER = 40;

/** The anchor a hash link should use: its own id if present on the page, else its fallback, else null. */
function resolveHash(l: NavLink): string | null {
  if (typeof document === "undefined") return l.to;
  for (const h of [l.to, l.fallback]) if (h && document.getElementById(h.slice(1))) return h;
  return null;
}

/** Scroll state for the fixed nav: whether it is past the hero top, and which section anchor is in view. */
function useNavScroll() {
  const [solid, setSolid] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      setSolid(window.scrollY > SOLID_AFTER);
      const line = NAV_H + window.innerHeight * 0.3;
      let current: string | null = null;
      let bestTop = -Infinity;
      for (const l of navLinks) {
        if (!l.to.startsWith("#")) continue;
        const h = resolveHash(l);
        const el = h ? document.getElementById(h.slice(1)) : null;
        if (!el) continue;
        const r = el.getBoundingClientRect();
        // The section whose top most recently crossed the reading line (and whose bottom is still below the nav).
        if (r.top <= line && r.bottom > NAV_H && r.top > bestTop) {
          bestTop = r.top;
          current = l.label;
        }
      }
      setActive(current);
    };
    const on = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => {
      window.removeEventListener("scroll", on);
      window.removeEventListener("resize", on);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  return { solid, active };
}

const linkBase = "relative font-mono text-[12px] tracking-[1.5px] transition-colors duration-300 hover:text-ivory";

function Nav() {
  const { solid, active } = useNavScroll();
  const cls = (l: NavLink) =>
    `${linkBase} ${active === l.label ? "text-ivory after:absolute after:inset-x-0 after:-bottom-[6px] after:h-px after:bg-ivory/80" : "text-ivory-dim"}`;
  return (
    <>
      {/* Keeps the hero plate exactly where it sat when the nav was in flow. */}
      <div aria-hidden="true" style={{ height: NAV_H }} />
      <nav
        data-solid={solid || undefined}
        className={`hero-nav fixed inset-x-0 top-0 z-50 flex h-[72px] min-w-[1280px] items-center justify-between border-b px-10 transition-[background-color,border-color,box-shadow,backdrop-filter] duration-300 ${
          solid
            ? "border-white/20 bg-cobalt/90 shadow-[0_6px_24px_-12px_rgba(0,0,0,0.55)] backdrop-blur-md"
            : "border-white/15 bg-transparent"
        }`}
      >
        <Link to="/" className="flex w-[260px] items-center gap-[10px]">
          <Cog size={22} strokeWidth={1.6} className="text-ivory" />
          <span className="whitespace-nowrap font-serif text-[26px] leading-none text-ivory">Busy Machines</span>
        </Link>
        <div className="flex items-center gap-9">
          {navLinks.map((l) =>
            l.external ? (
              <a key={l.label} href={l.to} target="_blank" rel="noreferrer" className={cls(l)}>
                {l.label}
              </a>
            ) : l.to.startsWith("#") ? (
              <a
                key={l.label}
                href={l.to}
                aria-current={active === l.label ? "location" : undefined}
                className={cls(l)}
                onClick={(e) => {
                  const h = resolveHash(l);
                  if (!h) return;
                  e.preventDefault();
                  document.getElementById(h.slice(1))?.scrollIntoView({ block: "start" });
                  history.replaceState(null, "", h);
                }}
              >
                {l.label}
              </a>
            ) : (
              <Link key={l.label} to={l.to} className={cls(l)}>
                {l.label}
              </Link>
            ),
          )}
        </div>
        <div className="flex w-[260px] justify-end">
          <Link to="/app/compare" className="flex items-center gap-4 bg-ivory py-2 pr-2 pl-4 transition-[transform,filter] duration-300 hover:-translate-y-0.5 hover:brightness-105">
            <span className="whitespace-nowrap text-[14px] font-medium text-cobalt-deep">Watch the race</span>
            <span className="grid size-7 place-items-center bg-cobalt">
              <ArrowRight size={16} className="text-ivory" />
            </span>
          </Link>
        </div>
      </nav>
    </>
  );
}

const FRAME_HALVES = [
  "M700 0 A180 180 0 0 1 880 180 L1220 180 L1220 320 A180 180 0 0 1 1220 680 L1220 820 L880 820 A180 180 0 0 1 700 1000",
  "M700 0 A180 180 0 0 0 520 180 L180 180 L180 320 A180 180 0 0 0 180 680 L180 820 L520 820 A180 180 0 0 0 700 1000",
];

const PLATE_H = 960;
const CHROME_H = 72 + 44; // nav + divider

/** Scales the 1440×960 plate down on short screens so the whole hero, tagline included, fits the first screen. */
function usePlateZoom() {
  const fit = () => (typeof window === "undefined" ? 1 : Math.min(1, Math.max(0.7, (window.innerHeight - CHROME_H) / PLATE_H)));
  const [z, setZ] = useState(fit);
  useEffect(() => {
    const on = () => setZ(fit());
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return z;
}

/** Staggered one-shot entrance delay (ms) for the hero plate. */
const at = (ms: number): CSSProperties => ({ animationDelay: `${ms}ms` });

const corner = "animate-enter absolute whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-ivory-dim";
const sideLabel = "animate-enter absolute top-[472px] whitespace-nowrap font-mono text-[14px] tracking-[1.5px] text-ivory";

/** Nav + hero plate + divider on one paper texture (Pen pOKAY: fYbWi, KekD1, QwFBT, z1YMqM, pE4fB). */
export function Hero() {
  const z = usePlateZoom();
  return (
    <header className="bg-paper-cobalt relative overflow-hidden">
      <Nav />
      {/* 1440-wide plate, centred and scaled without a transform (zoom) so the engraving can screen-blend with the paper */}
      <div className="flex justify-center" style={{ height: PLATE_H * z }}>
      <div className="relative h-[960px] w-[1440px] shrink-0" style={{ zoom: z }}>
        <div style={at(0)} className="hero-ring absolute top-[-70px] left-[170px] size-[1100px] rounded-full outline outline-1 -outline-offset-[0.5px] outline-white/[0.08]" />
        <div style={at(80)} className="hero-ring absolute top-[70px] left-[310px] size-[820px] rounded-full outline outline-1 -outline-offset-[0.5px] outline-white/[0.063]" />
        <svg
          aria-hidden="true"
          viewBox="0 0 1400 1000"
          preserveAspectRatio="none"
          className="absolute top-[50px] left-[100px] h-[860px] w-[1240px] overflow-visible"
        >
          {/* The frame as two mirrored halves from the top centre, drawn together so it grows symmetrically. */}
          {FRAME_HALVES.map((d) => (
            <path
              key={d}
              d={d}
              pathLength={1000}
              fill="none"
              stroke="#FFFFFF59"
              strokeWidth={1.2}
              vectorEffect="non-scaling-stroke"
              className="animate-draw"
              style={{ "--dash": 1000, animationDelay: "150ms", animationDuration: "1300ms" } as CSSProperties}
            />
          ))}
        </svg>

        <div className="hero-engraving hero-engrave-motion absolute top-[128px] left-[350px] size-[740px]" role="img" aria-label="Engraving of a winged rider on Pegasus carrying a lantern" />

        <div className="absolute top-[132px] left-0 flex h-[90px] w-full items-center justify-center">
          <h1 style={at(120)} className="animate-enter m-0 whitespace-nowrap font-serif text-[60px] font-normal tracking-[-1px] text-ivory">Busy Machines</h1>
        </div>
        <div className="absolute top-[820px] left-0 flex h-[90px] w-full items-center justify-center">
          <p style={at(620)} className="animate-enter m-0 flex flex-col items-center gap-[2px] font-serif text-[30px] leading-[33px] text-ivory">
            <span>Your GPUs should</span>
            <span className="italic">never wait</span>
          </p>
        </div>

        <span style={at(520)} className={`${sideLabel} left-[150px]`}>SAME GPUS,</span>
        <span style={at(560)} className={`${sideLabel} left-[1186px]`}>MORE SCIENCE</span>
        <span style={at(300)} className={`${corner} top-[179px] left-[259px]`}>BUILT FOR AI LABS</span>
        <span style={at(340)} className={`${corner} top-[179px] right-[259px]`}>COMPUTE INTELLIGENCE · RUNS ON MUBIT</span>
        <span style={at(380)} className={`${corner} top-[767px] left-[259px]`}>FOR TRAINING SWEEPS</span>
        <span style={at(420)} className={`${corner} top-[767px] right-[259px]`}>NIGHT SHIFT LONDON · 2026</span>
      </div>
      </div>
      <OrnamentDivider />
    </header>
  );
}
