import { Link } from "react-router-dom";
import { ArrowRight, Cog } from "lucide-react";
import { DOCS_URL } from "@/lib/links";
import { OrnamentDivider } from "./Ornament";

const navLinks: { label: string; to: string; external?: boolean }[] = [
  { label: "THE RACE", to: "/app/compare" },
  { label: "BENCHMARK", to: "#benchmarks" },
  { label: "HOW IT WORKS", to: "#engine" },
  { label: "DOCS", to: DOCS_URL, external: true },
];

function Nav() {
  return (
    <nav className="relative z-10 flex h-[72px] items-center justify-between border-b border-white/15 px-10">
      <Link to="/" className="flex w-[260px] items-center gap-[10px]">
        <Cog size={22} strokeWidth={1.6} className="text-ivory" />
        <span className="whitespace-nowrap font-serif text-[26px] leading-none text-ivory">Busy Machines</span>
      </Link>
      <div className="flex items-center gap-10">
        {navLinks.map((l) =>
          l.external ? (
            <a key={l.label} href={l.to} target="_blank" rel="noreferrer" className="font-mono text-[12px] tracking-[1.5px] text-ivory-dim hover:text-ivory">
              {l.label}
            </a>
          ) : l.to.startsWith("#") ? (
            <a key={l.label} href={l.to} className="font-mono text-[12px] tracking-[1.5px] text-ivory-dim hover:text-ivory">
              {l.label}
            </a>
          ) : (
            <Link key={l.label} to={l.to} className="font-mono text-[12px] tracking-[1.5px] text-ivory-dim hover:text-ivory">
              {l.label}
            </Link>
          ),
        )}
      </div>
      <div className="flex w-[260px] justify-end">
        <Link to="/app/compare" className="flex items-center gap-4 bg-ivory py-2 pr-2 pl-4 hover:brightness-105">
          <span className="whitespace-nowrap text-[14px] font-medium text-cobalt-deep">Watch the race</span>
          <span className="grid size-7 place-items-center bg-cobalt">
            <ArrowRight size={16} className="text-ivory" />
          </span>
        </Link>
      </div>
    </nav>
  );
}

const corner = "absolute whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-ivory-dim";
const sideLabel = "absolute top-[472px] whitespace-nowrap font-mono text-[14px] tracking-[1.5px] text-ivory";

/** Nav + hero plate + divider on one paper texture (Pen pOKAY: fYbWi, KekD1, QwFBT, z1YMqM, pE4fB). */
export function Hero() {
  return (
    <header className="bg-paper-cobalt relative overflow-hidden">
      <Nav />
      {/* 1440-wide plate, centred without a transform so the engraving can screen-blend with the paper */}
      <div className="relative ml-[calc((100%-1440px)/2)] h-[960px] w-[1440px]">
        <div className="absolute top-[-70px] left-[170px] size-[1100px] rounded-full outline outline-1 -outline-offset-[0.5px] outline-white/[0.08]" />
        <div className="absolute top-[70px] left-[310px] size-[820px] rounded-full outline outline-1 -outline-offset-[0.5px] outline-white/[0.063]" />
        <svg
          aria-hidden="true"
          viewBox="0 0 1400 1000"
          preserveAspectRatio="none"
          className="absolute top-[50px] left-[100px] h-[860px] w-[1240px] overflow-visible"
        >
          <path
            d="M520 180 A180 180 0 0 1 880 180 L1220 180 L1220 320 A180 180 0 0 1 1220 680 L1220 820 L880 820 A180 180 0 0 1 520 820 L180 820 L180 680 A180 180 0 0 1 180 320 L180 180 Z"
            fill="none"
            stroke="#FFFFFF59"
            strokeWidth={1.2}
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        <div className="hero-engraving absolute top-[128px] left-[350px] size-[740px]" role="img" aria-label="Engraving of a winged rider on Pegasus carrying a lantern" />

        <div className="absolute top-[132px] left-0 flex h-[90px] w-full items-center justify-center">
          <h1 className="m-0 whitespace-nowrap font-serif text-[60px] font-normal tracking-[-1px] text-ivory">Busy Machines</h1>
        </div>
        <div className="absolute top-[820px] left-0 flex h-[90px] w-full items-center justify-center">
          <p className="m-0 flex flex-col items-center gap-[2px] font-serif text-[30px] leading-[33px] text-ivory">
            <span>Your GPUs should</span>
            <span className="italic">never wait</span>
          </p>
        </div>

        <span className={`${sideLabel} left-[150px]`}>SAME GPUS,</span>
        <span className={`${sideLabel} left-[1186px]`}>MORE SCIENCE</span>
        <span className={`${corner} top-[179px] left-[259px]`}>BUILT FOR AI LABS</span>
        <span className={`${corner} top-[179px] right-[259px]`}>COMPUTE INTELLIGENCE · RUNS ON MUBIT</span>
        <span className={`${corner} top-[767px] left-[259px]`}>FOR TRAINING SWEEPS</span>
        <span className={`${corner} top-[767px] right-[259px]`}>NIGHT SHIFT LONDON · 2026</span>
      </div>
      <OrnamentDivider />
    </header>
  );
}
