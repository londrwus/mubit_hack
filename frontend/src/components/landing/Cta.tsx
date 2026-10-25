import { Link } from "react-router-dom";
import { ArrowRight, Cog, Play } from "lucide-react";
import { GithubIcon } from "@/components/icons";
import { Reveal } from "@/components/motion/Reveal";
import { useInView } from "@/hooks/useInView";
import { GITHUB_URL } from "@/lib/links";
import { OrnamentDivider } from "./Ornament";

const outlineBtn =
  "flex items-center gap-[10px] px-5 py-[15px] text-[16px] font-medium text-ivory outline outline-1 -outline-offset-[0.5px] outline-white/35 transition duration-300 hover:-translate-y-0.5 hover:bg-white/5";

/** S6 · Call to action + footer (Pen c50WCZ). */
export function Cta() {
  // The footer sits flush with the page bottom, so it can't trigger its own reveal: it follows the CTA block.
  const [ctaRef, ctaIn] = useInView<HTMLDivElement>({ threshold: 0.3 });
  return (
    <section className="bg-paper-cobalt relative flex min-h-[900px] flex-col overflow-hidden">
      <OrnamentDivider />
      <div ref={ctaRef} className="flex flex-1 flex-col items-center justify-center gap-9 py-16">
        <Reveal variant="scale" duration={1200}>
          <div className="seal size-[150px] rounded-full outline outline-1 -outline-offset-[0.5px] outline-ivory" role="img" aria-label="Seal: a winged hourglass" />
        </Reveal>
        <h2 className="m-0 flex flex-col items-center font-serif text-[96px] leading-[96px] font-normal tracking-[-2px] text-ivory">
          <Reveal as="span" variant="blur" delay={150} duration={1000} className="block">
            Same model.
          </Reveal>
          <Reveal as="span" variant="blur" delay={350} duration={1000} className="block italic">
            Busy machines.
          </Reveal>
        </h2>
        <div className="flex items-center gap-4 pt-2">
          <Reveal delay={600}>
            <Link
              to="/app"
              className="group flex items-center gap-[18px] bg-ivory py-[10px] pr-[10px] pl-5 transition duration-300 hover:-translate-y-0.5 hover:brightness-105 hover:shadow-[0_10px_24px_#0A0B1040]"
            >
              <span className="whitespace-nowrap text-[16px] font-semibold text-cobalt-deep">Try Busy Machines</span>
              <span className="grid size-8 place-items-center bg-cobalt">
                <ArrowRight size={16} className="text-ivory transition-transform duration-300 group-hover:translate-x-0.5" />
              </span>
            </Link>
          </Reveal>
          <Reveal delay={720}>
            <Link to="/app/compare" className={outlineBtn}>
              <Play size={16} /> Run the 60-second demo
            </Link>
          </Reveal>
          <Reveal delay={840}>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={outlineBtn}>
              <GithubIcon size={16} /> View on GitHub
            </a>
          </Reveal>
        </div>
      </div>
      <footer
        className={`reveal reveal-fade ${ctaIn ? "is-in" : ""} flex h-[72px] items-center justify-between border-t border-white/15 px-10`}
        style={{ transitionDelay: "900ms" }}
      >
        <div className="flex w-[300px] items-center gap-[10px]">
          <Cog size={18} strokeWidth={1.6} className="text-ivory" />
          <span className="font-serif text-[22px] text-ivory">Busy Machines</span>
        </div>
        <div className="whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-ivory-dim">
          EVERY NUMBER COMES FROM OUR PUBLIC RESULTS LOG · RUNS ON MUBIT
        </div>
        <div className="flex w-[300px] justify-end">
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="whitespace-nowrap font-mono text-[11px] tracking-[1.2px] text-ivory-dim hover:text-ivory">
            GITHUB.COM/LONDRWUS/MUBIT_HACK
          </a>
        </div>
      </footer>
    </section>
  );
}
