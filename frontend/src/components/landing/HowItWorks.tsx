import type { CSSProperties, ReactNode } from "react";
import { Reveal } from "@/components/motion/Reveal";
import { useInView } from "@/hooks/useInView";
import { Eyebrow } from "./Ornament";
import "./how.motion.css";

const RUN = 96;

/** Thin cobalt connector: draws in, then a dot travels along it. */
function Connector({ delay }: { delay: number }) {
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.4 });
  const vars = { "--d": `${delay}ms`, "--run": `${RUN - 10}px` } as CSSProperties;
  const dotVars = { "--d": `${delay + 900}ms`, "--run": `${RUN - 10}px` } as CSSProperties;
  return (
    <div ref={ref} aria-hidden="true" className={`flex shrink-0 items-center self-center ${inView ? "is-in" : ""}`} style={{ width: RUN }}>
      <svg width={RUN} height={14} viewBox={`0 0 ${RUN} 14`} className="overflow-visible">
        <line x1={0} y1={7} x2={RUN - 8} y2={7} stroke="var(--color-cobalt)" strokeWidth={1} className="how-line" style={vars} />
        <path d={`M ${RUN - 8} 3 L ${RUN} 7 L ${RUN - 8} 11`} fill="none" stroke="var(--color-cobalt)" strokeWidth={1} className="how-line" style={vars} />
        <circle cx={5} cy={7} r={2.6} fill="var(--color-cobalt)" className="how-dot" style={dotVars} />
      </svg>
    </div>
  );
}

function Node({ tag, title, caption, delay, children }: { tag: string; title: string; caption?: string; delay: number; children?: ReactNode }) {
  return (
    <Reveal
      variant="up"
      delay={delay}
      className="flex h-full flex-col gap-3 bg-plate px-7 pt-6 pb-6 outline outline-1 -outline-offset-[0.5px] outline-cobalt/20"
    >
      <span className="font-mono text-[10px] tracking-[1.8px] text-brass-ink">{tag}</span>
      <span className="font-serif text-[26px] leading-[30px] text-ink">{title}</span>
      {caption && <p className="m-0 text-[15px] leading-[22px] text-ink-soft">{caption}</p>}
      {children}
    </Reveal>
  );
}

type Part = { label: string; caption: string; mubit?: boolean };
const ENGINE: Part[] = [
  { label: "AI PLANNER", caption: "Proposes the next experiments.", mubit: true },
  { label: "SCHEDULER", caption: "Keeps every machine busy while the AI thinks." },
  { label: "CHECKER", caption: "Re-tests claimed wins on fresh data." },
];

/** I · How it works (compute intelligence): what the product is, before any number. No numbers on this screen. */
export function HowItWorks() {
  return (
    <section id="how" className="flex flex-col items-center bg-cream px-[120px] pt-[100px] pb-[100px]">
      <Reveal variant="fade">
        <Eyebrow>I · HOW IT WORKS · COMPUTE INTELLIGENCE</Eyebrow>
      </Reveal>
      <Reveal as="h2" variant="blur" delay={120} duration={1000} className="m-0 mt-[22px] text-center font-serif text-[54px] leading-[57px] font-normal tracking-[-1px] text-ink">
        A compute layer for training sweeps.
      </Reveal>
      <Reveal as="p" delay={260} className="m-0 mt-[20px] w-[820px] max-w-full text-center text-[19px] leading-[29px] text-ink-soft">
        Point it at your training code and your machines. It plans, runs and checks the experiments.
      </Reveal>

      <div className="mt-16 flex w-full max-w-[1200px] items-stretch">
        <div className="w-[250px] shrink-0">
          <Node tag="IN" title="Your code + machines" caption="Any training code. Modal today; any compute plugs in." delay={0} />
        </div>

        <Connector delay={300} />

        <Reveal variant="scale" delay={450} duration={900} className="min-w-0 flex-1 bg-plate px-7 pt-6 pb-5 outline outline-1 -outline-offset-[0.5px] outline-cobalt">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-[10px] tracking-[1.8px] text-brass-ink">THE ENGINE</span>
            <span className="font-serif text-[26px] leading-[30px] text-ink">Busy Machines</span>
          </div>
          <div className="mt-4 flex flex-col">
            {ENGINE.map((p, i) => (
              <Reveal
                key={p.label}
                variant="right"
                delay={650 + i * 180}
                className="flex items-center gap-4 border-t border-cobalt/15 py-[13px]"
              >
                <span className="w-[110px] shrink-0 font-mono text-[11px] tracking-[1.6px] text-cobalt">{p.label}</span>
                <span className="min-w-0 flex-1 text-[15px] leading-[21px] text-ink">{p.caption}</span>
                {p.mubit && (
                  <span className="shrink-0 bg-[#E6E4F4] px-2 py-[3px] font-mono text-[9.5px] tracking-[1.6px] text-ink">RUNS ON MUBIT</span>
                )}
              </Reveal>
            ))}
          </div>
        </Reveal>

        <Connector delay={1300} />

        <div className="w-[250px] shrink-0">
          <Node tag="OUT" title="A better model" caption="A lower bill. Nothing fake shipped." delay={1450} />
        </div>
      </div>
    </section>
  );
}
