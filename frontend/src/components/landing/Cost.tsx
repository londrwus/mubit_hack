import { Reveal } from "@/components/motion/Reveal";
import { CountUp } from "@/components/motion/CountUp";
import { Eyebrow } from "./Ornament";

type Pair = { title: string; tag: string; std: number; bm: number };

// Logged means: R78 (game-playing, 3 runs), R82 (tabular, 3 runs).
const PAIRS: Pair[] = [
  { title: "AI bill · game-playing", tag: "R78 · 3 RUNS", std: 3.3, bm: 0.83 },
  { title: "AI bill · tabular tuning", tag: "R82 · 3 RUNS", std: 2.77, bm: 0.7 },
  { title: "Total, AI + machines · game-playing", tag: "R78 · 3 RUNS", std: 3.65, bm: 1.18 },
];
const MONEY_MAX = 3.65;
const BAR_W = 300;

function Bar({ value, strong, delay }: { value: number; strong?: boolean; delay: number }) {
  return (
    <div className="flex items-center gap-4">
      <span className={`w-[130px] shrink-0 text-[15px] ${strong ? "font-medium text-cobalt" : "text-ink-soft"}`}>
        {strong ? "Busy Machines" : "Standard agent"}
      </span>
      <span
        className={`grow-x block h-[16px] ${strong ? "bg-cobalt" : "bg-bar-grey"}`}
        style={{ width: Math.max(4, Math.round((BAR_W * value) / MONEY_MAX)), transitionDelay: `${delay}ms` }}
      />
      <CountUp
        value={value}
        format={(n: number) => `$${n.toFixed(2)}`}
        delay={delay}
        duration={1100}
        className={`font-serif tabular-nums ${strong ? "text-[26px] italic text-cobalt" : "text-[24px] text-ink-soft"}`}
      />
    </div>
  );
}

type Layer = { tag: string; tagCls: string; boxCls: string; title: string; body: string; refs: string };
const LAYERS: Layer[] = [
  {
    tag: "MEASURED",
    tagCls: "bg-cobalt text-ivory",
    boxCls: "outline-cobalt/15",
    title: "Fewer, better-timed AI calls",
    body: "4× smaller AI bill on two tasks, with Opus answering every call.",
    refs: "R78 · R82",
  },
  {
    tag: "MUBIT · NEXT",
    tagCls: "bg-[#E6E4F4] text-ink",
    boxCls: "outline-dashed! outline-brass",
    title: "Small models we own",
    body: "Every call runs through Mubit, which turns our recurring steps into small models we own. Trained; none serving yet.",
    refs: "R79 · R80",
  },
];

/** II · Cost (Pen bVVJQ): the AI bill only. Three measured bills (R78, R82) and where the saving comes from. */
export function Cost() {
  return (
    <section id="cost" className="flex flex-col items-center bg-cream px-[120px] pt-[64px] pb-[60px]">
      <Reveal variant="fade">
        <Eyebrow>II · COST</Eyebrow>
      </Reveal>
      <Reveal as="h2" variant="blur" delay={120} duration={1000} className="m-0 mt-[22px] text-center font-serif text-[54px] leading-[57px] font-normal tracking-[-1px] text-ink">
        A quarter of the AI bill.
        <br />
        Built on Mubit.
      </Reveal>
      <Reveal as="p" delay={260} className="m-0 mt-[20px] w-[760px] max-w-full text-center text-[19px] leading-[29px] text-ink-soft">
        We call the AI only when it's worth it.
      </Reveal>

      <div className="mt-8 flex w-full max-w-[1200px] items-start gap-7">
        <Reveal variant="scale" duration={900} className="w-[600px] shrink-0">
          <div className="bg-plate px-9 pt-7 pb-6 outline outline-1 -outline-offset-[0.5px] outline-cobalt/15">
            <div className="font-mono text-[11px] tracking-[2px] text-brass-ink">WHAT ONE SWEEP COSTS · MEASURED</div>
            {PAIRS.map((p, i) => {
              const d = 300 + i * 180;
              return (
                <Reveal key={p.title} variant="fade" delay={d - 200} className="flex flex-col gap-2 border-b border-cobalt/15 py-3">
                  <div className="flex items-baseline justify-between gap-4">
                    <span className="text-[17px] font-medium text-ink">{p.title}</span>
                    <span className="shrink-0 font-mono text-[10px] tracking-[1.6px] text-ink-soft">{p.tag}</span>
                  </div>
                  <Bar value={p.std} delay={d} />
                  <Bar value={p.bm} strong delay={d + 150} />
                </Reveal>
              );
            })}
            <p className="m-0 mt-4 font-mono text-[11px] tracking-[0.4px] text-ink-soft">
              List-price AI estimates; Mubit fees not included (R72).
            </p>
          </div>
        </Reveal>

        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <Reveal variant="fade" className="font-mono text-[11px] tracking-[2px] text-brass-ink">
            WHERE THE SAVING COMES FROM
          </Reveal>
          {LAYERS.map((l, i) => (
            <Reveal
              key={l.tag}
              variant="right"
              delay={250 + i * 180}
              className={`flex flex-col gap-3 bg-plate px-8 pt-6 pb-6 outline outline-1 -outline-offset-[0.5px] transition-transform duration-300 hover:-translate-y-0.5 ${l.boxCls}`}
            >
              <span className={`self-start px-2 py-[3px] font-mono text-[10px] tracking-[1.8px] ${l.tagCls}`}>{l.tag}</span>
              <span className="font-serif text-[26px] leading-[30px] text-ink">{l.title}</span>
              <p className="m-0 text-[15px] leading-[23px] text-ink-soft">{l.body}</p>
              <span className="font-mono text-[10px] tracking-[1.6px] text-ink-soft">{l.refs}</span>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
