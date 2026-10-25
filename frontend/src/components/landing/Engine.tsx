import { BadgeCheck, LayoutGrid, Timer } from "lucide-react";
import { metric, side, displayed } from "@/lib/showcase";
import { Reveal } from "@/components/motion/Reveal";
import { CountUp } from "@/components/motion/CountUp";
import { Eyebrow, FieldNote } from "./Ornament";
import { TimeToResult } from "./TimeToResult";
import "./engine.motion.css";

const MACHINES = 30; // R78: 30 machines per side
// Where the busy machines sit on the standard agent's strip (the Pen layout; the count comes from the data).
const DESIGN_BUSY = [1, 2, 7, 11, 12, 17, 22, 26, 27];

function busyIndexes(count: number): Set<number> {
  if (count === DESIGN_BUSY.length) return new Set(DESIGN_BUSY);
  if (count >= MACHINES) return new Set(Array.from({ length: MACHINES }, (_, i) => i));
  const step = MACHINES / Math.max(count, 1);
  return new Set(Array.from({ length: count }, (_, i) => Math.floor(i * step + step / 2)));
}

/** One row of machines. `lag` (ms) offsets this row's stagger so the rows switch on one after the other. */
function Strip({
  title,
  italic,
  waiting,
  strong,
  lag = 0,
}: {
  title: string;
  italic?: boolean;
  waiting: number;
  strong?: boolean;
  lag?: number;
}) {
  const busy = busyIndexes(Math.round((MACHINES * (100 - waiting)) / 100));
  const shown = Math.round(waiting);
  return (
    <div className="flex items-center gap-9 py-[22px]">
      <div className={`w-[180px] shrink-0 font-serif text-[28px] leading-tight text-ink ${italic ? "italic" : ""}`}>{title}</div>
      <div className="flex min-w-0 flex-1 gap-1" aria-label={`${MACHINES - busy.size} of ${MACHINES} machines waiting`}>
        {Array.from({ length: MACHINES }, (_, i) => (
          <span key={i} className="engine-cell grow-y hatch-wait block h-16 min-w-0 max-w-[18px] flex-1" style={{ transitionDelay: `${lag + i * 10}ms` }}>
            {busy.has(i) && <span className="engine-fill bg-cobalt" style={{ transitionDelay: `${lag + 350 + i * 10}ms` }} />}
          </span>
        ))}
      </div>
      <div className={`flex w-[170px] shrink-0 items-end justify-end gap-[10px] font-serif ${strong ? "text-cobalt" : "text-ink-soft"}`}>
        <CountUp value={shown} format={(n) => `${Math.round(n)}%`} delay={lag + 250} duration={1000} className="text-[54px] leading-[54px]" />
        <span className="pb-[7px] text-[26px] italic leading-none">waiting</span>
      </div>
    </div>
  );
}

/** III · Time (Pen QYavo + T3KBM merged): the waiting strips once (R78 3-run means, whole percent), then what it turns into. */
export function Engine() {
  const waiting = metric("game_playing", "machines_waiting");
  const std = side(waiting, "standard");
  const bm = side(waiting, "busy_machines");
  return (
    <section id="time" className="flex flex-col items-center border-t border-cobalt/[0.12] bg-[#F3EEE2] px-[120px] pt-[110px] pb-[110px]">
      <span id="engine" aria-hidden className="block h-0 w-0" />
      <Reveal variant="fade">
        <Eyebrow>III · TIME · MACHINES WAITING ON THE AI</Eyebrow>
      </Reveal>
      <Reveal as="h2" variant="blur" delay={120} duration={1000} className="m-0 mt-[22px] text-center font-serif text-[54px] leading-[57px] font-normal tracking-[-1px] text-ink">
        Same model.
        <br />
        We changed when it thinks.
      </Reveal>
      <Reveal as="p" delay={260} className="m-0 mt-[22px] w-[760px] max-w-full text-center text-[19px] leading-[29px] text-ink-soft">
        The planner thinks less often, so the machines almost never wait.
      </Reveal>

      <Reveal variant="scale" duration={900} className="mt-14 w-full max-w-[1200px]">
        <div className="w-full bg-plate px-11 pt-9 pb-[26px] outline outline-1 -outline-offset-[0.5px] outline-cobalt/15" title={waiting.hover}>
          <div className="border-b border-cobalt/15">
            <Strip title="Standard agent" waiting={displayed(std)} />
          </div>
          <Strip title="Busy Machines" italic strong waiting={displayed(bm)} lag={300} />
          <div className="flex items-center gap-5 border-t border-cobalt/15 pt-[18px]">
            <span className="flex items-center gap-2 font-mono text-[11px] tracking-[1.4px] text-ink-soft">
              <span className="block size-3 bg-cobalt" /> BUSY
            </span>
            <span className="flex items-center gap-2 font-mono text-[11px] tracking-[1.4px] text-ink-soft">
              <span className="hatch-wait block size-3 outline outline-1 -outline-offset-[0.5px] outline-cobalt/20" /> WAITING
            </span>
            <span className="flex-1" />
            <span className="font-mono text-[11px] tracking-[1.4px] text-brass-ink">
              {MACHINES} MACHINES EACH · {std.r_id}
            </span>
          </div>
        </div>
      </Reveal>

      <Reveal variant="fade" delay={200} className="mt-5">
        <FieldNote>{`30 machines, one sweep: most sat waiting while the planner thought. (${std.r_id})`}</FieldNote>
      </Reveal>

      <div className="mt-10 flex items-center gap-12 text-[16px] font-medium text-ink">
        <Reveal as="span" delay={0} className="flex items-center gap-[10px]">
          <Timer size={20} strokeWidth={1.6} /> Plans less often
        </Reveal>
        <Reveal as="span" delay={140} className="flex items-center gap-[10px]">
          <LayoutGrid size={20} strokeWidth={1.6} /> Fills idle machines
        </Reveal>
        <Reveal as="span" delay={280} className="flex items-center gap-[10px]">
          <BadgeCheck size={20} strokeWidth={1.6} /> Checks every gain
        </Reveal>
      </div>

      <TimeToResult />
    </section>
  );
}
