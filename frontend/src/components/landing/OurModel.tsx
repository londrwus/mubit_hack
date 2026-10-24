import { ArrowRight, ArrowUpRight, Box } from "lucide-react";
import { HUGGING_FACE_URL, RESULTS_LOG_URL } from "@/lib/links";
import { metric, side } from "@/lib/showcase";
import { Eyebrow, FieldNote, OrnamentDivider } from "./Ornament";

const TRACK_H = 200;

function Column({ value, of, strong }: { value: number; of: number; strong?: boolean }) {
  const h = Math.round((TRACK_H * value) / of);
  return (
    <div className="flex flex-1 flex-col items-center">
      <div className="flex h-[200px] w-[132px] flex-col items-center justify-end gap-2 outline outline-1 -outline-offset-[0.5px] outline-cobalt/[0.12]">
        <span className={`font-serif text-[34px] leading-none ${strong ? "text-cobalt" : "text-ink-soft"}`}>{value}</span>
        {value > 0 ? (
          <span className="hatch-engrave block w-full" style={{ height: h }} />
        ) : (
          <span className="block h-[2px] w-full bg-cobalt/40" />
        )}
      </div>
    </div>
  );
}

/** S3 · Our model (Pen MgmgS): published Banyan Sapling vs our sapling-research v2 (R81). */
export function OurModel() {
  const caught = metric("sapling", "fakes_caught");
  const noAnswer = metric("sapling", "no_answer");
  const approved = metric("sapling", "fakes_approved");
  const pub = side(caught, "published_sapling");
  const ours = side(caught, "ours_v2_looser");
  const of = ours.of ?? 109;
  const pubNoAnswer = side(noAnswer, "published_sapling");
  const unsureOnAll = pubNoAnswer.of !== undefined && pubNoAnswer.value === pubNoAnswer.of;
  const oursApproved = side(approved, "ours_v2_looser");

  return (
    <section id="model" className="bg-paper-cobalt relative overflow-hidden">
      <OrnamentDivider />
      <div className="mx-auto flex max-w-[1440px] items-center gap-[88px] px-[120px] pt-[120px] pb-[128px]">
        <div className="flex w-[500px] shrink-0 flex-col items-start gap-[26px]">
          <Eyebrow tone="brass">II · OUR MODEL</Eyebrow>
          <h2 className="m-0 font-serif text-[60px] leading-[63px] font-normal tracking-[-0.8px] text-ivory">
            Our own model,
            <br />
            built on Mubit's,
            <br />
            with a single GPU.
          </h2>
          <div className="font-mono text-[12px] tracking-[1.6px] text-brass">ONE H100 · ABOUT $1 OF COMPUTE · R33, R42</div>
          <p className="m-0 text-[19px] leading-[29px] text-ivory-dim">
            sapling-research catches hallucinated wins: results that look real but never replicate.
          </p>
          <div className="flex items-center gap-7 pt-[14px]">
            {/* TODO(release): HUGGING_FACE_URL is a placeholder ("#hf") until the model page is published. */}
            <a
              href={HUGGING_FACE_URL}
              data-todo="hugging-face-url"
              title="Placeholder: the Hugging Face page is not published yet"
              className="flex items-center gap-4 bg-ivory py-2 pr-2 pl-4 hover:brightness-105"
            >
              <span className="whitespace-nowrap text-[15px] font-medium text-cobalt-deep">View on Hugging Face</span>
              <span className="grid size-[30px] place-items-center bg-cobalt">
                <ArrowUpRight size={16} className="text-ivory" />
              </span>
            </a>
            <a href={RESULTS_LOG_URL} target="_blank" rel="noreferrer" className="flex items-center gap-2 text-[15px] font-medium text-ivory hover:underline">
              Read the results <ArrowRight size={16} />
            </a>
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col gap-[22px]">
          <div className="flex flex-col gap-[30px] bg-cream px-11 pt-10 pb-9 shadow-[0_24px_60px_#0A0B1055] outline outline-1 -outline-offset-[0.5px] outline-ivory">
            <div className="flex flex-col gap-[14px]">
              <div className="flex items-center gap-[10px]">
                <Box size={20} strokeWidth={1.6} className="text-ink" />
                <span className="font-mono text-[20px] font-medium text-ink">sapling-research</span>
              </div>
              <div className="flex gap-2">
                <span className="rounded-[4px] bg-cobalt px-[10px] py-[5px] text-[13px] font-medium text-ivory">Open model</span>
                <span className="rounded-[4px] bg-cream px-[10px] py-[5px] text-[13px] font-medium text-ink outline outline-1 -outline-offset-[0.5px] outline-cobalt/[0.33]">
                  Built on Mubit's Banyan Sapling
                </span>
              </div>
            </div>
            <div className="h-px w-full bg-cobalt/20" />
            <div className="font-mono text-[11px] tracking-[1.2px] text-ink-soft" title={caught.hover}>
              HALLUCINATED WINS CAUGHT · OF {of}
            </div>
            <div className="flex flex-col">
              <div className="flex gap-10 border-b border-cobalt/40 px-6">
                <Column value={pub.value} of={of} />
                <Column value={ours.value} of={of} strong />
              </div>
              <div className="flex gap-10 px-6 pt-[14px]">
                <div className="flex flex-1 flex-col items-center gap-[6px] text-center">
                  <span className="font-serif text-[20px] text-ink">Mubit's model, as published</span>
                  {unsureOnAll && <span className="text-[15px] text-ink-soft">said “unsure” on every claim</span>}
                </div>
                <div className="flex-1 text-center font-serif text-[20px] italic text-cobalt">Ours</div>
              </div>
            </div>
            <p className="m-0 text-[15px] leading-[21px] text-ink-soft">
              Same base model; we retrained its checking layer on ML-research claims.
            </p>
            <div className="font-mono text-[11px] tracking-[1.2px] text-ink-soft">
              {ours.r_id} · {oursApproved.value === 0 ? "NONE WRONGLY APPROVED" : `${oursApproved.display.toUpperCase()} WRONGLY APPROVED`}
            </div>
          </div>
          <div className="px-1">
            <FieldNote dark>A sweep reported 99.9%. On fresh data it scored 88.7%. (R54)</FieldNote>
          </div>
        </div>
      </div>
    </section>
  );
}
