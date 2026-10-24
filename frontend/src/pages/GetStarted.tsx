import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  Check,
  CircleCheck,
  CircleDashed,
  Copy,
  EyeOff,
  Gamepad2,
  GitBranch,
  KeyRound,
  Laptop,
  Play,
  Server,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { PageHeader } from "@/components/platform/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { API_BASE } from "@/lib/api";
import { cn } from "@/lib/utils";

const PRESET = "rl-wide-bm-cadence-15m";
const CLI = `python scripts/bench.py run ${PRESET} --yes`;

// The preset's RaceConfig as POST /api/races takes it (backend/app.py create_race(cfg: RaceConfig)).
// = PRESETS["rl-wide-bm-cadence-15m"].model_dump(mode="json", exclude_defaults=True) in
// backend/benchmark/configs.py. The endpoint has no {"preset": ...} shortcut, so the full body is sent.
const PRESET_BODY = {
  name: "RL wide 15 min (15 s runs), Busy Machines with Opus + planner cadence 60 s",
  target: "rl",
  arms: ["opus", "opus_bm", "optuna"],
  budget_s: 900.0,
  gpus_per_arm: 30,
  cost_cap_usd: 8.0,
  run_budget_s: 15,
  max_proposals_per_call: 8,
  launch_cutoff_s: 17.0,
  prewarm: true,
  prewarm_spare: 2,
  mode: "benchmark",
  llm: "mubit",
  runner: "modal",
  record_mode: { opus: "off", opus_bm: "off" },
  max_estimated_gpu_usd: 1.25,
  engine_features: { opus_bm: ["backfill", "judge_policy", "lean_planner", "planner_cadence"] },
  feature_params: { opus_bm: { planner_cadence: { min_propose_interval_s: 60.0 } } },
};

const CURL = [
  `# 1. create the race (body = the ${PRESET} preset's RaceConfig)`,
  `curl -s -X POST ${API_BASE}/api/races -H "content-type: application/json" \\`,
  `  -d '${JSON.stringify(PRESET_BODY)}'`,
  `# 2. start it: POST ${API_BASE}/api/races/<id>/start`,
].join("\n");

function useCopied(): [string | null, (key: string, text: string) => void] {
  const [copied, setCopied] = useState<string | null>(null);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(null), 2200);
    return () => clearTimeout(t);
  }, [copied]);
  const copy = (key: string, text: string) => {
    void navigator.clipboard?.writeText(text).then(
      () => setCopied(key),
      () => setCopied(null),
    );
  };
  return [copied, copy];
}

function Status({ done }: { done: boolean }) {
  return done ? (
    <Badge variant="ok" size="status">
      <CircleCheck size={14} /> Done
    </Badge>
  ) : (
    <Badge variant="next" size="status">
      <CircleDashed size={14} /> Next
    </Badge>
  );
}

function Step({ n, title, sub, done, current, children }: { n: number; title: string; sub: string; done: boolean; current?: boolean; children: ReactNode }) {
  return (
    <section
      className={cn(
        "flex w-full items-start gap-[22px] rounded-[16px] bg-white px-7 py-[22px]",
        current
          ? "shadow-[0_4px_16px_#1D3B9418] outline outline-[1.5px] -outline-offset-[0.75px] outline-app-wash-line"
          : "shadow-[0_1px_2px_#1A2F7A0D] outline outline-1 -outline-offset-[0.5px] outline-app-line",
      )}
    >
      <div
        className={cn(
          "grid size-11 shrink-0 place-items-center rounded-[12px] font-serif text-[24px] leading-none",
          current ? "bg-gradient-to-b from-cobalt-bright to-cobalt text-white" : "bg-app-wash text-cobalt",
        )}
      >
        {n}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-[14px]">
        <div className="flex flex-col gap-1">
          <h2 className="m-0 text-[20px] font-semibold text-ink">{title}</h2>
          <p className="m-0 text-[14px] text-app-slate">{sub}</p>
        </div>
        {children}
      </div>
      <Status done={done} />
    </section>
  );
}

function Chip({ icon: Icon, children }: { icon: typeof Server; children: ReactNode }) {
  return (
    <span className="flex h-[30px] items-center gap-[6px] rounded-[8px] bg-white px-[10px] text-[12px] text-app-text outline outline-1 -outline-offset-[0.5px] outline-app-line">
      <Icon size={12} strokeWidth={1.8} className="text-app-slate" /> {children}
    </span>
  );
}

/** "/app": three onboarding steps (Pen P6OjMG). */
export default function GetStarted() {
  const [copied, copy] = useCopied();
  useEffect(() => {
    document.title = "Get started · Busy Machines";
  }, []);

  return (
    <>
      <PageHeader
        eyebrow="Get started"
        title="Start in three steps"
        subtitle="Bring your Mubit key and your compute. Busy Machines keeps every machine training while the AI thinks."
        right={
          <Badge size="status" className="px-3 font-mono">
            <CircleCheck size={13} /> 2 of 3 done
          </Badge>
        }
      />

      <div className="flex w-full flex-col gap-4 pt-2">
        <Step n={1} title="Add your Mubit key" sub="Paste one key. Your provider keys stay in Mubit." done>
          <div className="flex items-center gap-[10px]">
            <div className="flex h-11 w-[420px] items-center gap-2 rounded-[9px] bg-app-field px-3 outline outline-1 -outline-offset-[0.5px] outline-app-line">
              <KeyRound size={14} className="text-app-slate" />
              <span className="font-mono text-[12px] text-app-slate">MUBIT_API_KEY</span>
              <span className="flex-1 font-mono text-[13px] text-app-text">mbt_••••••••••••••••</span>
              <EyeOff size={14} className="text-app-slate" />
            </div>
            <span className="flex items-center gap-[6px] rounded-[8px] bg-app-side px-[10px] py-[6px] text-[12px] font-medium text-cobalt">
              <ShieldCheck size={13} /> Every AI call runs through Mubit
            </span>
          </div>
        </Step>

        <Step n={2} title="Connect your compute" sub="Experiments run on your own machines, never on ours." done>
          <div className="flex items-center gap-[10px]">
            <span className="flex h-11 items-center gap-2 rounded-[9px] bg-app-field px-3 outline outline-1 -outline-offset-[0.5px] outline-app-line">
              <span className="size-2 rounded-full bg-[#2E9B5F]" />
              <span className="text-[13px] font-semibold text-app-text">Modal</span>
              <span className="font-mono text-[12px] text-app-slate">token ak-••••••</span>
            </span>
            <span className="flex h-11 items-center gap-2 rounded-[9px] px-3 text-[13px] text-app-slate outline outline-1 -outline-offset-[0.5px] outline-app-line">
              <Laptop size={14} /> Local
            </span>
            <span className="flex h-11 items-center gap-2 rounded-[9px] px-3 text-[13px] text-app-slate outline outline-1 -outline-offset-[0.5px] outline-app-line">
              <Server size={14} /> Your cluster
            </span>
            <span className="text-[12.5px] text-app-slate">Modal today; any runner</span>
          </div>
        </Step>

        <Step
          n={3}
          title="Start a sweep"
          sub="Pick a preset. Both sides get the same machines, budget and starting code; only when the AI thinks differs."
          done={false}
          current
        >
          <div className="flex flex-wrap items-center gap-2">
            <Chip icon={Gamepad2}>Game-playing preset</Chip>
            <Chip icon={Server}>30 machines each</Chip>
            <Chip icon={Timer}>15 minutes</Chip>
            <Chip icon={GitBranch}>Same starting code</Chip>
          </div>

          <div className="w-full overflow-hidden rounded-[12px] bg-code">
            <div className="flex items-center gap-[6px] border-b border-code-line px-[14px] py-[10px]">
              <span className="rounded-[6px] bg-code-line px-[10px] py-1 font-mono text-[11px] font-semibold text-white">API</span>
              <span className="flex-1" />
              <Button variant="code" size="sm" onClick={() => copy("api", CURL)} title="Copies the full request as curl">
                {copied === "api" ? <Check size={12} /> : <Copy size={12} />} {copied === "api" ? "Copied" : "Copy"}
              </Button>
            </div>
            <div className="flex flex-col gap-2 px-5 py-4 font-mono text-[15px]">
              <div className="flex gap-[10px]">
                <span className="font-semibold text-[#7FA1FF]">POST</span>
                <span className="text-ivory">/api/races</span>
              </div>
              <div className="flex min-w-0 whitespace-nowrap">
                <span className="text-[#C9CEDD]">{"{"}</span>
                <span className="text-[#C9CEDD]">"target": </span>
                <span className="text-[#E3C07A]">"rl"</span>
                <span className="text-[#C9CEDD]">, "arms": </span>
                <span className="text-[#E3C07A]">["opus", "opus_bm", "optuna"]</span>
                <span className="text-[#C9CEDD]">, "gpus_per_arm": </span>
                <span className="text-[#E3C07A]">30</span>
                <span className="text-[#C9CEDD]">, </span>
                <span className="text-[#8D93A8]" title={`The rest of the ${PRESET} preset's RaceConfig. Copy includes every field.`}>
                  …
                </span>
                <span className="text-[#C9CEDD]">{"}"}</span>
              </div>
              <div className="flex gap-[10px]">
                <span className="font-semibold text-[#7FA1FF]">POST</span>
                <span className="text-ivory">/api/races/{"{id}"}/start</span>
              </div>
            </div>
            <div className="flex items-center gap-[10px] border-t border-code-line bg-code-2 px-5 py-[14px]">
              <span className="text-[11px] font-medium text-[#8D93A8]">or</span>
              <span className="font-mono text-[13px] text-[#8D93A8]">$</span>
              <span className="flex-1 font-mono text-[14px] text-ivory">{CLI}</span>
              <Button variant="code" size="sm" onClick={() => copy("cli", CLI)} title="Copy the CLI command">
                {copied === "cli" ? <Check size={12} /> : <Copy size={12} />}
              </Button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <Button
              size="lg"
              onClick={() => copy("start", CLI)}
              title="A real sweep calls Opus 5.5 through Mubit and runs on your compute. This copies the command to run it."
            >
              {copied === "start" ? <Check size={14} /> : <Play size={14} />}
              {copied === "start" ? "Command copied: run it from the repo root" : "Start a sweep"}
            </Button>
            <Link to="/app/compare" className="text-[13.5px] font-semibold text-cobalt hover:underline">
              Try it free: replay a logged sweep →
            </Link>
          </div>
        </Step>
      </div>
    </>
  );
}
