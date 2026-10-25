import { useState } from "react";
import { CircleAlert, CircleCheck, Eye, EyeOff, KeyRound, LoaderCircle, PlugZap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { checkMubitKey, type KeyCheck } from "@/lib/liveRace";

/** Step 1: paste a Mubit key (or leave it empty to check the server's key) and test it with one tiny call. */
export function KeyTest() {
  const [key, setKey] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<KeyCheck | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const test = async () => {
    setBusy(true);
    setRes(null);
    setErr(null);
    try {
      setRes(await checkMubitKey(key));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-[10px]">
        <label className="flex h-11 w-[420px] max-w-full items-center gap-2 rounded-[9px] bg-app-field px-3 outline outline-1 -outline-offset-[0.5px] outline-app-line focus-within:outline-cobalt-bright">
          <KeyRound size={14} className="shrink-0 text-app-slate" />
          <span className="shrink-0 font-mono text-[12px] text-app-slate">MUBIT_API_KEY</span>
          <input
            type={show ? "text" : "password"}
            value={key}
            onChange={(e) => {
              setKey(e.target.value);
              setRes(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && void test()}
            placeholder="empty = use the server's key"
            autoComplete="off"
            spellCheck={false}
            aria-label="Mubit API key"
            className="min-w-0 flex-1 bg-transparent font-mono text-[13px] text-app-text outline-none placeholder:text-app-muted"
          />
          <button
            type="button"
            onClick={() => setShow((v) => !v)}
            className="cursor-pointer text-app-slate hover:text-app-text"
            aria-label={show ? "Hide key" : "Show key"}
          >
            {show ? <Eye size={14} /> : <EyeOff size={14} />}
          </button>
        </label>
        <Button onClick={() => void test()} disabled={busy} title="One 1-token call to the cheapest model through Mubit, not recorded">
          {busy ? <LoaderCircle size={14} className="animate-spin" /> : <PlugZap size={14} />}
          {busy ? "Testing…" : "Test key"}
        </Button>
        {res && (
          <span className="animate-pop-in">
            {res.ok ? (
              <Badge variant="ok" size="status" title={res.request_id ? `Mubit request ${res.request_id}` : undefined}>
                <CircleCheck size={14} />
                {res.mode === "call"
                  ? `Connected · ${res.latency_ms} ms`
                  : "Server key configured (no call made)"}
              </Badge>
            ) : (
              <Badge variant="grey" size="status" className="max-w-[460px] whitespace-normal text-[#A23B2A]">
                <CircleAlert size={14} /> {res.error ?? "Key check failed"}
              </Badge>
            )}
          </span>
        )}
        {err && (
          <Badge variant="grey" size="status" className="text-[#A23B2A]">
            <CircleAlert size={14} /> {err}
          </Badge>
        )}
      </div>
      {res?.ok && res.mode === "call" && (
        <p className="animate-enter m-0 text-[12px] text-app-slate">
          Answered by {res.model} through the Mubit gateway. The key was used for this one call and is not stored.
        </p>
      )}
    </div>
  );
}
