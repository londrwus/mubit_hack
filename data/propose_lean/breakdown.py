"""Token breakdown of stored R48 propose replies (count_tokens via Mubit, record off; free)."""
import asyncio, json, random, sqlite3, statistics as st, sys
from pathlib import Path
sys.path.insert(0, r"C:/Users/Lenovo/Documents/mubit_hack")
from backend.agent.mubit_client import MubitClient
import workers.targets  # noqa: registers targets
from workers.targets.base import get_target

DB = r"C:/Users/Lenovo/Documents/mubit_hack/data/bench/bench.db"
spec = get_target("rl").spec
order = list(spec.config_space)

def val(v):
    if isinstance(v, bool): return "true" if v else "false"
    return str(v)

def settings(cfg): return ", ".join(f"{k}={val(cfg[k])}" for k in order if k in cfg)

async def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    races = [r[0] for r in c.execute("select id from races where substr(id,1,8) in ('8a76fa05','45ab180f','59d7f7eb')")]
    calls = c.execute(f"select id,output_tokens,experiment_ids_json,latency_ms from llm_calls where step_type='propose_experiments' and race_id in ({','.join('?'*3)}) order by created_at", races).fetchall()
    random.Random(0).shuffle(calls); calls = calls[:30]
    m = MubitClient(); H = {"x-middleware-record": "off"}
    async def ntok(text):
        r = await m._client.messages.count_tokens(model="claude-opus-5-5", messages=[{"role": "user", "content": text}], extra_headers=H)
        return r.input_tokens
    base = await ntok("x")
    rows = []
    for cid, out, ids_j, lat in calls:
        ids = json.loads(ids_j)
        exps = c.execute(f"select config_json,hypothesis from experiments where id in ({','.join('?'*len(ids))})", ids).fetchall()
        items = [{"config": settings(json.loads(cj)), "hypothesis": h} for cj, h in exps]
        full = json.dumps({"experiments": items})
        t_full = await ntok(full) - base + 1
        t_set = sum([await ntok(i["config"]) - base + 1 for i in items])
        t_hyp = sum([await ntok(i["hypothesis"]) - base + 1 for i in items])
        k = len(items)
        rows.append(dict(k=k, out=out, visible=t_full, settings=t_set, hyp=t_hyp, json=t_full - t_set - t_hyp,
                         thinking=out - t_full, words=st.mean(len(i["hypothesis"].split()) for i in items), lat=lat))
    await m.aclose()
    def per(key): return st.mean(r[key] / r["k"] for r in rows)
    print(f"n calls {len(rows)}; K {st.mean(r['k'] for r in rows):.2f}")
    for key in ("out", "visible", "settings", "hyp", "json", "thinking"):
        print(f"  {key:9s} per proposal mean {per(key):7.1f}  ({100*per(key)/per('out'):5.1f}% of billed output)")
    print(f"  hypothesis words mean {st.mean(r['words'] for r in rows):.1f}")
    th = [r["thinking"] for r in rows]; print(f"  thinking per call p10/p50/p90 {sorted(th)[3]}/{st.median(th)}/{sorted(th)[26]}; min {min(th)} max {max(th)}")
    Path(__file__).with_name("breakdown.json").write_text(json.dumps(rows, indent=1))
asyncio.run(main())
