"""Cross-race aggregates for the honest-claims audit. Read-only (analyze_races.load_db uses mode=ro)."""
import importlib.util, json, statistics, sys
from pathlib import Path

REPO = Path("C:/Users/Lenovo/Documents/mubit_hack")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("analyze_races", REPO / "scripts" / "analyze_races.py")
ar = importlib.util.module_from_spec(spec); sys.modules["analyze_races"] = ar; spec.loader.exec_module(ar)
from backend.benchmark.metrics import summarize_arm, quality, is_counted

SETS = {
    "rl R48": ["8a76fa052c28441cb2aea221f099a7d2", "45ab180f9fdc403e93f6f16960e6b0f9", "59d7f7ebe0bc4b6ba59edacd0d769f49"],
    "lcbench R38": ["5245bacc19804b3eacc3a45f68e3fa7b", "91a4e48d33eb44e8a93a0d580726c873", "f98b671246594117884b7f5aa0fc1f60"],
    "lcbench R44": ["2ded10bcaab747bfab6071c3f4104fae", "e3fddb29e50e48b2857c5f676904534a", "aaedd18076874e6f9386c0b573239988"],
    "cifar R49+s1": ["60477ba130e34d74acd157b460a347c0", "949fce51513c4368948b4eee22b66337"] + sys.argv[1:],
}
DB = REPO / "data/bench/bench.db"
out = {}
for name, ids in SETS.items():
    races = ar.load_db(DB, ids)
    rows_out = []
    for rows in races:
        race = rows["races"][0]
        cfg = race.get("config") or {}
        tv = cfg.get("target_value"); direction = cfg.get("direction") or ("max" if race["target"] in ("cifar", "lcbench", "rl") else "min")
        for arm in rows["arms"]:
            aid = arm["id"]
            exps = [e for e in rows["experiments"] if e["arm_id"] == aid]
            ivs = [i for i in rows["gpu_intervals"] if i["arm_id"] == aid]
            calls = [c for c in rows["llm_calls"] if c["arm_id"] == aid]
            s, x = summarize_arm(arm["kind"], direction=direction, target_value=float(tv), race_start=float(race["started_at"]),
                                 experiments=exps, intervals=ivs, llm_calls=calls)
            shipped = next((e for e in exps if e["id"] == x["shipped_experiment_id"]), None)
            best_ever = [e for e in exps if e.get("is_best") and e.get("status") == "succeeded"]
            trap_best = [e for e in best_ever if e.get("is_trap")]
            # gap: shipped model reported score vs held-out
            rows_out.append(dict(
                race=race["id"][:8], seed=race.get("seed"), arm=arm["kind"], direction=direction, goal=tv,
                best=s.best_score, shipped_q=x["shipped_quality"], shipped_rep=x["shipped_reported_score"],
                shipped_is_trap=bool(shipped and shipped.get("is_trap")),
                trap_runs=x["trap_runs"], fake_best=x["fake_best_accepted"], n_best_events=len(best_ever),
                judge_caught=x["fake_caught_by_judge"], genuine_rejected=x["genuine_rejected_by_judge"],
                gpu=s.gpu_cost_usd, llm=s.llm_cost_usd, total=s.gpu_cost_usd + s.llm_cost_usd,
                finished=s.experiments_finished, idle=s.gpu_idle_pct, ttt=s.time_to_target_s, ctt=s.cost_to_target_usd,
                idle_cost=x["idle_cost_usd"], llm_calls=s.llm_calls, invalid=s.llm_invalid_calls,
                n_exps=len(exps)))
    out[name] = rows_out

json.dump(out, open(Path(__file__).with_name("agg.json"), "w"), indent=1, default=str)
for name, rs in out.items():
    print("==", name)
    for r in rs:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("direction",)})
