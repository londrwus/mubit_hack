"""Final slide charts (light + dark PNGs) into data/charts/final/.

1. headline-two-tasks      R78 + R82, with R75 beside R82 ("first pre-registered setting, failed"):
                           AI bill per race and model quality, standard Opus vs Busy Machines.
2. gpus-waiting-r78        GPUs waiting over time in one R78 race (seed 1), from gpu_intervals.
3. quality-over-time-r82   Best model quality so far per arm in one R82 race (seed 1).
4. sapling-real-claims     R81: published Banyan Sapling vs our fine-tune (v2) vs Opus 5.5.

Every number is computed from the read-only bench DB with the same code as
``scripts/analyze_races.py`` (``summarize_arm`` via ``analyse``), or read from the R81 JSON, and is
checked against the value logged in docs/results_log.md before any chart is drawn (the script
aborts on a mismatch). No LLM calls, no compute, no writes outside data/charts/final/.

Usage:
    python -m uv run python scripts/final_charts.py [--db data/bench/bench.db] [--out data/charts/final]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import analyze_races as ar  # noqa: E402
from backend.benchmark.metrics import is_counted, quality  # noqa: E402

# --------------------------------------------------------------------------- data selection

RACES = {
    "R78": ["4ca4ed27", "d4913e95", "145c2a2d"],   # rl, seeds 0-2
    "R82": ["c1f4b1d9", "9d0cfd77", "b3922d0e"],   # lcbench, 30 s pace, seeds 0-2
    "R75": ["613e2537", "6afbbe8a", "c72b3af1"],   # lcbench, 90 s pace, seeds 0-2 (FAILED)
}
# Logged 3-run means (docs/results_log.md) the computed values must reproduce.
LOGGED = {
    "R78": {"opus": (3.30, -154.5), "opus_bm": (0.83, -154.3), "optuna_best": -157.3},
    "R82": {"opus": (2.774, 0.8897), "opus_bm": (0.698, 0.8884), "optuna_best": 0.8576},
    "R75": {"opus": (2.831, 0.8892), "opus_bm": (0.347, 0.8816), "optuna_best": 0.8493},
}
# One race per time chart: seed 1, the middle of the three runs on the Busy Machines vs standard
# gap (R78 waiting gap 67.6 / 67.4 / 66.6 points; R82 quality gap -0.0001 / -0.0008 / -0.0030).
R78_ONE = "d4913e95"
R82_ONE = "9d0cfd77"
R81_JSON = REPO_ROOT / "data" / "sapling" / "compare_published" / "compare_published.json"

ARMS = ("opus", "opus_bm")
SHORT = {"opus": "Standard AI assistant (Opus 5.5)", "opus_bm": "Busy Machines (Opus 5.5 + our engine)",
         "optuna": "No-AI tuning tool (Optuna)"}
TICK = {"opus": "Standard\nassistant", "opus_bm": "Busy\nMachines"}

# --------------------------------------------------------------------------- theme

GREY = {"light": "#8a8984", "dark": "#8a8984"}


def theme(name: str) -> dict[str, Any]:
    th = dict(ar.THEMES[name])
    arm = dict(th["arm"])
    # Colour follows the entity across every chart: Busy Machines = slot 1 (blue),
    # standard Opus = slot 2 (orange), Optuna = slot 3 (aqua). Validated all-pairs, both modes.
    arm["opus_bm"] = arm["mubit"]
    th["arm"] = arm
    th["grey"] = GREY[name]
    th["bad"] = "#c23b3a" if name == "light" else "#e66767"   # status: critical (FAILED tag only)
    th["good"] = "#0a7a3b" if name == "light" else "#3fb56b"  # status: good (MET tag only)
    return th


plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": True,
    "savefig.dpi": 125,
})
W, H = 16, 9  # 2000 x 1125 px


def new_fig(th: dict[str, Any], **kw: Any):
    fig = plt.figure(figsize=(W, H), facecolor=th["surface"], **kw)
    return fig


def style_ax(ax, th: dict[str, Any], *, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.set_facecolor(th["surface"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=th["muted"], labelsize=14, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=th["grid"], lw=0.8, zorder=0)
    if xgrid:
        ax.xaxis.grid(True, color=th["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def num(v: float, digits: int) -> str:
    """Typographic minus for stage-readable negatives."""
    return f"{v:.{digits}f}".replace("-", "−")


def header(fig, th, title: str, sub: str) -> None:
    fig.text(0.03, 0.955, title, ha="left", va="top", color=th["text"], fontsize=30, fontweight="bold")
    fig.text(0.03, 0.885, sub, ha="left", va="top", color=th["muted"], fontsize=17)


def footer(fig, th, lines: list[str]) -> None:
    fig.text(0.03, 0.018, "\n".join(lines), ha="left", va="bottom", color=th["muted"], fontsize=12.5,
             linespacing=1.45)


# --------------------------------------------------------------------------- loading


def full_ids(db: Path, prefixes: list[str]) -> list[str]:
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    out = []
    for p in prefixes:
        rows = c.execute("select id from races where id like ?", (p + "%",)).fetchall()
        if len(rows) != 1:
            raise SystemExit(f"race prefix {p}: {len(rows)} matches")
        out.append(rows[0][0])
    c.close()
    return out


def load_report(db: Path, rid: str) -> tuple[ar.TargetReport, dict[str, Any]]:
    ids = full_ids(db, RACES[rid])
    races = ar.load_db(db, ids)
    kept, ex = ar.screen(races)
    reps = ar.analyse(kept, exclusions=ex)
    if len(reps) != 1 or reps[0].n_races != 3 or reps[0].warnings or reps[0].is_dev:
        raise SystemExit(f"{rid}: expected one clean 3-race setup, got {[(r.setup_id, r.warnings) for r in reps]}")
    rows_by_race = {r["races"][0]["id"]: r for r in races}
    return reps[0], rows_by_race


def check_logged(rid: str, rep: ar.TargetReport) -> None:
    lg = LOGGED[rid]
    for arm in ARMS:
        llm = rep.arms[arm].stats["llm_cost_usd"].mean
        best = rep.arms[arm].stats["best_quality"].mean
        want_llm, want_best = lg[arm]
        tol_q = 0.05 if rep.target == "rl" else 0.00005
        if abs(llm - want_llm) > 0.006 or abs(best - want_best) > tol_q:
            raise SystemExit(f"{rid} {arm}: computed ${llm:.3f} / {best:.4f} != logged ${want_llm} / {want_best}")
    ob = rep.arms["optuna"].stats["best_quality"].mean
    if abs(ob - lg["optuna_best"]) > (0.05 if rep.target == "rl" else 0.00005):
        raise SystemExit(f"{rid} optuna best {ob} != logged {lg['optuna_best']}")


def per_seed(rep: ar.TargetReport, arm: str, field: str) -> list[float]:
    return [getattr(r, field) for r in sorted(rep.runs, key=lambda r: r.seed) if r.arm == arm]


# --------------------------------------------------------------------------- chart 1


def chart_headline(reps: dict[str, ar.TargetReport], th: dict[str, Any], path: Path) -> None:
    fig = new_fig(th)
    header(fig, th, "Same model, a quarter of the AI bill. And the race we lost first.",
           "Standard AI assistant vs Busy Machines, same model (Opus 5.5), same machines, clock, $ cap and seeds. "
           "3 runs per side.")
    cols = [
        ("R78", "Game-playing AI", "headline race", None),
        ("R82", "Tabular tuning, AI asked every 30 s", "chosen after R75 failed · success line met", "good"),
        ("R75", "Tabular tuning, AI asked every 90 s", "first pre-registered setting, failed", "bad"),
    ]
    left, right, top, bot = 0.075, 0.985, 0.70, 0.20
    gap_x, gap_y = 0.045, 0.085
    cw = (right - left - 2 * gap_x) / 3
    rh = (top - bot - gap_y) / 2
    bill_max = max(max(per_seed(reps[r], a, "llm_cost_usd")) for r in reps for a in ARMS) * 1.32
    lcb = [q for r in ("R82", "R75") for a in ARMS for q in per_seed(reps[r], a, "best_quality")]
    lcb_lim = (min(lcb) - 0.003, max(lcb) + 0.0085)
    for ci, (rid, name, tag, status) in enumerate(cols):
        rep = reps[rid]
        x0 = left + ci * (cw + gap_x)
        # column header
        fig.text(x0, top + 0.06, name, ha="left", va="bottom", color=th["text"], fontsize=18, fontweight="bold")
        fig.text(x0, top + 0.022, f"{rid} · {tag}", ha="left", va="bottom", fontsize=15,
                 color=th[status] if status else th["muted"], fontweight="bold" if status else "normal")
        # --- row 1: AI bill per race
        ax = fig.add_axes((x0, bot + rh + gap_y, cw, rh))
        style_ax(ax, th)
        means = []
        for i, arm in enumerate(ARMS):
            st = rep.arms[arm].stats["llm_cost_usd"]
            means.append(st.mean)
            ax.bar(i, st.mean, width=0.58, color=th["arm"][arm], zorder=2)
            seeds = per_seed(rep, arm, "llm_cost_usd")
            ax.scatter([i + d for d in (-0.12, 0, 0.12)], seeds, s=46, color=th["surface"],
                       edgecolor=th["text"], linewidth=1.4, zorder=4)
            ax.text(i, max(seeds) + bill_max * 0.035, f"${st.mean:.2f}", ha="center", va="bottom",
                    color=th["text"], fontsize=20, fontweight="bold")
        ratio = means[0] / means[1]
        ax.text(0.98, 0.95, f"{ratio:.1f}× less", transform=ax.transAxes, ha="right", va="top",
                color=th["text"], fontsize=17, fontweight="bold")
        ax.set_ylim(0, bill_max)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([])
        ax.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}")
        if ci == 0:
            ax.set_ylabel("AI bill per race", color=th["text"], fontsize=17, fontweight="bold")
        # --- row 2: model quality
        ax2 = fig.add_axes((x0, bot, cw, rh))
        style_ax(ax2, th)
        qm = []
        for i, arm in enumerate(ARMS):
            st = rep.arms[arm].stats["best_quality"]
            qm.append(st.mean)
            seeds = per_seed(rep, arm, "best_quality")
            ax2.scatter([i + d for d in (-0.12, 0, 0.12)], seeds, s=70, color=th["arm"][arm],
                        edgecolor=th["surface"], linewidth=1.5, zorder=4)
            ax2.plot([i - 0.26, i + 0.26], [st.mean, st.mean], color=th["text"], lw=2.5, zorder=3,
                     solid_capstyle="butt")
            digits = 1 if rep.target == "rl" else 4
            ax2.text(i + 0.30, st.mean, num(st.mean, digits), ha="left", va="center", color=th["text"],
                     fontsize=17, fontweight="bold")
        if rep.target == "rl":
            lo = min(min(per_seed(rep, a, "best_quality")) for a in ARMS) - 0.8
            hi = max(max(per_seed(rep, a, "best_quality")) for a in ARMS) + 1.6
            ax2.set_ylim(lo, hi)
            verdict = "A tie within noise"
            unit = "game score on unseen levels"
        else:
            ax2.set_ylim(*lcb_lim)
            d = qm[0] - qm[1]
            verdict = (f"Standard ahead by {d:.4f} (~1% more errors)" if rid == "R82"
                       else f"Standard ahead by {d:.4f} (7% more errors)")
            unit = "accuracy on held-out data"
            ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.3f}")
        ax2.text(0.02, 0.97, verdict, transform=ax2.transAxes, ha="left", va="top", color=th["text"],
                 fontsize=14.5, fontweight="bold")
        ax2.text(0.02, 0.86, f"{unit} · higher is better · axis zoomed", transform=ax2.transAxes, ha="left",
                 va="top", color=th["muted"], fontsize=12)
        ax2.set_xlim(-0.6, 1.6)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([TICK[a] for a in ARMS], color=th["text"], fontsize=15)
        if ci == 0:
            ax2.set_ylabel("Model quality", color=th["text"], fontsize=17, fontweight="bold")
    handles = [Patch(color=th["arm"][a], label=SHORT[a]) for a in ARMS] + [
        Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=th["surface"], markeredgecolor=th["text"],
               label="one run")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.855), ncol=3, frameon=False,
               fontsize=14, labelcolor=th["text"], handlelength=1.2)
    o = {r: reps[r].arms["optuna"].stats["best_quality"].mean for r in reps}
    footer(fig, th, [
        "n = 3 runs per side in each. R78 races 4ca4ed27, d4913e95, 145c2a2d · R82 c1f4b1d9, 9d0cfd77, b3922d0e · "
        "R75 613e2537, 6afbbe8a, c72b3af1. Bar or line = mean, dots = single runs.",
        "AI bill at list price. R82's 30 s pace was chosen after R75 failed, and frozen before it ran. Tabular machine time "
        "simulated; game-playing on real cloud workers.",
        f"No-AI tuning tool (Optuna), in every race, not shown: model quality R78 {num(o['R78'], 1)}, R82 {o['R82']:.4f}, "
        f"R75 {o['R75']:.4f} (worst in all three); it accepted fakes as best (R78 53, R82 15, R75 19 trap runs).",
    ])
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- chart 2


def waiting_series(intervals: list[dict[str, Any]], t0: float, t1: float, step: float) -> tuple[list[float], list[float]]:
    edges = []
    t = t0
    while t < t1 - 1e-9:
        edges.append((t, min(t + step, t1)))
        t += step
    xs, ys = [], []
    for a, b in edges:
        idle = paid = 0.0
        for iv in intervals:
            s, e = float(iv["start"]), float(iv["end"])
            ov = min(e, b) - max(s, a)
            if ov <= 0:
                continue
            paid += ov
            if iv["state"] != "busy":
                idle += ov
        xs.append(((a + b) / 2 - t0) / 60.0)
        ys.append(100.0 * idle / paid if paid else 0.0)
    return xs, ys


def chart_waiting(rep: ar.TargetReport, rows_by_race: dict[str, Any], th: dict[str, Any], path: Path) -> None:
    rid = next(r for r in rows_by_race if r.startswith(R78_ONE))
    rows = rows_by_race[rid]
    race = rows["races"][0]
    t0, t1 = float(race["started_at"]), float(race["ended_at"])
    arm_ids = {a["kind"]: a["id"] for a in rows["arms"]}
    run = {r.arm: r for r in rep.runs if r.race_id == rid}
    fig = new_fig(th)
    header(fig, th, "GPUs waiting: the standard assistant leaves them idle while it thinks",
           "Share of the 30 paid machines per side with nothing to run, in 30-second windows, over one 15-minute "
           "race (game-playing AI).")
    ax = fig.add_axes((0.075, 0.2, 0.70, 0.62))
    style_ax(ax, th)
    for arm in ARMS:
        ivs = [i for i in rows["gpu_intervals"] if i["arm_id"] == arm_ids[arm]]
        xs, ys = waiting_series(ivs, t0, t1, 30.0)
        ax.fill_between(xs, ys, color=th["arm"][arm], alpha=0.10, lw=0, zorder=1)
        ax.plot(xs, ys, color=th["arm"][arm], lw=2.5, zorder=3)
        avg = run[arm].gpu_idle_pct
        y_lab = max(avg, 9.0)
        ax.text(15.3, y_lab, f"{SHORT[arm].split(' (')[0]}\n{avg:.1f}% waiting on average",
                ha="left", va="center", color=th["text"], fontsize=16, fontweight="bold", clip_on=False)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 16, 3))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} min")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylabel("GPUs waiting", color=th["text"], fontsize=17, fontweight="bold")
    ax.tick_params(axis="x", pad=10)
    handles = [Line2D([], [], color=th["arm"][a], lw=3, label=SHORT[a]) for a in ARMS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.86), ncol=2, frameon=False,
               fontsize=14, labelcolor=th["text"])
    o = rep.arms["opus"].stats["gpu_idle_pct"].mean
    b = rep.arms["opus_bm"].stats["gpu_idle_pct"].mean
    po = " / ".join(f"{v:.1f}" for v in per_seed(rep, "opus", "gpu_idle_pct"))
    pb = " / ".join(f"{v:.1f}" for v in per_seed(rep, "opus_bm", "gpu_idle_pct"))
    footer(fig, th, [
        f"R78 race {rid[:8]} (seed 1), n = 1 race shown of 3; chosen as the middle run on the gap. "
        f"All 3 runs: standard {po}%, Busy Machines {pb}% (means {o:.1f}% vs {b:.1f}%).",
        "Waiting = paid machine time with no experiment running, from the race's machine log (gpu_intervals). "
        "Machines here are 30 cloud CPU workers per side, 15-second runs.",
        f"Busy Machines fills the gaps with a no-AI sampler while the AI thinks. No-AI tuning tool (Optuna), not shown: "
        f"{rep.arms['optuna'].stats['gpu_idle_pct'].mean:.1f}% waiting, worse model (R78).",
    ])
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- chart 3


def best_so_far(exps: list[dict[str, Any]], t0: float, t1: float) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    best = None
    for e in sorted((e for e in exps if is_counted(e)), key=lambda e: float(e["ended_at"])):
        q = quality(e)
        if best is None or q > best:
            best = q
            xs.append((float(e["ended_at"]) - t0) / 60.0)
            ys.append(best)
    if xs:
        xs.append((t1 - t0) / 60.0)
        ys.append(ys[-1])
    return xs, ys


def chart_quality_time(rep: ar.TargetReport, rows_by_race: dict[str, Any], th: dict[str, Any], path: Path) -> None:
    rid = next(r for r in rows_by_race if r.startswith(R82_ONE))
    rows = rows_by_race[rid]
    race = rows["races"][0]
    t0, t1 = float(race["started_at"]), float(race["ended_at"])
    arm_ids = {a["kind"]: a["id"] for a in rows["arms"]}
    run = {r.arm: r for r in rep.runs if r.race_id == rid}
    fig = new_fig(th)
    header(fig, th, "Model quality over one race: Busy Machines keeps pace, not ahead",
           "Best checked model so far, per side, over one 15-minute tabular tuning race (AI asked every 30 s).")
    ax = fig.add_axes((0.075, 0.2, 0.70, 0.62))
    style_ax(ax, th, xgrid=False)
    arms = ("opus", "opus_bm", "optuna")
    finals = {}
    for arm in arms:
        exps = [e for e in rows["experiments"] if e["arm_id"] == arm_ids[arm]]
        xs, ys = best_so_far(exps, t0, t1)
        ax.step(xs, ys, where="post", color=th["arm"][arm], lw=2.6, zorder=3)
        ax.scatter(xs[:-1], ys[:-1], s=36, color=th["arm"][arm], edgecolor=th["surface"], linewidth=1.2, zorder=4)
        finals[arm] = ys[-1]
        if abs(ys[-1] - (run[arm].best_quality or 0)) > 1e-9:
            raise SystemExit(f"chart 3 {arm}: final {ys[-1]} != analysed best {run[arm].best_quality}")
    lo = 0.80
    ax.set_ylim(lo, 0.895)
    ax.set_xlim(0, 15)
    # direct labels, nudged apart so they never collide
    order = sorted(arms, key=lambda a: -finals[a])
    gap = 0.0055
    ys_lab: list[float] = []
    for arm in order:
        y = finals[arm] if not ys_lab else min(finals[arm], ys_lab[-1] - gap)
        ys_lab.append(y)
    shift = (sum(finals[a] for a in order) - sum(ys_lab)) / len(order)
    shift = min(shift, min(finals[a] - y for a, y in zip(order, ys_lab)) + gap)  # keep the gaps
    for arm, y in zip(order, ys_lab):
        y += shift
        ax.text(15.25, y, f"{SHORT[arm].split(' (')[0]}  {finals[arm]:.4f}", ha="left", va="center",
                color=th["text"], fontsize=16, fontweight="bold", clip_on=False)
    ax.set_xticks(range(0, 16, 3))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} min")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax.tick_params(axis="x", pad=10)
    ax.set_ylabel("Model quality", color=th["text"], fontsize=17, fontweight="bold")
    ax.text(0.01, 0.97, f"accuracy on held-out data · higher is better · axis starts at {lo:.2f}",
            transform=ax.transAxes, va="top", color=th["muted"], fontsize=12.5)
    handles = [Line2D([], [], color=th["arm"][a], lw=3, label=SHORT[a]) for a in arms]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.86), ncol=3, frameon=False,
               fontsize=14, labelcolor=th["text"])
    gaps = [b - o for o, b in zip(per_seed(rep, "opus", "best_quality"), per_seed(rep, "opus_bm", "best_quality"))]
    footer(fig, th, [
        f"R82 race {rid[:8]} (seed 1), n = 1 race shown of 3; chosen as the middle run. Busy Machines minus standard, "
        f"all 3 runs: {', '.join(num(g, 4) for g in gaps)} (standard ahead in every run).",
        "Only checked runs count: shortcut runs that test on seen data never count for any side. "
        "Tabular machine time simulated. This 30 s setting was chosen after R75 (90 s) failed.",
        f"3-run means: standard {rep.arms['opus'].stats['best_quality'].mean:.4f}, Busy Machines "
        f"{rep.arms['opus_bm'].stats['best_quality'].mean:.4f}, no-AI tool {rep.arms['optuna'].stats['best_quality'].mean:.4f}.",
    ])
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- chart 4

R81_EXPECT = {  # docs/results_log.md R81 (ground-truth rows): caught, approved, real accepted, no answer
    "stock_sapling_published": (0, 0, 0, 152),
    "sapling_research_v2_a01": (8, 0, 0, 105),
    "sapling_research_v2_a05": (57, 0, 0, 24),
    "opus_5_5_recorded": (109, 0, 0, 0),
}


def chart_sapling(th: dict[str, Any], path: Path) -> None:
    d = json.loads(R81_JSON.read_text(encoding="utf-8"))["results"]
    who = [
        ("stock_sapling_published", "Published Banyan Sapling", th["grey"], None),
        ("sapling_research_v2_a01", "Our fine-tune, strict", th["arm"]["opus_bm"], "//"),
        ("sapling_research_v2_a05", "Our fine-tune, looser", th["arm"]["opus_bm"], None),
        ("opus_5_5_recorded", "Opus 5.5", th["arm"]["opus"], None),
    ]
    vals = {}
    for k, *_ in who:
        g = d[k]["ground_truth"]
        if g["n_ground_truth_rows"] != 152 or g["n_fake"] != 109 or g["n_real"] != 7:
            raise SystemExit(f"R81 {k}: unexpected row counts")
        v = (g["fakes_caught (violation)"], g["fakes_approved (supported)"], g["real_gains_accepted (supported)"],
             g["unsure (no calibrated answer)"])
        if v != R81_EXPECT[k]:
            raise SystemExit(f"R81 {k}: {v} != logged {R81_EXPECT[k]}")
        vals[k] = v
    panels = [
        (0, 109, "Fakes caught", "of 109 fake improvements · higher is better"),
        (1, 109, "Fakes approved", "of 109 fake improvements · lower is better"),
        (3, 152, "No answer", "of 152 claims · lower is better"),
        (2, 7, "Real improvements confirmed", "of 7 real gains · higher is better"),
    ]
    fig = new_fig(th)
    header(fig, th, "Catching fake improvements in our own race results",
           "226 real claims from today's races, 152 with known truth. Each checker says: real, fake, or no answer.")
    left, right, top, bot = 0.03, 0.985, 0.69, 0.175
    lab_w = 0.16
    gap_x, gap_y = 0.05, 0.155
    pw = (right - left - lab_w * 2 - gap_x) / 2
    ph = (top - bot - gap_y) / 2
    ys = list(range(len(who)))[::-1]
    for pi, (idx, denom, title, how) in enumerate(panels):
        r, c = divmod(pi, 2)
        x0 = left + lab_w + c * (pw + gap_x + lab_w)
        y0 = bot + (1 - r) * (ph + gap_y)
        ax = fig.add_axes((x0, y0, pw, ph))
        style_ax(ax, th, ygrid=False, xgrid=True)
        for y, (k, name, col, hatch) in zip(ys, who):
            v = vals[k][idx]
            if k == "opus_5_5_recorded" and idx == 3:
                ax.text(denom * 0.01, y, "always answers", va="center", ha="left", color=th["muted"], fontsize=15)
                continue
            if v:
                ax.barh(y, v, height=0.62, color=col if not hatch else th["surface"], edgecolor=col,
                        hatch=hatch, linewidth=2 if hatch else 0, zorder=2)
            ax.text(v + denom * 0.015, y, f"{v}", va="center", ha="left", color=th["text"], fontsize=17,
                    fontweight="bold")
        ax.set_xlim(0, denom * 1.12)
        ax.set_ylim(-0.6, len(who) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([w[1] for w in who], color=th["text"], fontsize=14.5)
        ax.axvline(0, color=th["muted"], lw=1, zorder=1)
        ax.text(0, 1.17, title, transform=ax.transAxes, ha="left", va="bottom", color=th["text"], fontsize=19,
                fontweight="bold")
        ax.text(0, 1.06, how, transform=ax.transAxes, ha="left", va="bottom", color=th["muted"], fontsize=13)
    footer(fig, th, [
        "R81: n = 226 judge-format claims from our races; truth known for 152 (109 fake, 36 noise, 7 real). Opus graded "
        "on those 152 only (77 other labels came from Opus). Only 7 real gains: counts, not rates.",
        "Our fine-tune = sapling-research v2: Banyan Sapling's verification head retrained, backbone frozen; a community "
        "fine-tune, not an official MubitLabs model. Strict / looser = calibration α 0.01 / 0.05.",
        "Opus is still the better judge. Time per check ~0.1 s (Sapling, ours) vs ~1.7 s (Opus). "
        "A later version trained on race claims (v4, R83) failed its gate and is not shown.",
    ])
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(REPO_ROOT / "data" / "bench" / "bench.db"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "charts" / "final"))
    args = ap.parse_args(argv)
    db, out = Path(args.db), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reps, rows = {}, {}
    for rid in RACES:
        reps[rid], rows[rid] = load_report(db, rid)
        check_logged(rid, reps[rid])
    written = []
    for name in ("light", "dark"):
        th = theme(name)
        jobs = [
            ("1-headline-two-tasks", lambda p: chart_headline(reps, th, p)),
            ("2-gpus-waiting-r78", lambda p: chart_waiting(reps["R78"], rows["R78"], th, p)),
            ("3-quality-over-time-r82", lambda p: chart_quality_time(reps["R82"], rows["R82"], th, p)),
            ("4-sapling-real-claims", lambda p: chart_sapling(th, p)),
        ]
        for stem, fn in jobs:
            p = out / f"{stem}-{name}.png"
            fn(p)
            written.append(p)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
