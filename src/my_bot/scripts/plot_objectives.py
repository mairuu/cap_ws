#!/usr/bin/env python3
"""Draw the report's objective figures from the session files. No ROS needed.

WHAT THIS IS FOR. Chapter 5 of the report has three figure slots that the
measurement scripts fill with numbers but not with pictures:

    slam      fig:slam-error     slam_error_result.png      objective 1, <= 10 cm
    object    fig:object-error   object_position_error.png  objective 3, <= 50 cm
    resource  fig:resource       resource_usage.png         objective 4, <= 80 %

Each one reads the file its measurement script already writes, so the figure
and the table's number come from the same bytes:

    slam      ~/maps/slam_accuracy.jsonl      slam_accuracy_check.py mark ...
    object    ~/maps/tape_session.jsonl       landmark_tape_measure.py ... --pass-label
    resource  ~/maps/resource_session.jsonl   resource_report.py --label ...
              + the per-sample CSV each window's row points at

    python3 plot_objectives.py all --out ~/cap_ref/figures
    python3 plot_objectives.py slam --slam-session ~/maps/slam_accuracy.jsonl
    python3 plot_objectives.py resource --window "full stack"   # label substring

Every figure prints the numbers it was drawn from, so the caption and the
evaluation table can be copied from the same terminal.

LANGUAGE. Labels are Thai by default (font Loma). Matplotlib here has no
complex-text shaping, so a tone mark over an upper vowel sits a little off;
`--lang en` gives English labels if that is not acceptable in print.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Categorical slots, fixed order (dataviz reference palette, light mode).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LIMIT = "#52514e"          # criterion lines are ink, not a status colour

T = {
    "th": {
        "truth": "ตำแหน่งจริง (วัดด้วยตลับเมตร)",
        "reported": "ตำแหน่งที่ SLAM รายงาน",
        "x": "x ในกรอบพิกัดแผนที่ (เมตร)",
        "y": "y ในกรอบพิกัดแผนที่ (เมตร)",
        "err_cm": "ความคลาดเคลื่อน (เซนติเมตร)",
        "mark": "จุดอ้างอิง",
        "raw": "ค่าดิบ แต่ละรอบ",
        "aligned": "หลังชดเชยการหมุนของแกน",
        "limit_slam": "เกณฑ์ 10 ซม.",
        "slam_a": "(ก) ตำแหน่งจุดอ้างอิงบนแผนที่",
        "slam_b": "(ข) ความคลาดเคลื่อนต่อจุด",
        "obj_a": "(ก) ตำแหน่งวัตถุที่บันทึกได้ในแต่ละมุมมอง",
        "obj_b": "(ข) ความคลาดเคลื่อนต่อมุมมอง",
        "obj_truth": "ตำแหน่งจริงของวัตถุ",
        "limit_obj": "เกณฑ์ 50 ซม.",
        "pass": "มุมมอง",
        "time": "เวลา (วินาที)",
        "pct": "อัตราการใช้งาน (%)",
        "cpu": "CPU เฉลี่ย 6 คอร์",
        "cpu_raw": "CPU ต่อวินาที",
        "gpu": "GPU (GR3D)",
        "ram": "RAM",
        "limit_res": "เกณฑ์ 80 %",
        "res_a": "(ก) ตลอดรอบการทำงาน",
        "res_b": "(ข) CPU เฉลี่ยแต่ละสถานการณ์",
        "cpu_mean": "CPU เฉลี่ย (%)",
    },
    "en": {
        "truth": "true position (tape)",
        "reported": "SLAM-reported position",
        "x": "x, map frame (m)",
        "y": "y, map frame (m)",
        "err_cm": "error (cm)",
        "mark": "reference mark",
        "raw": "raw, per lap",
        "aligned": "after axis rotation removed",
        "limit_slam": "10 cm criterion",
        "slam_a": "(a) reference marks on the map",
        "slam_b": "(b) error per mark",
        "obj_a": "(a) published position per viewpoint",
        "obj_b": "(b) error per viewpoint",
        "obj_truth": "true object position",
        "limit_obj": "50 cm criterion",
        "pass": "viewpoint",
        "time": "time (s)",
        "pct": "utilisation (%)",
        "cpu": "CPU, 6-core mean",
        "cpu_raw": "CPU, per sample",
        "gpu": "GPU (GR3D)",
        "ram": "RAM",
        "limit_res": "80 % criterion",
        "res_a": "(a) across the run",
        "res_b": "(b) mean CPU per scenario",
        "cpu_mean": "mean CPU (%)",
    },
}


def style(lang):
    fam = ["DejaVu Sans"]
    if lang == "th":
        names = {f.name for f in font_manager.fontManager.ttflist}
        thai = next((n for n in ("Loma", "Garuda", "Norasi", "Kinnari")
                     if n in names), None)
        if thai is None:
            print("!! no Thai font found (fonts-tlwg-*); falling back to --lang en")
            return "en"
        fam = [thai, "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": fam, "font.size": 10, "axes.unicode_minus": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "figure.facecolor": "white",
    })
    return lang


def load_jsonl(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def limit_line(ax, y, text, horizontal=True):
    ax.axhline(y, color=LIMIT, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points",
                ha="right", va="bottom", color=INK2, fontsize=9)


def save(fig, out, name):
    os.makedirs(os.path.expanduser(out), exist_ok=True)
    path = os.path.join(os.path.expanduser(out), name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------- objective 1
def fit_rotation(pairs):
    """Rotation about the origin taking truth onto reported (as in
    slam_accuracy_check.py --summary), and each mark's residual after it."""
    num = sum(t[0] * r[1] - t[1] * r[0] for t, r in pairs)
    den = sum(t[0] * r[0] + t[1] * r[1] for t, r in pairs)
    th = math.atan2(num, den)
    c, s = math.cos(th), math.sin(th)
    return th, [math.hypot(c * t[0] - s * t[1] - r[0], s * t[0] + c * t[1] - r[1])
                for t, r in pairs]


def plot_slam(args, L):
    rows = load_jsonl(args.slam_session)
    rows = [r for r in rows or [] if "truth_x" in r]
    if not rows:
        print(f"slam: no visits with --truth in {args.slam_session}; skipped.")
        return 1
    marks = {}
    for r in rows:
        marks.setdefault(r["mark"], []).append(r)
    names = list(marks)
    laps = max(len(v) for v in marks.values())

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) plan view
    for n, vs in marks.items():
        tx, ty = vs[0]["truth_x"], vs[0]["truth_y"]
        a.add_patch(plt.Circle((tx, ty), args.slam_max, fill=False, ec=INK2,
                               lw=0.8, ls=(0, (3, 2))))
        for v in vs:
            a.plot([tx, v["x"]], [ty, v["y"]], color=GRID, lw=1, zorder=1)
        a.annotate(n, (tx, ty), xytext=(6, 6), textcoords="offset points",
                   fontsize=9, color=INK)
    a.scatter([r["truth_x"] for r in rows], [r["truth_y"] for r in rows],
              marker="x", s=60, color=INK, lw=1.6, label=L["truth"], zorder=3)
    a.scatter([r["x"] for r in rows], [r["y"] for r in rows], s=36, color=BLUE,
              ec="white", lw=1.5, label=L["reported"], zorder=4)
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel(L["x"]); a.set_ylabel(L["y"]); a.set_title(L["slam_a"], loc="left")
    a.legend(loc="best", fontsize=9)

    # (b) error per mark: one dot per visit, a diamond for the aligned residual
    for i, n in enumerate(names):
        errs = [v["error_m"] * 100 for v in marks[n]]
        jit = [(k - (len(errs) - 1) / 2) * 0.08 for k in range(len(errs))]
        b.scatter([i + j for j in jit], errs, s=40, color=BLUE, ec="white",
                  lw=1.5, zorder=3, label=L["raw"] if i == 0 else None)
    aligned = None
    pairs = [((marks[n][0]["truth_x"], marks[n][0]["truth_y"]),
              (statistics.fmean(v["x"] for v in marks[n]),
               statistics.fmean(v["y"] for v in marks[n]))) for n in names]
    if len(pairs) >= 3:
        th, aligned = fit_rotation(pairs)
        b.scatter(range(len(names)), [e * 100 for e in aligned], marker="D",
                  s=40, color=ORANGE, ec="white", lw=1.5, zorder=4,
                  label=f"{L['aligned']} ({math.degrees(th):+.1f}°)")
    limit_line(b, args.slam_max * 100, L["limit_slam"])
    b.set_xticks(range(len(names)), names)
    b.set_xlabel(L["mark"]); b.set_ylabel(L["err_cm"])
    b.set_ylim(0, max(args.slam_max * 100 * 1.4,
                      max(r["error_m"] for r in rows) * 100 * 1.15))
    b.set_title(L["slam_b"], loc="left")
    b.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    worst = max(r["error_m"] for r in rows)
    reps = [max(math.hypot(v["x"] - statistics.fmean(w["x"] for w in vs),
                           v["y"] - statistics.fmean(w["y"] for w in vs))
                for v in vs) for vs in marks.values() if len(vs) > 1]
    print(f"slam: {len(rows)} visits, {len(names)} marks, up to {laps} lap(s)")
    print(f"      worst raw error      {worst*100:.1f} cm  "
          f"({'PASS' if worst <= args.slam_max else 'FAIL'} vs {args.slam_max*100:.0f})")
    print(f"      mean raw error       {statistics.fmean(r['error_m'] for r in rows)*100:.1f} cm")
    if aligned:
        print(f"      worst aligned error  {max(aligned)*100:.1f} cm")
    if reps:
        print(f"      worst repeatability  {max(reps)*100:.1f} cm")
    else:
        print("      repeatability        n/a -- one visit per mark; drive more laps")
    save(fig, args.out, "slam_error_result.png")
    return 0


# ---------------------------------------------------------------- objective 3
def plot_object(args, L):
    rows = load_jsonl(args.object_session)
    rows = [r for r in rows or [] if r.get("truth_x") is not None]
    if not rows:
        print(f"object: no passes in {args.object_session}; skipped.")
        return 1
    tx, ty = rows[-1]["truth_x"], rows[-1]["truth_y"]
    rows = [r for r in rows if (r["truth_x"], r["truth_y"]) == (tx, ty)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    a.add_patch(plt.Circle((tx, ty), args.object_max, fill=False, ec=INK2,
                           lw=1, ls=(0, (5, 3))))
    a.annotate(L["limit_obj"], (tx, ty + args.object_max), xytext=(0, 3),
               textcoords="offset points", ha="center", color=INK2, fontsize=9)
    for i, r in enumerate(rows):
        c = SERIES[i % len(SERIES)]
        pts = r.get("samples") or []
        if pts:
            a.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, color=c,
                      alpha=0.25, lw=0, zorder=2)
        a.scatter([r["x"]], [r["y"]], s=48, color=c, ec="white", lw=1.5,
                  zorder=4, label=f"{r['pass_label']}  {r['error']*100:.0f} cm")
    a.scatter([tx], [ty], marker="*", s=180, color=INK, zorder=5,
              label=L["obj_truth"])
    span = max(args.object_max * 1.3,
               max(math.hypot(r["x"] - tx, r["y"] - ty) for r in rows) * 1.3)
    a.set_xlim(tx - span, tx + span); a.set_ylim(ty - span, ty + span)
    a.set_aspect("equal")
    a.set_xlabel(L["x"]); a.set_ylabel(L["y"]); a.set_title(L["obj_a"], loc="left")
    a.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.0, 1.0))

    labels = [r["pass_label"] for r in rows]
    errs = [r["error"] * 100 for r in rows]
    bars = b.bar(range(len(rows)), errs, width=0.6, ec="white", lw=2,
                 color=[SERIES[i % len(SERIES)] for i in range(len(rows))])
    for bar, e in zip(bars, errs):
        b.annotate(f"{e:.1f}", (bar.get_x() + bar.get_width() / 2, e),
                   xytext=(0, 3), textcoords="offset points", ha="center",
                   fontsize=9, color=INK)
    limit_line(b, args.object_max * 100, L["limit_obj"])
    b.set_xticks(range(len(rows)), labels)
    b.set_ylim(0, max(args.object_max * 100 * 1.25, max(errs) * 1.2))
    b.set_xlabel(L["pass"]); b.set_ylabel(L["err_cm"])
    b.set_title(L["obj_b"], loc="left")
    b.grid(axis="x", visible=False)
    fig.tight_layout()

    cx = statistics.fmean(r["x"] for r in rows)
    cy = statistics.fmean(r["y"] for r in rows)
    across = max(math.hypot(r["x"] - cx, r["y"] - cy) for r in rows) * 2
    print(f"object: {len(rows)} pass(es) at truth ({tx:.2f}, {ty:.2f})")
    print(f"      mean error   {statistics.fmean(errs):.1f} cm, worst {max(errs):.1f} cm "
          f"({'PASS' if max(errs) <= args.object_max*100 else 'FAIL'} vs "
          f"{args.object_max*100:.0f})")
    print(f"      across-pass spread {across*100:.1f} cm; "
          f"duplicates (worst) {max(r['duplicates'] for r in rows)}")
    save(fig, args.out, "object_position_error.png")
    return 0


# ---------------------------------------------------------------- objective 4
def read_samples(path):
    with open(os.path.expanduser(path)) as fh:
        return list(csv.DictReader(fh))


def rolling(xs, n):
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= n:
            acc -= xs[i - n]
        out.append(acc / min(i + 1, n))
    return out


def plot_resource(args, L):
    rows = load_jsonl(args.resource_session) or []
    if not rows:
        print(f"resource: nothing in {args.resource_session}; skipped.")
        return 1
    with_csv = [r for r in rows if r.get("samples_csv")
                and os.path.exists(os.path.expanduser(r["samples_csv"]))]
    if args.window:
        with_csv = [r for r in with_csv if args.window in (r.get("label") or "")]
    series = with_csv[-1] if with_csv else None
    many = len(rows) > 1

    fig, axes = plt.subplots(1, 2 if many else 1, figsize=(10 if many else 7, 4.2),
                             gridspec_kw={"width_ratios": [1.6, 1]} if many else None)
    a = axes[0] if many else axes
    if series:
        s = read_samples(series["samples_csv"])
        t = [float(r["t_s"]) for r in s]
        cpu = [float(r["cpu_mean"]) for r in s]
        a.plot(t, cpu, color=BLUE, lw=0.8, alpha=0.3, label=L["cpu_raw"])
        a.plot(t, rolling(cpu, args.smooth), color=BLUE, lw=2,
               label=f"{L['cpu']} ({args.smooth} s)")
        gpu = [(float(r["t_s"]), float(r["gpu"])) for r in s if r["gpu"] != ""]
        if gpu:
            a.plot([g[0] for g in gpu], rolling([g[1] for g in gpu], args.smooth),
                   color=ORANGE, lw=2, label=L["gpu"])
        ram = [(float(r["t_s"]), 100 * float(r["ram_mb"]) / float(r["ram_total_mb"]))
               for r in s if r["ram_mb"] != ""]
        if ram:
            a.plot([g[0] for g in ram], [g[1] for g in ram], color=AQUA, lw=2,
                   label=L["ram"])
        mean = statistics.fmean(cpu)
        a.axhline(mean, color=BLUE, lw=1, ls=":", zorder=1)
        a.annotate(f"{mean:.1f} %", xy=(0, mean), xycoords=("axes fraction", "data"),
                   xytext=(3, 3), textcoords="offset points", color=INK, fontsize=9)
        limit_line(a, args.cpu_max, L["limit_res"])
        a.set_xlim(0, t[-1]); a.set_ylim(0, 105)
        a.set_xlabel(L["time"]); a.set_ylabel(L["pct"])
        a.set_title(f"{L['res_a']}: {series.get('label') or ''}", loc="left",
                    fontsize=10)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8)
    else:
        a.text(0.5, 0.5, "no per-sample CSV yet\n(re-run resource_report.py)",
               ha="center", va="center", transform=a.transAxes, color=INK2)
        a.set_axis_off()
        print("resource: no window has a per-sample CSV -- the time series needs a "
              "window recorded by the current resource_report.py.")

    if many:
        b = axes[1]
        labels = [r.get("label") or "-" for r in rows]
        vals = [r["cpu_mean"] for r in rows]
        bars = b.barh(range(len(rows)), vals, height=0.6, color=BLUE, ec="white", lw=2)
        for bar, v in zip(bars, vals):
            b.annotate(f"{v:.1f}", (v, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0), textcoords="offset points", va="center",
                       fontsize=9, color=INK)
        b.axvline(args.cpu_max, color=LIMIT, lw=1.2, ls=(0, (5, 3)))
        b.annotate(L["limit_res"], xy=(args.cpu_max, 1.0),
                   xycoords=("data", "axes fraction"), xytext=(-3, -2),
                   textcoords="offset points", ha="right", va="top",
                   color=INK2, fontsize=9)
        b.set_yticks(range(len(rows)), [l[:28] for l in labels], fontsize=8)
        b.invert_yaxis()
        b.set_xlim(0, 100); b.set_xlabel(L["cpu_mean"])
        b.set_title(L["res_b"], loc="left")
        b.grid(axis="y", visible=False)
    fig.tight_layout()

    print(f"resource: {len(rows)} window(s)")
    for r in rows:
        print(f"      {(r.get('label') or '-')[:40]:<40} cpu {r['cpu_mean']:5.1f} %  "
              f"gpu {(r.get('gpu_mean') or 0):5.1f} %  RAM peak {r.get('ram_peak_mb')} MB  "
              f"tj max {r.get('tj_max')} C  "
              f"({'PASS' if r['cpu_mean'] <= args.cpu_max else 'FAIL'})")
    save(fig, args.out, "resource_usage.png")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["slam", "object", "resource", "all"])
    ap.add_argument("--out", default="~/cap_ref/figures")
    ap.add_argument("--lang", choices=["th", "en"], default="th")
    ap.add_argument("--slam-session", default="~/maps/slam_accuracy.jsonl")
    ap.add_argument("--object-session", default="~/maps/tape_session.jsonl")
    ap.add_argument("--resource-session", default="~/maps/resource_session.jsonl")
    ap.add_argument("--window", help="resource: plot the latest window whose "
                                     "label contains this text")
    ap.add_argument("--smooth", type=int, default=15,
                    help="resource: rolling-mean window, samples (default 15)")
    ap.add_argument("--slam-max", type=float, default=0.10)
    ap.add_argument("--object-max", type=float, default=0.50)
    ap.add_argument("--cpu-max", type=float, default=80.0)
    args = ap.parse_args()

    L = T[style(args.lang)]
    todo = ["slam", "object", "resource"] if args.which == "all" else [args.which]
    fn = {"slam": plot_slam, "object": plot_object, "resource": plot_resource}
    rc = [fn[w](args, L) for w in todo]
    return 1 if any(rc) else 0


if __name__ == "__main__":
    sys.exit(main())
