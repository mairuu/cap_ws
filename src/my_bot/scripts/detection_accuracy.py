#!/usr/bin/env python3
"""Score the deployed detector against hand-labelled frames. Objective 2.

WHAT THIS IS FOR. The report's objective 2 sets a number -- detection accuracy
not below 80 %, averaged over the target classes -- and `records/objective-tests.md`
records why no number exists yet: `yolo26s` came pretrained on COCO, so there is
no held-out split of ours to score. The answer that document recommends is to
build an in-situ test set from a bag and score THE DEPLOYED DETECTOR IN THE ROOM
IT IS DEPLOYED IN. This script is that measurement, end to end.

Four steps, four subcommands:

    extract   pull every Nth frame out of a bag, plus the /detections that were
              published for those same frames -- the ONNX predictions that
              actually ran, not a re-run
    predict   run another model (e.g. the .pt) over the SAME frames, so the
              export decision can be scored rather than asserted
    score     match predictions to hand labels at IoU >= 0.5, report per-class
              precision / recall / F1 and the macro average
    compare   draw the two-backend figure from the session file

TWO TIERS, ON PURPOSE. A class needs enough instances before a percentage means
anything. Classes at or above --min-instances get scored and enter the macro
average; everything else is reported in a coverage table with its instance count
and no percentage. Set the class list and the threshold BEFORE labelling, and
say so in the report -- choosing them after seeing the scores is the thing an
examiner is entitled to ask about.

WHAT COUNTS AS THE CRITERION. Precision alone rises with a higher confidence
threshold and recall alone rises with a lower one, so neither is checked against
80 % on its own. The macro F1 is the criterion; P and R are reported beside it
so the trade-off is visible.

LABEL FORMAT. YOLO text, one .txt per frame next to a classes.txt -- what
labelImg, CVAT and Label Studio all export:

    <class_index> <cx> <cy> <w> <h>        all normalised 0..1

A frame with no target object still needs an empty .txt file, or it is treated
as unlabelled and skipped. Labels must be EXHAUSTIVE for the target classes: a
real chair you did not label turns a correct detection into a false positive.

TYPICAL RUN

    # 1. ~120 frames spread across the bag, with the live ONNX predictions
    python3 detection_accuracy.py extract \\
        --bag ~/bags/2026-09-22-163401 --out ~/eval/insitu --every 23

    # 2. label ~/eval/insitu/frames/*.jpg into ~/eval/insitu/labels/

    # 3. the number for the report
    python3 detection_accuracy.py score --data ~/eval/insitu \\
        --pred ~/eval/insitu/pred_onnx.json --run-label onnx-fp16 \\
        --classes person chair backpack laptop --figure ~/eval/accuracy.png

    # 4. the same frames through the .pt, then the comparison figure
    ~/yolo/venv/bin/python detection_accuracy.py predict --data ~/eval/insitu \\
        --model ~/yolo/yolo26s.pt --out ~/eval/insitu/pred_pt.json
    python3 detection_accuracy.py score --data ~/eval/insitu \\
        --pred ~/eval/insitu/pred_pt.json --run-label pt-torch \\
        --classes person chair backpack laptop
    python3 detection_accuracy.py compare --session ~/maps/accuracy_session.jsonl \\
        --figure ~/eval/accuracy_backends.png

`extract` needs the ROS 2 environment (rosbag2_py, vision_msgs, cv2).
`predict` needs the ultralytics venv. `score` and `compare` need neither --
numpy and matplotlib only, so they run on a laptop.

NOT MEASURED BY THIS SCRIPT. Localisation quality beyond the IoU gate, and
anything about the classes the target list leaves out. Out-of-set detections
(the model calling a corridor a `bus`) are counted and printed, because the
closed 80-class set is a stated limitation of the project and this is the
cheapest evidence for it, but they are not scored.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

DEFAULT_SESSION = "~/maps/accuracy_session.jsonl"
DEFAULT_IMAGE_TOPIC = "/image/compressed"
DEFAULT_DET_TOPIC = "/detections"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def expand(p):
    return os.path.abspath(os.path.expanduser(p))


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def die(msg, code=2):
    print("!! " + msg, file=sys.stderr)
    raise SystemExit(code)


def iou(a, b):
    """a, b = (x1, y1, x2, y2) in pixels."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# vision_msgs compatibility -- the field names moved between releases
# ---------------------------------------------------------------------------

def det_box_xyxy(det):
    """Detection2D -> (x1, y1, x2, y2). Handles Pose2D.position (4.x) and the
    older flat centre."""
    c = det.bbox.center
    if hasattr(c, "position"):
        cx, cy = c.position.x, c.position.y
    else:
        cx, cy = c.x, c.y
    w, h = det.bbox.size_x, det.bbox.size_y
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def det_class_score(det):
    """Detection2D -> (class_id, score). 4.x nests them under .hypothesis."""
    if not det.results:
        return None, 0.0
    r = det.results[0]
    h = getattr(r, "hypothesis", r)
    cid = getattr(h, "class_id", None)
    if cid is None:
        cid = getattr(h, "id", "")
    return str(cid), float(getattr(h, "score", 0.0))


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def cmd_extract(args):
    try:
        import cv2
        import numpy as np
        from rclpy.serialization import deserialize_message
        from rosbag2_py import (ConverterOptions, SequentialReader,
                                StorageFilter, StorageOptions)
        from sensor_msgs.msg import CompressedImage
        from vision_msgs.msg import Detection2DArray
    except ImportError as e:
        die("extract needs the ROS 2 environment (source the workspace): %s" % e)

    out = expand(args.out)
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)

    reader = SequentialReader()
    reader.open(StorageOptions(uri=expand(args.bag), storage_id=args.storage),
                ConverterOptions("", ""))
    reader.set_filter(StorageFilter(topics=[args.image_topic, args.det_topic]))

    images, dets = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.image_topic:
            m = deserialize_message(data, CompressedImage)
            images.append((stamp_ns(m.header), m))
        elif topic == args.det_topic:
            m = deserialize_message(data, Detection2DArray)
            dets.append((stamp_ns(m.header), m))

    if not images:
        die("no messages on %s -- check the topic name with `ros2 bag info`"
            % args.image_topic)
    images.sort(key=lambda x: x[0])
    dets.sort(key=lambda x: x[0])
    print("bag: %d images, %d detection messages" % (len(images), len(dets)))

    chosen = images[::args.every]
    if args.max and len(chosen) > args.max:
        chosen = chosen[:args.max]
    print("sampling every %d -> %d frames" % (args.every, len(chosen)))

    det_stamps = [d[0] for d in dets]
    slop_ns = int(args.slop * 1e9)
    preds, manifest, matched, out_of_set = {}, [], 0, defaultdict(int)

    import bisect
    for i, (ts, msg) in enumerate(chosen):
        name = "%06d" % i
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            print("!! frame %s did not decode, skipped" % name)
            continue
        h, w = img.shape[:2]
        cv2.imwrite(os.path.join(frames_dir, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

        boxes = []
        if det_stamps:
            j = bisect.bisect_left(det_stamps, ts)
            best, bestd = None, None
            for k in (j - 1, j, j + 1):
                if 0 <= k < len(det_stamps):
                    d = abs(det_stamps[k] - ts)
                    if bestd is None or d < bestd:
                        best, bestd = k, d
            if best is not None and bestd <= slop_ns:
                matched += 1
                for det in dets[best][1].detections:
                    cid, score = det_class_score(det)
                    if cid is None:
                        continue
                    boxes.append({"cls": cid, "conf": score,
                                  "xyxy": [round(v, 2) for v in det_box_xyxy(det)]})
                    out_of_set[cid] += 1
        preds[name] = boxes
        manifest.append({"frame": name, "stamp_ns": ts, "w": w, "h": h})

    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump({"bag": expand(args.bag), "every": args.every,
                   "image_topic": args.image_topic, "det_topic": args.det_topic,
                   "slop_s": args.slop, "frames": manifest}, fh, indent=1)
    src = "bag %s (live %s)" % (os.path.basename(expand(args.bag)), args.det_topic)
    with open(os.path.join(out, "pred_onnx.json"), "w") as fh:
        json.dump({"source": src, "frames": preds}, fh)

    print("\nwrote %d frames to %s" % (len(manifest), frames_dir))
    print("detections matched within %.0f ms: %d / %d frames"
          % (args.slop * 1000, matched, len(manifest)))
    if matched < len(manifest):
        print("!! %d frames had no detection message within the window -- they will"
              % (len(manifest) - matched))
        print("   score as all-misses. Raise --slop or check the two stamps agree.")
    print("\nclasses the detector produced, by instance count:")
    for cid, n in sorted(out_of_set.items(), key=lambda kv: -kv[1]):
        print("   %-18s %5d" % (cid, n))
    print("\nNEXT: label %s/*.jpg into %s/labels/ (YOLO txt + classes.txt),"
          % (frames_dir, out))
    print("      an empty .txt for a frame with no target object.")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        die("predict needs the ultralytics venv "
            "(~/yolo/venv/bin/python): %s" % e)

    data = expand(args.data)
    frames_dir = os.path.join(data, "frames")
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not files:
        die("no frames in %s -- run extract first" % frames_dir)

    model = YOLO(expand(args.model))
    names = model.names
    preds = {}
    for f in files:
        res = model.predict(os.path.join(frames_dir, f), conf=args.conf,
                            imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            cid = int(b.cls.item())
            boxes.append({"cls": str(names.get(cid, cid)),
                          "conf": float(b.conf.item()),
                          "xyxy": [round(float(v), 2) for v in b.xyxy[0].tolist()]})
        preds[os.path.splitext(f)[0]] = boxes

    out = expand(args.out)
    with open(out, "w") as fh:
        json.dump({"source": "model %s conf %.2f imgsz %d device %s"
                   % (os.path.basename(expand(args.model)), args.conf,
                      args.imgsz, args.device),
                   "frames": preds}, fh)
    total = sum(len(v) for v in preds.values())
    print("wrote %s -- %d frames, %d detections" % (out, len(preds), total))


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def load_labels(data, manifest):
    """YOLO txt -> {frame: [(cls_name, x1,y1,x2,y2)]}. Frames with no .txt are
    treated as unlabelled and dropped, which is not the same as empty."""
    labels_dir = os.path.join(data, "labels")
    cf = os.path.join(labels_dir, "classes.txt")
    if not os.path.exists(cf):
        cf = os.path.join(data, "classes.txt")
    if not os.path.exists(cf):
        die("no classes.txt in %s or %s" % (labels_dir, data))
    with open(cf) as fh:
        names = [l.strip() for l in fh if l.strip()]

    size = {m["frame"]: (m["w"], m["h"]) for m in manifest}
    gt, unlabelled = {}, []
    for frame, (w, h) in size.items():
        path = os.path.join(labels_dir, frame + ".txt")
        if not os.path.exists(path):
            unlabelled.append(frame)
            continue
        rows = []
        with open(path) as fh:
            for ln, line in enumerate(fh, 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    die("%s line %d: expected 5 fields, got %d" % (path, ln, len(parts)))
                ci = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                if ci >= len(names):
                    die("%s line %d: class index %d but classes.txt has %d names"
                        % (path, ln, ci, len(names)))
                rows.append((names[ci],
                             (cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h))
        gt[frame] = rows
    return gt, unlabelled, names


def cmd_score(args):
    data = expand(args.data)
    with open(os.path.join(data, "manifest.json")) as fh:
        manifest = json.load(fh)["frames"]
    with open(expand(args.pred)) as fh:
        pred_blob = json.load(fh)
    preds = pred_blob["frames"]

    gt, unlabelled, label_names = load_labels(data, manifest)
    if not gt:
        die("no label files found -- nothing to score")
    if unlabelled:
        print("!! %d of %d frames have no .txt and are EXCLUDED from scoring"
              % (len(unlabelled), len(manifest)))
        print("   (an empty file means 'no target object'; a missing file means"
              " 'not labelled yet')")

    targets = args.classes or sorted({c for rows in gt.values() for c, *_ in rows})
    print("\nscoring %d labelled frames, IoU >= %.2f, conf >= %.2f"
          % (len(gt), args.iou, args.conf))
    print("target classes: %s" % ", ".join(targets))

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    n_gt = defaultdict(int); n_pred = defaultdict(int)
    out_of_set = defaultdict(int)

    for frame in gt:
        g_all = gt[frame]
        p_all = preds.get(frame, [])
        for b in p_all:
            if b["cls"] not in targets and b["conf"] >= args.conf:
                out_of_set[b["cls"]] += 1
        for cls in targets:
            g = [row[1:] for row in g_all if row[0] == cls]
            p = sorted([b for b in p_all
                        if b["cls"] == cls and b["conf"] >= args.conf],
                       key=lambda b: -b["conf"])
            n_gt[cls] += len(g); n_pred[cls] += len(p)
            used = set()
            for b in p:
                best, best_i = -1, 0.0
                for i, gb in enumerate(g):
                    if i in used:
                        continue
                    v = iou(b["xyxy"], gb)
                    if v > best_i:
                        best, best_i = i, v
                if best >= 0 and best_i >= args.iou:
                    used.add(best); tp[cls] += 1
                else:
                    fp[cls] += 1
            fn[cls] += len(g) - len(used)

    scored, coverage, rows = [], [], []
    for cls in targets:
        p, r, f = prf(tp[cls], fp[cls], fn[cls])
        row = {"class": cls, "gt": n_gt[cls], "pred": n_pred[cls],
               "tp": tp[cls], "fp": fp[cls], "fn": fn[cls],
               "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
        rows.append(row)
        (scored if n_gt[cls] >= args.min_instances else coverage).append(row)

    print("\n%-14s %6s %5s %5s %5s   %7s %7s %7s"
          % ("class", "truth", "TP", "FP", "FN", "prec", "recall", "F1"))
    print("-" * 68)
    for row in rows:
        mark = "" if row["gt"] >= args.min_instances else "   (< min, not averaged)"
        print("%-14s %6d %5d %5d %5d   %6.1f%% %6.1f%% %6.1f%%%s"
              % (row["class"], row["gt"], row["tp"], row["fp"], row["fn"],
                 100 * row["precision"], 100 * row["recall"], 100 * row["f1"], mark))

    if not scored:
        die("no class reached --min-instances %d; either label more frames or "
            "lower the threshold BEFORE looking at the scores" % args.min_instances)

    macro = {k: sum(r[k] for r in scored) / len(scored)
             for k in ("precision", "recall", "f1")}
    verdict = "PASS" if 100 * macro["f1"] >= args.criterion else "FAIL"
    print("-" * 68)
    print("macro over %d class(es): prec %.1f%%  recall %.1f%%  F1 %.1f%%"
          % (len(scored), 100 * macro["precision"], 100 * macro["recall"],
             100 * macro["f1"]))
    print("criterion: macro F1 >= %.0f%%  ->  %s" % (args.criterion, verdict))

    if out_of_set:
        print("\nout-of-set detections (not scored; evidence for the closed-set limit):")
        for cls, n in sorted(out_of_set.items(), key=lambda kv: -kv[1])[:12]:
            print("   %-18s %5d" % (cls, n))

    session = {
        "kind": "detection_accuracy",
        "run_label": args.run_label,
        "pred_source": pred_blob.get("source", expand(args.pred)),
        "data": data,
        "frames_labelled": len(gt),
        "frames_total": len(manifest),
        "iou": args.iou, "conf": args.conf,
        "min_instances": args.min_instances, "criterion_f1": args.criterion,
        "classes_scored": [r["class"] for r in scored],
        "classes_coverage_only": [r["class"] for r in coverage],
        "per_class": rows,
        "macro": {k: round(v, 4) for k, v in macro.items()},
        "verdict": verdict,
        "out_of_set": dict(sorted(out_of_set.items(), key=lambda kv: -kv[1])[:20]),
    }
    path = expand(args.session)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(session) + "\n")
    print("\nappended to %s" % path)

    if args.figure:
        draw_per_class(rows, macro, args)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

INK, INK2, ACC = "#1a1a1a", "#52606D", "#2F6F4E"


def style(lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = "DejaVu Sans"
    if lang == "th":
        names = {f.name for f in font_manager.fontManager.ttflist}
        thai = next((n for n in ("Loma", "Garuda", "Norasi", "Kinnari")
                     if n in names), None)
        if thai:
            fam = thai
        else:
            print("!! no Thai font found (fonts-tlwg-*); falling back to English")
            lang = "en"
    plt.rcParams.update({"font.family": fam, "font.size": 10,
                         "axes.unicode_minus": False, "figure.dpi": 150})
    return plt, lang


def draw_per_class(rows, macro, args):
    plt, lang = style(args.lang)
    import numpy as np
    shown = [r for r in rows if r["gt"] >= args.min_instances]
    if not shown:
        return
    txt = {"th": {"t": "ความแม่นยำของการตรวจจับ แยกตามคลาสเป้าหมาย",
                  "y": "ร้อยละ", "p": "Precision", "r": "Recall", "f": "F1",
                  "c": "เกณฑ์ F1 เฉลี่ย %.0f%%" % args.criterion,
                  "n": "จำนวนจริง %d"},
           "en": {"t": "Detection accuracy by target class",
                  "y": "percent", "p": "Precision", "r": "Recall", "f": "F1",
                  "c": "criterion: macro F1 %.0f%%" % args.criterion,
                  "n": "truth %d"}}[lang]

    x = np.arange(len(shown)); w = 0.26
    fig, ax = plt.subplots(figsize=(1.9 * len(shown) + 3.2, 4.0))
    for off, key, col, lab in ((-w, "precision", "#9BB7D4", txt["p"]),
                               (0.0, "recall", "#5A87B5", txt["r"]),
                               (w, "f1", ACC, txt["f"])):
        ax.bar(x + off, [100 * r[key] for r in shown], w, label=lab, color=col)
    ax.axhline(args.criterion, ls="--", lw=1.4, color="#444", label=txt["c"])
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(%s)" % (r["class"], txt["n"] % r["gt"]) for r in shown])
    ax.set_ylabel(txt["y"]); ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.set_title("%s   —   macro F1 %.1f%%" % (txt["t"], 100 * macro["f1"]), pad=30)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.5, color="#E3E6E4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = expand(args.figure)
    fig.savefig(out); print("wrote %s" % out)


def cmd_compare(args):
    path = expand(args.session)
    if not os.path.exists(path):
        die("no session file at %s" % path)
    runs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "detection_accuracy":
                runs.append(r)
    if len(runs) < 2:
        die("need at least two scored runs in %s; found %d" % (path, len(runs)))
    runs = runs[-args.last:]

    plt, lang = style(args.lang)
    import numpy as np
    txt = {"th": {"t": "เปรียบเทียบความแม่นยำระหว่างรูปแบบแบบจำลอง",
                  "y": "ร้อยละ (ค่าเฉลี่ยข้ามคลาส)",
                  "p": "Precision", "r": "Recall", "f": "F1"},
           "en": {"t": "Accuracy by model backend", "y": "percent (macro)",
                  "p": "Precision", "r": "Recall", "f": "F1"}}[lang]

    x = np.arange(len(runs)); w = 0.26
    fig, ax = plt.subplots(figsize=(2.6 * len(runs) + 3.0, 4.0))
    for off, key, col, lab in ((-w, "precision", "#9BB7D4", txt["p"]),
                               (0.0, "recall", "#5A87B5", txt["r"]),
                               (w, "f1", ACC, txt["f"])):
        vals = [100 * r["macro"][key] for r in runs]
        ax.bar(x + off, vals, w, label=lab, color=col)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 1.2, "%.1f" % v, ha="center", fontsize=8, color=INK2)
    ax.axhline(runs[-1]["criterion_f1"], ls="--", lw=1.4, color="#444",
               label="criterion %.0f%%" % runs[-1]["criterion_f1"])
    ax.set_xticks(x)
    ax.set_xticklabels([r.get("run_label") or "run %d" % i for i, r in enumerate(runs, 1)])
    ax.set_ylabel(txt["y"]); ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    ax.set_title(txt["t"], pad=30)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.5, color="#E3E6E4"); ax.set_axisbelow(True)
    fig.tight_layout()
    out = expand(args.figure); fig.savefig(out); print("wrote %s" % out)

    print("\n%-16s %8s %8s %8s  %s" % ("run", "prec", "recall", "F1", "classes"))
    for r in runs:
        print("%-16s %7.1f%% %7.1f%% %7.1f%%  %s"
              % (r.get("run_label", "?"), 100 * r["macro"]["precision"],
                 100 * r["macro"]["recall"], 100 * r["macro"]["f1"],
                 ",".join(r["classes_scored"])))
    print("\nEqual accuracy at a higher frame rate is the evidence for the export")
    print("decision. If F1 differs by more than a point or two, say so instead.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Objective 2: score the deployed detector on hand-labelled frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pull frames + live detections from a bag")
    e.add_argument("--bag", required=True)
    e.add_argument("--out", required=True, help="dataset directory to create")
    e.add_argument("--every", type=int, default=23,
                   help="keep every Nth image (default 23 ~= 120 frames of a 215 s bag)")
    e.add_argument("--max", type=int, default=0, help="cap the frame count (0 = no cap)")
    e.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    e.add_argument("--det-topic", default=DEFAULT_DET_TOPIC)
    e.add_argument("--slop", type=float, default=0.05,
                   help="seconds; how close a detection stamp must be to the frame")
    e.add_argument("--storage", default="sqlite3")
    e.add_argument("--quality", type=int, default=95)
    e.set_defaults(func=cmd_extract)

    p = sub.add_parser("predict", help="run another model over the extracted frames")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("score", help="match predictions to labels and report P/R/F1")
    s.add_argument("--data", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--classes", nargs="+", default=None,
                   help="target classes; default = every class present in the labels")
    s.add_argument("--iou", type=float, default=0.5)
    s.add_argument("--conf", type=float, default=0.5)
    s.add_argument("--min-instances", type=int, default=50,
                   help="a class below this is coverage-only, not averaged")
    s.add_argument("--criterion", type=float, default=80.0,
                   help="percent; checked against the macro F1")
    s.add_argument("--run-label", default="run")
    s.add_argument("--session", default=DEFAULT_SESSION)
    s.add_argument("--figure", default=None)
    s.add_argument("--lang", choices=("th", "en"), default="th")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="draw the backend comparison from the session")
    c.add_argument("--session", default=DEFAULT_SESSION)
    c.add_argument("--figure", required=True)
    c.add_argument("--last", type=int, default=2, help="how many runs to draw")
    c.add_argument("--lang", choices=("th", "en"), default="th")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
