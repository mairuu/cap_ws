#!/usr/bin/env python3
"""Draw the live ROS 2 node/topic graph to a PNG, headless. Report figure.

WHAT THIS IS FOR. The report's `ros2_node_graph.png` is what `rqt_graph`
shows, but rqt_graph needs a display and the Jetson has none. This asks the
running graph the same questions rqt_graph does (every node's publishers and
subscribers), writes Graphviz DOT, and renders it with `dot`. No display, no
clicking, and the same picture every time for the same stack.

    make up PROFILE=demo                       # the stack must be running
    python3 node_graph.py --out ~/cap_ref/figures/ros2_node_graph.png

WHAT IS HIDDEN, and why. Like rqt_graph's "Nodes/Topics (active)" with "hide
debug" and "hide tf": a topic is drawn only when a shown node publishes it AND
a shown node subscribes to it, and the plumbing every ROS 2 graph has is
dropped -- /rosout, /parameter_events, lifecycle transition_event and bond
topics, action feedback/status, /tf and /tf_static (the TF tree is its own
figure), Nav2's costmap_raw / published_footprint / costmap_updates, and rviz
(a viewer, possibly on another machine). `--keep-tf` and `--nav2-internals`
put those back; `--hide REGEX` drops more.

GROUPS. Nodes are boxed by system layer (--no-groups turns it off), matched on
name by the table below; anything unmatched is drawn outside the boxes, so a
new node shows up rather than disappearing.
"""

import argparse
import os
import re
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node

HIDE_TOPICS = [
    r"^/rosout$", r"^/parameter_events$", r"/transition_event$", r"^/bond$",
    r"/_action/", r"^/tf$", r"^/tf_static$", r"^/clock$",
    # Nav2's internal costmap plumbing: real flows, but they triple the edge
    # count and say nothing about the system. --nav2-internals keeps them.
    r"/costmap_raw$", r"/published_footprint$", r"/costmap_updates$",
]
NAV2_INTERNAL = HIDE_TOPICS[-3:]
# rviz is a viewer on whichever machine opened it, not part of the robot.
HIDE_NODES = [r"^_", r"^launch_ros_", r"^transform_listener_impl_",
              r"^node_graph$", r"^rviz", r"^stack_wait$"]

# (label, colour, node-name regexes) -- first match wins.
GROUPS = [
    ("ฐานหุ่นยนต์ (ros2_control, LiDAR)", "#e8f1fb", [
        r"^robot_state_publisher$", r"^controller_manager$", r"^diff_cont$",
        r"^joint_broad$", r"^imu_broad$", r"^ydlidar", r"^ekf", r"^twist_mux$",
        r"^teleop"]),
    ("SLAM", "#fdf0e9", [r"slam_toolbox$"]),
    ("นำทาง (Nav2)", "#eaf7f1", [
        r"^controller_server$", r"^planner_server$", r"^bt_navigator",
        r"^behavior_server$", r"^smoother_server$", r"^velocity_smoother$",
        r"^waypoint_follower$", r"^lifecycle_manager", r"costmap",
        r"^collision_monitor$", r"^explore"]),
    ("ตรวจจับวัตถุ", "#fdf6e3", [
        r"^yolo", r"^cam2image$", r"^camera", r"^focus_lock$"]),
    ("เว็บแอปพลิเคชัน", "#f1f1ef", [r"bridge", r"^control_node$", r"^rosbridge"]),
    ("ผสานข้อมูลเชิงความหมาย", "#f3eefb", [r"^semantic", r"republish$"]),
]


def matches(name, patterns):
    return any(re.search(p, name) for p in patterns)


def collect(settle):
    rclpy.init()
    node = Node("node_graph")
    # Discovery is asynchronous: ask too early and half the graph is missing.
    end = time.time() + settle
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    edges_pub, edges_sub = [], []
    names = []
    for name, ns in node.get_node_names_and_namespaces():
        full = (ns.rstrip("/") + "/" + name) if ns not in ("", "/") else "/" + name
        names.append((name, full))
        try:
            pubs = node.get_publisher_names_and_types_by_node(name, ns)
            subs = node.get_subscriber_names_and_types_by_node(name, ns)
        except Exception as e:  # node vanished between the two calls
            print(f"  skipped {full}: {e}")
            continue
        edges_pub += [(full, t) for t, _ in pubs]
        edges_sub += [(full, t) for t, _ in subs]
    node.destroy_node()
    rclpy.shutdown()
    return names, edges_pub, edges_sub


def q(s):
    return '"' + s.replace('"', r'\"') + '"'


def build_dot(names, pubs, subs, hide_topics, groups):
    shown = {full for name, full in names if not matches(name, HIDE_NODES)}
    pubs = [(n, t) for n, t in pubs if n in shown and not matches(t, hide_topics)]
    subs = [(n, t) for n, t in subs if n in shown and not matches(t, hide_topics)]
    active = {t for _, t in pubs} & {t for _, t in subs}
    pubs = [(n, t) for n, t in pubs if t in active]
    subs = [(n, t) for n, t in subs if t in active]
    connected = {n for n, _ in pubs} | {n for n, _ in subs}

    out = ["digraph ros {",
           '  graph [rankdir=LR, fontname="Loma", fontsize=13, nodesep=0.18, '
           'ranksep=0.9, pad=0.2, compound=true];',
           '  node [fontname="DejaVu Sans", fontsize=10];',
           '  edge [color="#52514e", arrowsize=0.6];']
    placed = set()
    if groups:
        for i, (label, colour, pats) in enumerate(GROUPS):
            members = sorted(n for n in connected
                             if n not in placed and matches(n.rsplit("/", 1)[-1], pats))
            if not members:
                continue
            placed.update(members)
            out.append(f"  subgraph cluster_{i} {{")
            out.append(f'    label={q(label)}; style="rounded,filled"; '
                       f'fillcolor="{colour}"; color="#c8c7c2";')
            for n in members:
                out.append(f'    {q(n)} [shape=ellipse, style=filled, '
                           f'fillcolor="white", color="#2a78d6"];')
            out.append("  }")
    for n in sorted(connected - placed):
        out.append(f'  {q(n)} [shape=ellipse, style=filled, fillcolor="white", '
                   f'color="#2a78d6"];')
    for t in sorted(active):
        out.append(f'  {q("T" + t)} [label={q(t)}, shape=box, style="rounded", '
                   f'color="#8a8984", fontsize=9];')
    for n, t in sorted(set(pubs)):
        out.append(f"  {q(n)} -> {q('T' + t)};")
    for n, t in sorted(set(subs)):
        out.append(f"  {q('T' + t)} -> {q(n)};")
    out.append("}")
    return "\n".join(out), connected, active, placed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="~/cap_ref/figures/ros2_node_graph.png")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="seconds of discovery before asking (default 4)")
    ap.add_argument("--keep-tf", action="store_true", help="draw /tf and /tf_static")
    ap.add_argument("--nav2-internals", action="store_true",
                    help="draw Nav2's costmap_raw / footprint / updates topics")
    ap.add_argument("--hide", action="append", default=[], metavar="REGEX",
                    help="also hide topics matching this (repeatable)")
    ap.add_argument("--no-groups", action="store_true", help="no layer boxes")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    hide = [p for p in HIDE_TOPICS
            if not (args.keep_tf and "tf" in p)
            and not (args.nav2_internals and p in NAV2_INTERNAL)] + args.hide
    names, pubs, subs = collect(args.settle)
    if len(names) < 2:
        print("no ROS graph visible -- is the stack up, and ROS_DOMAIN_ID the same?")
        return 1
    dot, nodes, topics, placed = build_dot(names, pubs, subs, hide, not args.no_groups)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dot_path = os.path.splitext(out)[0] + ".dot"
    with open(dot_path, "w") as fh:
        fh.write(dot)
    subprocess.run(["dot", "-Tpng", f"-Gdpi={args.dpi}", dot_path, "-o", out], check=True)

    print(f"{len(names)} nodes visible, {len(nodes)} drawn, {len(topics)} topics drawn")
    ungrouped = sorted(nodes - placed)
    if ungrouped and not args.no_groups:
        print("  outside any layer box: " + ", ".join(ungrouped))
    print(f"  -> {out}\n  -> {dot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
