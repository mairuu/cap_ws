#!/usr/bin/env python3
"""Verify a multi-machine ROS 2 link carries what this robot actually sends.

Two streams, because they fail for different reasons and only one of them is
usually tested:

  small  std_msgs/String at 10 Hz  -- discovery and the plain unicast path.
  large  nav_msgs/OccupancyGrid    -- the same shape as our real /map
         (162 x 249 @ 0.05 m, ~40 kB). Anything over the ~64 kB UDP datagram
         limit is fragmented by Fast DDS, and fragments are dropped silently
         when the OS socket buffers are smaller than the burst. That failure
         looks like "RViz shows the laser but the map never appears", and it
         is NOT a discovery problem, so a talker/listener test passes right
         through it.

Run the publisher on the robot and the subscriber on the viewing machine:

    # Jetson
    ./check_ros2_link.py --pub
    # laptop
    ./check_ros2_link.py --sub

Exits non-zero if either stream fails to arrive, so it can gate a checklist.
"""
import argparse, sys, time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

# Matches the real map published on 9 Sep: 162 x 249 cells at 0.05 m.
MAP_W, MAP_H, MAP_RES = 162, 249, 0.05
SMALL_HZ, LARGE_HZ = 10.0, 2.0


class Pub(Node):
    def __init__(self):
        super().__init__("link_test_pub")
        self.small = self.create_publisher(String, "/link_test/small", 10)
        self.large = self.create_publisher(OccupancyGrid, "/link_test/large", 1)
        self.n = 0
        self.create_timer(1.0 / SMALL_HZ, self.tick_small)
        self.create_timer(1.0 / LARGE_HZ, self.tick_large)

        self.grid = OccupancyGrid()
        self.grid.header.frame_id = "map"
        self.grid.info.resolution = MAP_RES
        self.grid.info.width = MAP_W
        self.grid.info.height = MAP_H
        # Not all -1: an unknown-everywhere grid compresses trivially on some
        # transports and would not exercise the fragmentation path honestly.
        self.grid.data = [(i % 101) - 1 for i in range(MAP_W * MAP_H)]
        self.get_logger().info(
            f"publishing {MAP_W}x{MAP_H} grid (~{MAP_W*MAP_H/1000:.0f} kB) "
            f"at {LARGE_HZ} Hz and a String at {SMALL_HZ} Hz")

    def tick_small(self):
        m = String()
        m.data = f"seq {self.n}"
        self.n += 1
        self.small.publish(m)

    def tick_large(self):
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.large.publish(self.grid)


class Sub(Node):
    def __init__(self, secs):
        super().__init__("link_test_sub")
        self.small_n = self.large_n = 0
        self.bytes = 0
        self.create_subscription(String, "/link_test/small", self.on_small, 10)
        self.create_subscription(OccupancyGrid, "/link_test/large", self.on_large, 1)
        self.t0 = time.monotonic()
        self.secs = secs

    def on_small(self, _):
        self.small_n += 1

    def on_large(self, m):
        self.large_n += 1
        self.bytes = len(m.data)

    def done(self):
        return time.monotonic() - self.t0 >= self.secs

    def report(self):
        el = time.monotonic() - self.t0
        sr, lr = self.small_n / el, self.large_n / el
        print(f"\nover {el:.1f} s:")
        print(f"  small  {self.small_n:4d} msgs  {sr:5.2f} Hz  (expect ~{SMALL_HZ})")
        print(f"  large  {self.large_n:4d} msgs  {lr:5.2f} Hz  (expect ~{LARGE_HZ})"
              f"  {self.bytes/1000:.0f} kB each" if self.bytes else
              f"  large  {self.large_n:4d} msgs  {lr:5.2f} Hz  (expect ~{LARGE_HZ})")
        ok = True
        if self.small_n == 0:
            print("\nFAIL  no small messages. Discovery never completed.")
            print("      Check ROS_DOMAIN_ID matches, ROS_LOCALHOST_ONLY=0, and that")
            print("      both machines' addresses are in the Fast DDS initial peers.")
            ok = False
        elif sr < SMALL_HZ * 0.8:
            print(f"\nWARN  small stream lossy ({sr:.2f} of {SMALL_HZ} Hz). Weak link.")
        if self.large_n == 0:
            print("\nFAIL  small arrived but large did not -- this is the fragmentation")
            print("      failure, NOT a discovery failure. Raise the socket buffers:")
            print("        sudo sysctl -w net.core.rmem_max=8388608")
            print("      and add to the participant <rtps> in ~/.ros2/fastdds_hotspot.xml:")
            print("        <sendSocketBufferSize>1048576</sendSocketBufferSize>")
            print("        <listenSocketBufferSize>4194304</listenSocketBufferSize>")
            print("      Both machines. Then re-run.")
            ok = False
        elif lr < LARGE_HZ * 0.8:
            print(f"\nWARN  large stream lossy ({lr:.2f} of {LARGE_HZ} Hz). The map will")
            print("      redraw slowly in RViz. Same socket-buffer fix as above.")
        if ok:
            print("\nOK    both streams arrived. /scan, /map and costmaps will cross.")
        return ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pub", action="store_true", help="run on the robot")
    g.add_argument("--sub", action="store_true", help="run on the viewing machine")
    ap.add_argument("--secs", type=float, default=10.0, help="how long --sub listens")
    a = ap.parse_args()

    rclpy.init()
    if a.pub:
        n = Pub()
        try:
            rclpy.spin(n)
        except KeyboardInterrupt:
            pass
        n.destroy_node()
        rclpy.shutdown()
        return 0

    n = Sub(a.secs)
    print(f"listening {a.secs:.0f} s ...")
    while rclpy.ok() and not n.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    ok = n.report()
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
