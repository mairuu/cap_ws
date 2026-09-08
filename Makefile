SHELL := /bin/bash
ROS_DISTRO ?= humble

.PHONY: build sim real slam nav explore rviz save-map yolo teleop teleop-nav udev ports clean

# yolo_ros lives in the older dev_ws, not here, so its install has to be on the
# path for `make yolo`. cap_ws is sourced after it and wins on the packages
# (my_bot, ydlidar_ros2_driver) that exist in both.
DEV_WS ?= /home/jetson/dev_ws

build:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && colcon build --symlink-install

# Which world `make sim` loads, by basename from src/my_bot/worlds/.
#
# Defaults to `room` (a bounded 8x6 m room with two offset doorways and a
# couple of pillars) because empty.world is an infinite ground plane: with no
# walls the map never gains a boundary, so there are always more frontiers and
# `make explore` never terminates. For the old bare-plane behaviour:
#   make sim WORLD=empty
WORLD ?= room
sim: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot launch_sim.launch.py \
	  world:=$(CURDIR)/src/my_bot/worlds/$(WORLD).world

# Real-robot bringup: robot_state_publisher, the controller manager talking to
# the ESP32, both controllers, and the YDLidar.
#
# The lidar and the ESP32 are both USB serial adapters and whichever enumerates
# first becomes /dev/ttyUSB0, so raw ttyUSB* numbers are a coin flip across
# reboots. `make udev` installs stable names for both; run it once per machine.
# Override only if you need a raw port:  make real LIDAR_PORT=/dev/ttyUSB0
LIDAR_PORT ?= /dev/ydlidar
real: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot real_robot.launch.py lidar_port:=$(LIDAR_PORT)

# Online async SLAM. Starts ONLY the mapper, so `make real` (or `make sim`)
# must already be running in another terminal -- otherwise there is no /scan
# and no odom TF and this just sits there waiting.
# Against the simulator instead:  make slam SIM_TIME=true
SIM_TIME ?= false
slam: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot slam.launch.py use_sim_time:=$(SIM_TIME)

# Nav2 + twist_mux: the layer that turns a goal into wheel velocities. slam
# alone cannot drive the robot -- it only publishes /map and map->odom.
#
# Needs `make real` (or `make sim`) AND `make slam` already running. Then click
# "2D Goal Pose" in RViz, or start `make explore`.
#
# Against the simulator:  make nav SIM_TIME=true
#
# SAFETY: teleop outranks Nav2 through twist_mux, so this overrides an
# autonomous run at any time and is the only e-stop the robot has:
#   ros2 run teleop_twist_keyboard teleop_twist_keyboard \
#     --ros-args -r /cmd_vel:=/cmd_vel_teleop
nav: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot navigation.launch.py use_sim_time:=$(SIM_TIME)

# Frontier exploration: the robot picks its own goals and maps the room
# unattended. Sits on top of Nav2 and drives nothing itself, so ALL THREE of
# `make real`/`make sim`, `make slam` and `make nav` must already be running.
#
# If clicking a goal in RViz does not work, this will not either -- fix nav first.
#
# Against the simulator:  make explore SIM_TIME=true
explore: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot explore.launch.py use_sim_time:=$(SIM_TIME)

# RViz with the navigation layout: map, robot, laser, global/local plan and the
# explore_lite frontier markers, Fixed Frame already set to map.
#
# Plain `rviz2` opens with nothing but an empty grid -- every display has to be
# added by hand -- which looks exactly like a broken stack. Use this instead.
rviz: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	rviz2 -d install/my_bot/share/my_bot/config/nav.rviz

# Write the map slam_toolbox currently holds to disk, as <MAP>.pgm + <MAP>.yaml.
# Run while the mapper is still up.  make save-map MAP=~/maps/lab
MAP ?= $(HOME)/my_map
save-map:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 run nav2_map_server map_saver_cli -f $(MAP)

# YOLO detection off the USB webcam. Publishes /yolo/detections and
# /yolo/tracking (yolo_msgs/DetectionArray) plus /yolo/dbg_image.
#
# Runs standalone -- it needs no other bringup, just the camera. Inference uses
# the hand-built venv under dev_ws (JetPack torch + TensorRT); the launch puts
# it on PYTHONPATH itself, so do NOT source the venv's activate first and do
# NOT let anything run `uv sync` against it. See src/my_bot/launch/yolo.launch.py.
#
# Swap models:  make yolo MODEL=/home/jetson/yolo/yolo26s.engine
MODEL ?= /home/jetson/yolo/yolo26n.engine
yolo: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source $(DEV_WS)/install/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot yolo.launch.py model:=$(MODEL)

# One-time setup: pin the ESP32 and the lidar to /dev/esp32 and /dev/ydlidar so
# they stop trading ttyUSB numbers. Interactive -- run with BOTH plugged in.
udev:
	src/my_bot/scripts/setup_udev.sh

# What is currently plugged in, and where the stable names point.
ports:
	@for n in esp32 ydlidar; do \
		if [ -e /dev/$$n ]; then \
			printf '  /dev/%-8s -> %s\n' $$n "$$(readlink -f /dev/$$n)"; \
		else \
			printf '  /dev/%-8s MISSING (run: make udev)\n' $$n; \
		fi; \
	done
	@echo '  --- all serial devices ---'
	@d=$$(ls -1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null); \
	if [ -n "$$d" ]; then echo "$$d" | sed 's/^/  /'; else echo '  (none plugged in)'; fi

# Manual driving with NO navigation running. Publishes straight to the
# controller, bypassing twist_mux (which only exists while `make nav` is up).
teleop:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/diff_cont/cmd_vel_unstamped

# Manual driving WHILE `make nav` / `make explore` is running. THIS IS THE
# E-STOP -- use it, not `make teleop`, during any autonomous run.
#
# It publishes to /cmd_vel_teleop, which twist_mux gives priority 100 against
# Nav2's priority 10, so any keypress instantly takes control away from the
# robot. `make teleop` publishes DOWNSTREAM of the mux instead, so during an
# autonomous run it does not override Nav2 -- the two just fight over the same
# topic and the robot jerks between them. Getting this wrong is the difference
# between stopping the robot and making it worse.
#
# Press k (or anything with zero velocity) to stop. Hold the terminal focused.
teleop-nav:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel_teleop

clean:
	rm -rf build install log
