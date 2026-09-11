SHELL := /bin/bash
ROS_DISTRO ?= humble

.PHONY: build sim real slam nav explore rviz save-map yolo camera calib calib-report calib-scale teleop teleop-nav udev ports net net-check viewer-sync lidar-deps clean

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
#
# USE_LIDAR=false brings up drive and odometry WITHOUT the lidar. Needed until
# ydlidar_ros2_driver is built from source -- it is not an apt package, and a
# missing executable takes the whole launch down, base included:
#   make real USE_LIDAR=false
LIDAR_PORT ?= /dev/ydlidar
USE_LIDAR ?= true
real: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot real_robot.launch.py lidar_port:=$(LIDAR_PORT) use_lidar:=$(USE_LIDAR)

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
#
# SAVE_TIMEOUT is not decoration. map_saver_cli defaults to 2 s and gives up
# with "Failed to spin map subscription" -- which reads exactly like a dead
# mapper, and is not. /map is latched TRANSIENT_LOCAL, so a fresh subscriber
# must be sent the whole grid on connect, and a room-sized map does not arrive
# in 2 s. Seen 10 Sep on a 255x557 map; 60 s saved it first try.
MAP ?= $(HOME)/my_map
SAVE_TIMEOUT ?= 60.0
save-map:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 run nav2_map_server map_saver_cli -f $(MAP) \
	  --ros-args -p save_map_timeout:=$(SAVE_TIMEOUT)

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

# Camera intrinsics -- Day 4 sec 4. Needs the camera and nothing else: no
# robot, no lidar, no Nav2. Two terminals.
#
#   terminal 1:  make camera
#   terminal 2:  make calib
#   then:        make calib-report            (after pressing SAVE)
#
# The RECOVERY.md sketch of this target is WRONG on all four counts and is kept
# there only as history: it says usb_cam, /camera/image_raw, 8x6 and 0.025. The
# camera is cam2image on /image, and the board is 9x6 / 20 mm. See
# reference/nvme-recovery-audit.md.
#
# WIDTH/HEIGHT are not decoration. cam2image DEFAULTS TO 320x240, and intrinsics
# do not transfer across resolutions -- fx, fy, cx and cy all scale. Calibrate
# at the size the robot actually runs, which is what yolo.launch.py sets.
# THE C615 IS AN AUTOFOCUS CAMERA AND AUTOFOCUS CHANGES fx. It is varifocal:
# refocusing moves the lens, so the focal length -- the thing we are calibrating
# -- is not a constant while `focus_automatic_continuous` is 1. Observed on this
# board 11 Sep: focus_absolute was left at 51, AF was re-enabled, and the driver
# had moved it to 85 by the next read, unprompted. A calibration captured across
# that is a fit of one pinhole model to several different cameras, and worse, at
# run time the lens keeps moving away from whatever was calibrated.
#
# So: lock it, and lock it to the SAME value for calibration and for the demo.
# The two calls cannot be combined -- setting focus_absolute in the same
# VIDIOC_S_EXT_CTRLS transaction that still has AF enabled is rejected outright.
#
# 51 is the device default. Pick a value that is sharp at the demo's working
# distance and then do not touch it.   make camera FOCUS=auto   restores AF.
CAM_WIDTH  ?= 640
CAM_HEIGHT ?= 480
CAM_FPS    ?= 15.0
CAM_DEV    ?= /dev/video0
FOCUS      ?= 51
camera:
	@if [ "$(FOCUS)" = "auto" ]; then \
	  v4l2-ctl -d $(CAM_DEV) -c focus_automatic_continuous=1; \
	  echo "AUTOFOCUS ON -- fx is not constant. Do not calibrate like this."; \
	else \
	  v4l2-ctl -d $(CAM_DEV) -c focus_automatic_continuous=0 && \
	  v4l2-ctl -d $(CAM_DEV) -c focus_absolute=$(FOCUS) && \
	  echo "focus locked at $(FOCUS)"; \
	fi
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	ros2 run image_tools cam2image --ros-args \
	  -p width:=$(CAM_WIDTH) -p height:=$(CAM_HEIGHT) \
	  -p frequency:=$(CAM_FPS) -p reliability:=reliable \
	  -p frame_id:=camera_link --log-level cam2image:=warn

# --no-service-check is REQUIRED, not optional: cameracalibrator otherwise waits
# for a set_camera_info service, and cam2image offers none, so it sits there
# looking hung. For the same reason COMMIT does nothing -- press SAVE, which
# writes /tmp/calibrationdata.tar.gz, and then run `make calib-report`.
BOARD  ?= 9x6
SQUARE ?= 0.020
calib:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	ros2 run camera_calibration cameracalibrator \
	  --size $(BOARD) --square $(SQUARE) --no-service-check \
	  -c c615 image:=/image

# cameracalibrator computes the reprojection error and then throws it away
# (calibrator.py:797) -- the number beside the CALIBRATE button is the LINEAR
# error, which is not the gate. This recovers it from the saved tarball, per
# image, and installs config/c615_640x480.yaml when it passes.
calib-report: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 run my_bot camera_calib_report.py \
	  --size $(BOARD) --square $(SQUARE) --write

# The check the calibration CANNOT do on itself. A chessboard fit has no
# absolute length in it: scale the squares and the solver scales the board
# distances and returns the same K. So reprojection error cannot see an fx that
# is 15% wrong from a degenerate capture -- every board at the same depth lets
# fx trade against distance freely, and the fit looks excellent.
#
# This puts the board at tape-measured distances and regresses what the model
# thinks against what the tape says. Slope 1.000 means fx is right; the
# intercept absorbs the entrance-pupil offset, which is why it wants more than
# one distance. Needs `make camera` running in another terminal.
DISTANCES ?= 0.4,0.7,1.0
calib-scale: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 run my_bot camera_check_scale.py \
	  --size $(BOARD) --square $(SQUARE) --distances $(DISTANCES)

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
#
# SPEED is the STARTING speed, 0.10 m/s rather than teleop_twist_keyboard's own
# 0.5. That 0.5 is what smeared the Day 3 map: the X3 Pro sweeps 360 deg in
# ~86 ms and slam_toolbox does not deskew, so at 0.5 m/s every scan is sheared
# 8.7 cm along the path and no rigid transform can absorb it. The same floor
# mapped clean at 0.10. Nav2 itself drives at 0.055.
#
# THE ACTUAL LIMIT IS NOT HERE. This publishes to /cmd_vel_teleop_raw, and
# teleop_speed_guard (launched by navigation.launch.py, next to twist_mux)
# clamps it onto /cmd_vel_teleop before twist_mux ever sees it. That is what
# makes `q` harmless: teleop_twist_keyboard's q raises its speed PERMANENTLY
# and shows the new value only in this terminal, which nobody watches while
# looking at RViz, so a default alone does not prevent a ruined map.
#
# Raise the real limit deliberately, at launch, not with a keypress:
#   make nav TELEOP_MAX_LINEAR:=0.3
#
# This does NOT weaken the e-stop -- `k` sends a zero Twist and the guard
# passes zero through unclamped, at any limit.
#
# NOTE `make nav` must be running, or nothing subscribes to _raw and the robot
# will not move at all. That is deliberate: the guard and twist_mux are one
# safety chain and neither should run without the other.
SPEED ?= 0.10
TURN  ?= 0.5
teleop-nav:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel_teleop_raw \
	  -p speed:=$(SPEED) -p turn:=$(TURN)

# One-time per machine: configure multi-machine ROS 2 over the phone hotspot.
# Run it on the Jetson AND on any laptop that runs RViz or teleop.
#
# Sets ROS_DOMAIN_ID (42, deliberately not 0 -- see the script),
# ROS_LOCALHOST_ONLY=0, and a Fast DDS profile that adds each machine as a
# UNICAST initial peer. The hotspot is an access point and drops
# client-to-client multicast; Fast DDS discovers by multicast by default, so
# without this two machines that ping each other fine see none of each other's
# topics. Re-run after any address change -- the hotspot hands out DHCP and
# the peer list is literal:
#   make net PEERS=172.20.10.2,172.20.10.7
#
# THIS TARGET ONLY EXISTS HERE. The laptop has no cap_ws and no Makefile -- by
# design, it needs nothing built. Run the script directly over there instead:
#   ~/cap_view/setup_ros2_network.sh --peers 172.20.10.2,172.20.10.5
# `make viewer-sync` is what puts it there.
DOMAIN ?= 42
PEERS  ?= 172.20.10.2,172.20.10.5
net:
	src/my_bot/scripts/setup_ros2_network.sh --domain $(DOMAIN) --peers $(PEERS)

# Push the three files the viewing laptop needs into ~/cap_view/ on it.
#
# They are COPIES, not links, and nothing detects drift: edit nav.rviz here and
# the laptop keeps showing the old layout until this runs. Re-run after
# changing any of the three.
#
# Deliberately not a full cap_ws checkout on the laptop -- the URDF is
# primitive geometry with no meshes, so RViz renders the robot straight from
# /robot_description over the wire, and every display in nav.rviz is a standard
# message type. Nothing over there needs colcon.
VIEWER ?= ju@172.20.10.5
viewer-sync:
	ssh $(VIEWER) 'mkdir -p ~/cap_view'
	scp src/my_bot/config/nav.rviz \
	    src/my_bot/scripts/setup_ros2_network.sh \
	    src/my_bot/scripts/check_ros2_link.py \
	    $(VIEWER):~/cap_view/
	@echo
	@echo 'On $(VIEWER), once:'
	@echo '  ~/cap_view/setup_ros2_network.sh --peers $(PEERS)'
	@echo 'then, in its own terminal:'
	@echo '  rviz2 -d ~/cap_view/nav.rviz'

# Prove the link carries what the robot actually sends, before blaming RViz.
# Publisher here, subscriber on the other machine:
#   make net-check                                   # on the robot
#   ./check_ros2_link.py --sub                       # on the laptop
#
# It sends a 10 Hz String AND a 162x249 OccupancyGrid, the size of our real
# /map. Those fail for different reasons: no String at all is a discovery
# problem, String-but-no-grid is UDP fragmentation and needs bigger socket
# buffers. A plain talker/listener test passes straight through the second one.
net-check:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	src/my_bot/scripts/check_ros2_link.py --pub

# One-time setup: build the YDLidar stack from source. NEITHER piece is an apt
# package, and without them there is no /scan at all -- `make real` defaults to
# use_lidar:=true and cannot come up.
#
# Two separate things, in this order:
#   1. YDLidar-SDK   plain CMake, installs libydlidar_sdk.a into /usr/local.
#                    ydlidar_ros2_driver's find_package(ydlidar_sdk) fails
#                    without it, so it MUST be installed before the colcon build.
#   2. ydlidar_ros2_driver   a colcon package, cloned into src/.
#
# THE BRANCH IS NOT OPTIONAL. Upstream's default branch is `master` and it is
# Dashing-era: the launch files pass node_executable= / node_name= (removed in
# Foxy) and the node calls the one-argument declare_parameter(name), which
# Humble deprecated and which throws when no override is supplied. The `humble`
# branch is the one that builds and runs here.
#
# Verified 9 Sep 2026: SDK 01cdda4, driver humble @ 4ef70d3, /scan at 11.57 Hz.
# Re-running this is safe -- both steps skip work that is already done.
SDK_DIR ?= $(HOME)/YDLidar-SDK
lidar-deps:
	@if [ ! -d $(SDK_DIR) ]; then 		git clone https://github.com/YDLIDAR/YDLidar-SDK.git $(SDK_DIR); 	fi
	cmake -S $(SDK_DIR) -B $(SDK_DIR)/build -DCMAKE_BUILD_TYPE=Release
	cmake --build $(SDK_DIR)/build -j$$(nproc)
	sudo cmake --install $(SDK_DIR)/build
	@if [ ! -d src/ydlidar_ros2_driver ]; then 		git clone -b humble https://github.com/YDLIDAR/ydlidar_ros2_driver.git 			src/ydlidar_ros2_driver; 	fi
	@branch=$$(git -C src/ydlidar_ros2_driver rev-parse --abbrev-ref HEAD); 	if [ "$$branch" != humble ]; then 		echo "ERROR: src/ydlidar_ros2_driver is on '$$branch', not 'humble'."; 		echo "       master is Dashing-era and will not run on Humble."; 		echo "       git -C src/ydlidar_ros2_driver checkout humble"; 		exit 1; 	fi
	$(MAKE) build

clean:
	rm -rf build install log
