SHELL := /bin/bash
ROS_DISTRO ?= humble

.PHONY: build sim real slam nav explore rviz save-map yolo yolo-onnx camera calib calib-report calib-scale semantic test bridge bridge-venv ui ui-deps teleop teleop-nav udev ports net net-check viewer-sync lidar-deps clean

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

# YOLO detection + tracking off the USB webcam -- Day 5, decision D-11 B.
# Publishes vision_msgs/Detection2DArray on /detections (track id in
# Detection2D.id, capture-time stamp) and an annotated /detections/image.
#
# Runs standalone -- it needs no other bringup. It starts cam2image itself
# with focus LOCKED at FOCUS (same value as the calibration; the C615 is
# varifocal and autofocus moves fx), so do not also run `make camera`; two
# processes cannot hold /dev/video0. If a camera IS already up:
#   make yolo USE_CAMERA=false
#
# Inference uses the hand-built venv at ~/yolo/venv (JetPack torch + CUDA,
# plus onnxruntime-gpu for the .onnx path -- the JetPack aarch64 wheel, NOT
# the PyPI one).
# The launch puts its site-packages on PYTHONPATH itself: do NOT source the
# venv's activate first, and NEVER run `uv sync` against it. Rebuild it only
# with yolo/setup_yolo_venv.sh. See src/my_bot/launch/yolo.launch.py.
#
# THE MODEL IS AN .onnx, BUILT FROM THE .pt HERE. `make yolo` runs yolo-onnx
# first, which exports $(PT_MODEL) -> $(MODEL) with yolo/export_onnx.py. That
# export is IDEMPOTENT -- it is skipped when the .onnx is newer than the .pt and
# its embedded metadata matches IMGSZ and ONNX_HALF -- so a normal bring-up pays
# nothing for it. Change IMGSZ and it rebuilds, because a static ONNX graph has
# its input resolution baked in. Unlike a TensorRT .engine, an .onnx is portable
# and does not die with a JetPack change.
#
# The weights are yolo26s, not yolo26n, and the export is fp16 (ONNX_HALF).
# Both were measured on this board 16 Sep, 640x640, track() wall time:
#   yolo26n .pt torch    35.5 ms      yolo26s .onnx fp32   45.8 ms
#   yolo26s .pt torch    36.6 ms      yolo26s .onnx fp16   35.2 ms
# Two things that table settles. The bigger model is nearly free -- 1.1 ms over
# nano -- because this pipeline is launch-bound, not compute-bound, the same
# finding as 14 Sep. And ONNX fp32 is a 9 ms REGRESSION against plain torch;
# only the fp16 export pays for itself. Do not ship ONNX_HALF=false.
# Re-run the gate check below after any model change -- 15 Hz is 66.7 ms.
#
# Force a rebuild:  make yolo-onnx FORCE=true
# Export only:      make yolo-onnx
# Run the .pt:      make yolo MODEL=$(HOME)/yolo/yolo26s.pt    (no export)
# Swap models:      make yolo PT_MODEL=$(HOME)/yolo/yolov8n.pt MODEL=$(HOME)/yolo/yolov8n.onnx
# Smaller input:    make yolo IMGSZ=480                        (re-exports)
# No GPU:           make yolo DEVICE=cpu           (demo-day fallback, ~5 Hz)
# Gate check:       ros2 run my_bot detection_report.py --seconds 300
MODEL      ?= $(HOME)/yolo/yolo26s.onnx
PT_MODEL   ?= $(MODEL:.onnx=.pt)
IMGSZ      ?= 640
DEVICE     ?= cuda:0
USE_CAMERA ?= true
ONNX_OPSET ?= 17
ONNX_HALF  ?= true
FORCE      ?= false
YOLO_VENV  ?= $(HOME)/yolo/venv

# Skips itself when MODEL is not an .onnx, so `make yolo MODEL=.../x.pt` still
# works as it did on Day 5 -- the torch path is the fallback, not dead code.
yolo-onnx:
	@if [ "$(suffix $(MODEL))" != ".onnx" ]; then \
	  echo "MODEL=$(MODEL) is not .onnx -- skipping the export."; \
	else \
	  $(YOLO_VENV)/bin/python yolo/export_onnx.py \
	    --model $(PT_MODEL) --out $(MODEL) --imgsz $(IMGSZ) \
	    --opset $(ONNX_OPSET) --device $(DEVICE) \
	    $(if $(filter true,$(ONNX_HALF)),--half,) \
	    $(if $(filter true,$(FORCE)),--force,); \
	fi

yolo: build yolo-onnx
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch my_bot yolo.launch.py model:=$(MODEL) imgsz:=$(IMGSZ) \
	  device:=$(DEVICE) use_camera:=$(USE_CAMERA) focus:=$(FOCUS) \
	  camera_device:=$(CAM_DEV) camera_fps:=$(CAM_FPS)

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
#
# Starts at SPEED, not teleop_twist_keyboard's own 0.5 m/s default. 0.5 is what
# smeared the Day 3 map on 9 and 10 Sep: the X3 Pro sweeps 360 degrees over a
# full 100 ms and the stamp is already ~88 ms old at receipt, so 0.5 m/s shears
# each scan by ~8.7 cm along the path. No rigid transform absorbs that, and
# optimisation can move a scan's pose but cannot un-shear the scan, so the
# doubled wall stays drawn. diff_cont's ceiling (0.15 m/s) stops `q` running
# away, but mapping wants 0.10.
teleop:
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/diff_cont/cmd_vel_unstamped \
	  -p speed:=$(SPEED) -p turn:=$(TURN)

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

# ---------------------------------------------------------------------------
# Day 6 -- semantic fusion, bridge, UI
# ---------------------------------------------------------------------------

# Camera + lidar fusion node, plus the image_transport republish that gives
# the browser bridge /image/compressed. Needs make real (TF, odom, /scan),
# make slam (map frame) and make yolo (/detections) in other terminals.
# Runs under PLAIN SYSTEM PYTHON -- never source the YOLO venv first; the
# node needs rclpy/tf2/yaml from the system and nothing from the venv.
#   make semantic
#   make semantic PERSIST=            (no landmarks.json)
PERSIST ?= $(HOME)/maps/landmarks.json
semantic: build
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	source install/setup.bash && \
	ros2 launch semantic_objects semantic.launch.py persist_path:=$(PERSIST)

# Unit tests for the fusion maths, no ROS graph needed. Run from the package
# dir on purpose: system pytest is 6.2.5 and has no pythonpath ini option,
# so a bare `pytest` from here cannot import the package.
test:
	cd src/semantic_objects && python3 -m pytest test -q

# Browser bridge (FastAPI) on port 8000. Lives in cap_ref/semantic-object.
# Its venv is uv, pinned to the system python 3.10 (rclpy is cpython-310)
# with system site-packages; the ROS setup must be sourced BEFORE uvicorn
# starts or rclpy is not importable. First run: make bridge-venv.
BRIDGE_DIR  ?= $(HOME)/cap_ref/semantic-object/semantic_bridge
BRIDGE_VENV ?= $(HOME)/semantic-bridge-venv
bridge-venv:
	uv venv --python /usr/bin/python3.10 --system-site-packages $(BRIDGE_VENV)
	uv pip install --python $(BRIDGE_VENV)/bin/python -e "$(BRIDGE_DIR)[dev]"
bridge:
	@test -x $(BRIDGE_VENV)/bin/uvicorn || { echo "no bridge venv -- run: make bridge-venv"; exit 1; }
	source /opt/ros/$(ROS_DISTRO)/setup.bash && \
	cd $(BRIDGE_DIR) && \
	$(BRIDGE_VENV)/bin/uvicorn semantic_bridge.main:app --host 0.0.0.0 --port 8000

# Vite dev server for the map UI on port 3000, reachable from the laptop at
# http://<jetson-ip>:3000. Needs Node 20 (NodeSource; apt's 12.x cannot run
# Vite 5). The UI talks to the bridge at VITE_BACKEND_URL from
# semantic_map_ui/.env -- that must be the JETSON's address, not localhost,
# because the browser runs on the laptop. First run: make ui-deps.
UI_DIR ?= $(HOME)/cap_ref/semantic-object/semantic_map_ui
ui-deps:
	cd $(UI_DIR) && npm ci
ui:
	@test -d $(UI_DIR)/node_modules || { echo "no node_modules -- run: make ui-deps"; exit 1; }
	@test -f $(UI_DIR)/.env || echo "WARNING: $(UI_DIR)/.env missing; UI will look for the bridge on localhost"
	cd $(UI_DIR) && npm run dev -- --host

clean:
	rm -rf build install log
