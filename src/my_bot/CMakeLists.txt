cmake_minimum_required(VERSION 3.8)
project(my_bot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# ros2_control hardware interface for the real ESP32 base controller.
add_library(my_bot_hardware SHARED
  hardware/diffdrive_serial.cpp
  hardware/serial_port.cpp
)
target_compile_features(my_bot_hardware PUBLIC cxx_std_17)
target_include_directories(my_bot_hardware PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hardware/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(my_bot_hardware PUBLIC
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)
pluginlib_export_plugin_description_file(hardware_interface my_bot_hardware.xml)

install(TARGETS my_bot_hardware
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY hardware/include/ DESTINATION include)

ament_export_libraries(my_bot_hardware)
ament_export_include_directories(include)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

# Diagnostics meant to be run against a live robot, via `ros2 run my_bot ...`.
# setup_udev.sh is deliberately NOT here -- the Makefile calls it from the
# source tree because it must run before anything is built.
install(PROGRAMS
  scripts/check_scan_world_fixed.py
  scripts/calibrate_straight.py
  scripts/calibrate_correct.py
  scripts/calibrate_spin.py
  # Day-1 bring-up diagnostics. These talk to /dev/esp32 directly and do not
  # need a ROS graph, but installing them means `ros2 run my_bot motor_check.py`
  # works from anywhere instead of only from the source tree.
  scripts/serial_probe.py
  scripts/encoder_report.py
  scripts/motor_check.py
  scripts/boot_check.py
  # Day-2 gate checks. Neither commands motion; both need the ROS stack up.
  scripts/tf_check.py
  scripts/odom_check.py
  # Day-3 lidar checks. Listen only; need the lidar driver but no TF.
  scripts/scan_dropout_report.py
  scripts/check_scan_bearing.py
  # Sits in the e-stop path: launched by navigation.launch.py next to twist_mux.
  scripts/teleop_speed_guard.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
