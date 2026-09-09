#include "my_bot/diffdrive_serial.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <unordered_map>
#include <thread>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace my_bot
{

namespace
{

// Reads a hardware <param> if present, otherwise leaves the default alone.
// Every parameter in this interface is optional so the xacro only has to
// spell out what actually differs from the defaults.
std::string param_or(
  const std::unordered_map<std::string, std::string> & params,
  const std::string & key, const std::string & fallback)
{
  const auto it = params.find(key);
  return it == params.end() ? fallback : it->second;
}

// Consecutive failed encoder reads tolerated before the interface gives up
// and reports an error to the controller manager. At 30 Hz this is ~1 s.
constexpr int MAX_READ_FAILURES = 30;

}  // namespace

void DiffDriveSerial::Wheel::set_counts_per_rev(int counts)
{
  counts_per_rev = counts;
  rads_per_count = (2.0 * M_PI) / static_cast<double>(counts);
}

hardware_interface::CallbackReturn DiffDriveSerial::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  const auto & params = info_.hardware_parameters;

  try {
    left_.joint_name = param_or(params, "left_wheel_name", "left_wheel_joint");
    right_.joint_name = param_or(params, "right_wheel_name", "right_wheel_joint");
    cfg_.device = param_or(params, "device", cfg_.device);
    cfg_.baud_rate = std::stoi(param_or(params, "baud_rate", "57600"));
    cfg_.timeout_ms = std::stoi(param_or(params, "timeout_ms", "1000"));
    cfg_.loop_rate = std::stod(param_or(params, "loop_rate", "30"));
    cfg_.boot_delay_ms = std::stoi(param_or(params, "boot_delay_ms", "2000"));

    // enc_counts_per_rev is the shared figure; the per-wheel parameters
    // override it individually and are what you want once you've measured
    // each side separately.
    const std::string shared_counts = param_or(params, "enc_counts_per_rev", "2514");
    left_.set_counts_per_rev(
      std::stoi(param_or(params, "enc_counts_per_rev_left", shared_counts)));
    right_.set_counts_per_rev(
      std::stoi(param_or(params, "enc_counts_per_rev_right", shared_counts)));

    // PID gains are only pushed to the firmware when all four are given;
    // otherwise the firmware keeps whatever it booted with.
    if (params.count("pid_p") && params.count("pid_d") &&
      params.count("pid_i") && params.count("pid_o"))
    {
      cfg_.pid_p = std::stoi(params.at("pid_p"));
      cfg_.pid_d = std::stoi(params.at("pid_d"));
      cfg_.pid_i = std::stoi(params.at("pid_i"));
      cfg_.pid_o = std::stoi(params.at("pid_o"));
      cfg_.set_pid_gains = true;
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(logger_, "Bad hardware parameter: %s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (left_.counts_per_rev <= 0 || right_.counts_per_rev <= 0) {
    RCLCPP_FATAL(logger_, "encoder counts per revolution must be positive");
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (cfg_.loop_rate <= 0.0) {
    RCLCPP_FATAL(logger_, "loop_rate must be positive");
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (info_.joints.size() != 2) {
    RCLCPP_FATAL(
      logger_, "Expected exactly 2 joints, got %zu", info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }

  for (const auto & joint : info_.joints) {
    if (joint.command_interfaces.size() != 1 ||
      joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
    { 
      RCLCPP_FATAL(
        logger_, "Joint '%s' needs exactly one velocity command interface",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    bool has_position = false;
    bool has_velocity = false;
    for (const auto & state : joint.state_interfaces) {
      has_position |= (state.name == hardware_interface::HW_IF_POSITION);
      has_velocity |= (state.name == hardware_interface::HW_IF_VELOCITY);
    }
    if (!has_position || !has_velocity) {
      RCLCPP_FATAL(
        logger_, "Joint '%s' needs both position and velocity state interfaces",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.name != left_.joint_name && joint.name != right_.joint_name) {
      RCLCPP_FATAL(
        logger_,
        "Joint '%s' matches neither left_wheel_name ('%s') nor right_wheel_name ('%s')",
        joint.name.c_str(), left_.joint_name.c_str(), right_.joint_name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(
    logger_,
    "Configured for %s @ %d baud, %d/%d counts per rev (L/R), %.1f Hz firmware frame",
    cfg_.device.c_str(), cfg_.baud_rate, left_.counts_per_rev, right_.counts_per_rev,
    cfg_.loop_rate);

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> DiffDriveSerial::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;

  for (Wheel * wheel : {&left_, &right_}) {
    interfaces.emplace_back(
      wheel->joint_name, hardware_interface::HW_IF_POSITION, &wheel->position);
    interfaces.emplace_back(
      wheel->joint_name, hardware_interface::HW_IF_VELOCITY, &wheel->velocity);
  }

  return interfaces;
}

std::vector<hardware_interface::CommandInterface> DiffDriveSerial::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;

  for (Wheel * wheel : {&left_, &right_}) {
    interfaces.emplace_back(
      wheel->joint_name, hardware_interface::HW_IF_VELOCITY, &wheel->command);
  }

  return interfaces;
}

bool DiffDriveSerial::exchange(const std::string & command, std::string & reply)
{
  if (!serial_.write_line(command)) {
    return false;
  }
  return serial_.read_line(reply);
}

bool DiffDriveSerial::send_and_drain(const std::string & command)
{
  std::string reply;
  return exchange(command, reply);
}

bool DiffDriveSerial::stop_motors()
{
  // Worth noting: the firmware's AUTO_STOP watchdog is currently commented
  // out, so nothing on the robot stops the motors if this process dies. This
  // explicit stop on deactivate is the only thing standing between a clean
  // shutdown and a robot that keeps driving.
  return send_and_drain("m 0 0");
}

hardware_interface::CallbackReturn DiffDriveSerial::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  std::string error;
  if (!serial_.open(cfg_.device, cfg_.baud_rate, cfg_.timeout_ms, error)) {
    RCLCPP_FATAL(logger_, "Serial open failed: %s", error.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Put the board into a known state rather than hoping the port open left it
  // in one. On a DevKit, RTS pulls EN low and DTR pulls GPIO0 low. Holding
  // DTR deasserted (GPIO0 high) while pulsing RTS gives a plain reset into the
  // application; letting DTR go low across the same edge is what drops the
  // chip into the ROM download mode instead, where it never answers us.
  serial_.set_control_lines(false, true);   // EN low: held in reset
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  serial_.set_control_lines(false, false);  // release: boot the application

  // Wait out the boot, then discard the ROM chatter and banner. The ROM logs
  // at 115200 regardless of our baud rate, so it arrives as garbage bytes.
  std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.boot_delay_ms));
  serial_.flush_input();

  // Confirm the board is actually talking before trusting anything else.
  // A first attempt can still land mid-banner, so give it a few tries.
  bool responding = false;
  for (int attempt = 1; attempt <= 5 && !responding; ++attempt) {
    std::string reply;
    long ignored_left = 0;
    long ignored_right = 0;
    if (exchange("e", reply)) {
      std::istringstream ss(reply);
      responding = static_cast<bool>(ss >> ignored_left >> ignored_right);
    }
    if (!responding) {
      RCLCPP_WARN(logger_, "No usable encoder reply yet (attempt %d/5)", attempt);
      serial_.flush_input();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
  }

  if (!responding) {
    RCLCPP_FATAL(
      logger_, "No reply from %s -- is the firmware running and the port free?",
      cfg_.device.c_str());
    serial_.close();
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (cfg_.set_pid_gains) {
    std::ostringstream ss;
    ss << "u " << cfg_.pid_p << ":" << cfg_.pid_d << ":"
       << cfg_.pid_i << ":" << cfg_.pid_o;
    if (!send_and_drain(ss.str())) {
      RCLCPP_FATAL(logger_, "Firmware did not answer the PID gain update");
      serial_.close();
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  // Zeroing the counters here means wheel position starts at 0 and the first
  // read() doesn't see a step change from whatever the board had accumulated.
  if (!send_and_drain("r")) {
    RCLCPP_FATAL(logger_, "Firmware did not acknowledge the encoder reset");
    serial_.close();
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Only the live state resets here; joint names and the calibration figures
  // come from on_init and must survive a reconfigure.
  for (Wheel * wheel : {&left_, &right_}) {
    wheel->encoder = 0;
    wheel->position = 0.0;
    wheel->velocity = 0.0;
    wheel->command = 0.0;
  }
  consecutive_read_failures_ = 0;

  RCLCPP_INFO(logger_, "Connected to base controller on %s", cfg_.device.c_str());
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn DiffDriveSerial::on_cleanup(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (serial_.is_open()) {
    stop_motors();
    serial_.close();
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn DiffDriveSerial::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!serial_.is_open()) {
    RCLCPP_FATAL(logger_, "Cannot activate: serial port is not open");
    return hardware_interface::CallbackReturn::ERROR;
  }

  left_.command = 0.0;
  right_.command = 0.0;
  consecutive_read_failures_ = 0;

  if (!stop_motors()) {
    RCLCPP_FATAL(logger_, "Firmware did not acknowledge the initial stop");
    return hardware_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(logger_, "Activated");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn DiffDriveSerial::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (serial_.is_open() && !stop_motors()) {
    RCLCPP_ERROR(logger_, "Firmware did not acknowledge the stop on deactivate");
  }

  RCLCPP_INFO(logger_, "Deactivated");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type DiffDriveSerial::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  std::string reply;
  if (!exchange("e", reply)) {
    if (++consecutive_read_failures_ >= MAX_READ_FAILURES) {
      RCLCPP_ERROR(
        logger_, "No encoder reply for %d consecutive cycles; giving up",
        consecutive_read_failures_);
      return hardware_interface::return_type::ERROR;
    }
    RCLCPP_WARN_THROTTLE(
      logger_, clock_, 1000, "Encoder read timed out; holding previous state");
    return hardware_interface::return_type::OK;
  }

  // Expected shape is "<left> <right>". Anything else means we're out of sync
  // with the firmware -- most likely leftover boot output -- so drop the frame
  // rather than integrating garbage into odometry.
  long left_counts = 0;
  long right_counts = 0;
  std::istringstream ss(reply);
  if (!(ss >> left_counts >> right_counts)) {
    if (++consecutive_read_failures_ >= MAX_READ_FAILURES) {
      RCLCPP_ERROR(logger_, "Encoder replies unparseable for %d cycles; giving up",
        consecutive_read_failures_);
      return hardware_interface::return_type::ERROR;
    }
    RCLCPP_WARN_THROTTLE(
      logger_, clock_, 1000, "Unparseable encoder reply '%s'; dropping frame",
      reply.c_str());
    return hardware_interface::return_type::OK;
  }

  consecutive_read_failures_ = 0;

  const double dt = period.seconds();

  left_.encoder = left_counts;
  right_.encoder = right_counts;

  const double left_prev = left_.position;
  const double right_prev = right_.position;

  left_.position = static_cast<double>(left_counts) * left_.rads_per_count;
  right_.position = static_cast<double>(right_counts) * right_.rads_per_count;

  // A zero or negative period would come from a clock glitch; keep the last
  // velocity rather than dividing by it.
  if (dt > 0.0) {
    left_.velocity = (left_.position - left_prev) / dt;
    right_.velocity = (right_.position - right_prev) / dt;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type DiffDriveSerial::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // rad/s -> counts/s -> counts per firmware PID frame, which is what `m` wants.
  // Each wheel uses its own calibration figure.
  const long left_ticks =
    std::lround(left_.command / left_.rads_per_count / cfg_.loop_rate);
  const long right_ticks =
    std::lround(right_.command / right_.rads_per_count / cfg_.loop_rate);

  std::ostringstream ss;
  ss << "m " << left_ticks << " " << right_ticks;

  if (!send_and_drain(ss.str())) {
    RCLCPP_ERROR(logger_, "Failed to write motor command");
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

}  // namespace my_bot

PLUGINLIB_EXPORT_CLASS(my_bot::DiffDriveSerial, hardware_interface::SystemInterface)
