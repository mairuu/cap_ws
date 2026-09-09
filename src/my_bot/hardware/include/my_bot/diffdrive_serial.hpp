// ros2_control hardware interface for the ESP32 base controller in ~/firmware.
//
// Speaks that firmware's line protocol over a serial port:
//
//   e                     -> "<left> <right>"   cumulative encoder counts
//   r                     -> "OK"               zero both counters
//   m <tick_l> <tick_r>   -> "OK"               closed-loop ticks per PID frame
//   u <Kp>:<Kd>:<Ki>:<Ko> -> "OK"               replace the PID gains
//
// The `m` command's units are ticks per *firmware PID frame*, not per second,
// so `loop_rate` here must match PID_RATE_HZ in the firmware's config.h or
// every commanded speed is scaled by the ratio between them.
//
// This exposes position and velocity state per wheel and takes a velocity
// command; diff_drive_controller does the kinematics and odometry on top.
//
// Encoder scaling is per wheel: set enc_counts_per_rev for a shared figure,
// or enc_counts_per_rev_left / enc_counts_per_rev_right to override either
// side independently.

#ifndef MY_BOT__DIFFDRIVE_SERIAL_HPP_
#define MY_BOT__DIFFDRIVE_SERIAL_HPP_

#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/logger.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "my_bot/serial_port.hpp"

namespace my_bot
{

class DiffDriveSerial : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  struct Wheel
  {
    std::string joint_name;
    long encoder = 0;   // cumulative counts as reported by the firmware
    double position = 0.0;  // rad
    double velocity = 0.0;  // rad/s
    double command = 0.0;   // rad/s, written by diff_drive_controller

    // Per-wheel, because hand-measured counts/rev differ measurably between
    // the two sides on this robot. Sharing one figure biases heading.
    int counts_per_rev = 2514;
    double rads_per_count = 0.0;

    void set_counts_per_rev(int counts);
  };

  struct Config
  {
    std::string device = "/dev/ttyUSB0";
    int baud_rate = 57600;
    int timeout_ms = 1000;
    double loop_rate = 30.0;
    // Opening the port toggles DTR/RTS, which on an ESP32 DevKit drives the
    // auto-reset circuit and reboots the board. Wait it out, then discard the
    // boot banner, before trusting any reply.
    int boot_delay_ms = 2000;
    bool set_pid_gains = false;
    int pid_p = 20;
    int pid_d = 12;
    int pid_i = 0;
    int pid_o = 50;
  };

  // Sends one command and collects its reply. Returns false on I/O failure or
  // timeout; `reply` is only meaningful when it returns true.
  bool exchange(const std::string & command, std::string & reply);

  // Fire-and-forget variant for commands whose "OK" we don't need to inspect.
  // The reply is still drained so it can't desynchronise the next exchange.
  bool send_and_drain(const std::string & command);

  bool stop_motors();

  Config cfg_;
  Wheel left_;
  Wheel right_;
  SerialPort serial_;

  // Encoder reads can occasionally time out on a busy USB-serial link. One
  // bad frame is survivable, a sustained run of them is not.
  int consecutive_read_failures_ = 0;

  rclcpp::Logger logger_ = rclcpp::get_logger("DiffDriveSerial");

  // Steady clock so throttled warnings keep their spacing even if sim time
  // or a system clock step is in play.
  rclcpp::Clock clock_{RCL_STEADY_TIME};
};

}  // namespace my_bot

#endif  // MY_BOT__DIFFDRIVE_SERIAL_HPP_
