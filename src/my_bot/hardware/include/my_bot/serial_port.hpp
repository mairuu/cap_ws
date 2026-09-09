// Minimal blocking serial port on top of termios.
//
// Deliberately not a general-purpose library: it does exactly what the
// ESP32 base controller's line protocol needs -- write a CR-terminated
// command, read one LF-terminated reply -- and nothing else. That keeps
// my_bot free of any external serial dependency.

#ifndef MY_BOT__SERIAL_PORT_HPP_
#define MY_BOT__SERIAL_PORT_HPP_

#include <string>

namespace my_bot
{

class SerialPort
{
public:
  SerialPort() = default;
  ~SerialPort();

  SerialPort(const SerialPort &) = delete;
  SerialPort & operator=(const SerialPort &) = delete;

  // Opens and configures the port for 8N1, raw mode, no flow control.
  // On failure returns false and fills `error` with a human-readable reason.
  bool open(const std::string & device, int baud_rate, int timeout_ms, std::string & error);

  void close();
  bool is_open() const { return fd_ >= 0; }

  // Drives the DTR and RTS modem lines. Both are set in a single ioctl so
  // they never transiently disagree -- which matters on boards that wire
  // them into a reset circuit.
  bool set_control_lines(bool dtr, bool rts);

  // Discards anything already sitting in the kernel's input buffer.
  void flush_input();

  // Writes `command` followed by a carriage return, which is what the
  // firmware's parser terminates on.
  bool write_line(const std::string & command);

  // Reads up to the next newline. The trailing CR/LF are stripped.
  // Returns false on timeout or I/O error.
  bool read_line(std::string & line);

private:
  int fd_ = -1;
  int timeout_ms_ = 1000;
};

}  // namespace my_bot

#endif  // MY_BOT__SERIAL_PORT_HPP_
