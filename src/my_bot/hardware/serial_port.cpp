#include "my_bot/serial_port.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

namespace my_bot
{

namespace
{

// Only the rates plausibly used by this firmware. Anything else is rejected
// rather than silently falling back to a wrong speed.
bool to_speed_t(int baud_rate, speed_t & out)
{
  switch (baud_rate) {
    case 9600: out = B9600; return true;
    case 19200: out = B19200; return true;
    case 38400: out = B38400; return true;
    case 57600: out = B57600; return true;
    case 115200: out = B115200; return true;
    default: return false;
  }
}

// A single reply line is never anywhere near this long; the cap just stops
// a stuck port from growing the string without bound.
constexpr size_t MAX_LINE_LEN = 256;

}  // namespace

SerialPort::~SerialPort() { close(); }

bool SerialPort::open(
  const std::string & device, int baud_rate, int timeout_ms, std::string & error)
{
  close();

  speed_t speed;
  if (!to_speed_t(baud_rate, speed)) {
    error = "unsupported baud rate " + std::to_string(baud_rate);
    return false;
  }

  timeout_ms_ = timeout_ms;

  // O_NOCTTY: this port must never become the process's controlling terminal.
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
  if (fd_ < 0) {
    error = "cannot open " + device + ": " + std::strerror(errno);
    return false;
  }

  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    error = "tcgetattr failed: " + std::string(std::strerror(errno));
    close();
    return false;
  }

  cfmakeraw(&tty);
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);

  tty.c_cflag |= (CLOCAL | CREAD);   // ignore modem lines, enable receiver
  tty.c_cflag &= ~CSTOPB;            // one stop bit
  tty.c_cflag &= ~CRTSCTS;           // no hardware flow control

  // HUPCL would drop DTR on close, which on a DevKit board yanks the ESP32's
  // auto-reset line and reboots it every time the controller shuts down.
  tty.c_cflag &= ~HUPCL;

  // VMIN=0 with VTIME set makes read() a bounded blocking call: it returns
  // whatever has arrived, or 0 if the timer expires first. VTIME is in
  // deciseconds and must be at least 1 to avoid a pure non-blocking spin.
  tty.c_cc[VMIN] = 0;
  int vtime = timeout_ms / 100;
  tty.c_cc[VTIME] = static_cast<cc_t>(vtime < 1 ? 1 : (vtime > 255 ? 255 : vtime));

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    error = "tcsetattr failed: " + std::string(std::strerror(errno));
    close();
    return false;
  }

  // Opening a tty asserts DTR and RTS. On an ESP32 DevKit those lines drive
  // the auto-reset circuit: RTS asserted pulls EN low and holds the chip in
  // reset, so the board stays mute no matter what we send it. Drop both and
  // leave the board alone; callers that want a reset ask for one explicitly.
  if (!set_control_lines(false, false)) {
    error = "could not clear DTR/RTS: " + std::string(std::strerror(errno));
    close();
    return false;
  }

  flush_input();
  return true;
}

bool SerialPort::set_control_lines(bool dtr, bool rts)
{
  if (fd_ < 0) {
    return false;
  }

  int status = 0;
  if (ioctl(fd_, TIOCMGET, &status) != 0) {
    return false;
  }

  if (dtr) {
    status |= TIOCM_DTR;
  } else {
    status &= ~TIOCM_DTR;
  }

  if (rts) {
    status |= TIOCM_RTS;
  } else {
    status &= ~TIOCM_RTS;
  }

  return ioctl(fd_, TIOCMSET, &status) == 0;
}

void SerialPort::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void SerialPort::flush_input()
{
  if (fd_ >= 0) {
    tcflush(fd_, TCIFLUSH);
  }
}

bool SerialPort::write_line(const std::string & command)
{
  if (fd_ < 0) {
    return false;
  }

  const std::string payload = command + "\r";
  size_t written = 0;

  while (written < payload.size()) {
    const ssize_t n = ::write(fd_, payload.data() + written, payload.size() - written);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    written += static_cast<size_t>(n);
  }

  return true;
}

bool SerialPort::read_line(std::string & line)
{
  if (fd_ < 0) {
    return false;
  }

  line.clear();

  // termios VTIME bounds each individual read(); this bounds the whole line,
  // so a peer dribbling bytes forever can't stall the control loop.
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

  while (std::chrono::steady_clock::now() < deadline) {
    char c;
    const ssize_t n = ::read(fd_, &c, 1);

    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }

    if (n == 0) {
      continue;  // VTIME expired with nothing to show; the deadline decides.
    }

    if (c == '\n') {
      return true;
    }

    if (c != '\r' && line.size() < MAX_LINE_LEN) {
      line.push_back(c);
    }
  }

  return false;
}

}  // namespace my_bot
