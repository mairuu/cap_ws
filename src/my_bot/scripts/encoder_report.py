#!/usr/bin/env python3
"""Poll the ESP32's encoders at 10 Hz and print ticks, delta and rad/s.

The first thing to reach for when a wheel misbehaves. Works with NO ROS
running at all -- that is the point: it separates "the encoders are wrong"
from "ros2_control is wrong", and those get confused constantly.

    ./encoder_report.py                 # just watch
    ./encoder_report.py --zero          # send `r` first, start from 0
    ./encoder_report.py --seconds 20    # stop by itself

It only ever sends `e` (and `r` with --zero). It NEVER commands motion, so it
is safe to run with the robot on the ground -- though every use below wants it
on blocks anyway.

WHAT IT IS FOR, ON DAY 1

Steps 5.3 and 5.4: spin each wheel FORWARD by hand and watch its own column.

    count increases  -> that side's *_ENC_INVERT is right
    count decreases  -> flip LEFT_ENC_INVERT / RIGHT_ENC_INVERT in config.h
                        and reflash. Do this BEFORE the first `m` command.
    count stays 0    -> that encoder is not being read at all. If it is the
                        right wheel, this is the 23/22-vs-32/33 pin conflict
                        (Day 1 step 5.9), not a sign error.

Getting the sign wrong is the highest-consequence mistake in the build: the PID
then reads the error as growing while it pushes, and runs away to full PWM and
holds it there. On blocks that is noise. On the ground it is a wall.

WHAT THE NUMBERS MEAN, AND DO NOT

  * ticks are 4x quadrature counts straight off the PCNT units, accumulated
    since boot or since the last `r`. The absolute value means nothing; only
    the change does.
  * rad/s is derived from TICKS_PER_REV below, which is a *recovered*
    calibration (records/calibration.md), not something this script measures.
    If it is wrong, rad/s is wrong by the same factor everywhere and the ticks
    column is still trustworthy. Hand-spinning cannot check it either -- that
    needs calibrate_straight.py against a tape.
  * the two sides use slightly different divisors on purpose (2475 / 2470).
    They are the same parts; the residual split is tyre diameter, and it lands
    in wheel_radius, not here.
  * a rad/s of a hand-spun wheel is a rough number: the poll is 10 Hz, so a
    delta covers ~100 ms of an uneven push.

CONNECTING RESETS THE BOARD

Opening the port reboots the ESP32 and zeroes both counts, and there is no
avoiding it: clearing DTR/RTS before open() only stops pyserial driving those
lines, while the kernel asserts them as the tty is opened (measured 8 Sep 2026
-- three consecutive opens, the boot banner every time). So this script always
starts from 0, which makes --zero very nearly a no-op, and it is why the
hand-spin checks want this script rather than repeated serial_probe.py calls:
one connection, held open, counts that accumulate.

A large jump in BOTH columns at once, especially back toward zero, means the
board reset again mid-run rather than the wheels moving. The script says so
when it sees the boot banner go past.
"""

import argparse
import sys
import time

import serial

PORT = '/dev/esp32'
BAUD = 57600
RATE_HZ = 10.0
REPLY_TIMEOUT = 0.5
DRAIN_SECS = 0.25      # s of quiet before we call the port settled
DRAIN_CAP = 2.0        # s; a port that never goes quiet is not the ESP32

# records/calibration.md, recovered 2026-09-08. Near-equal on purpose.
TICKS_PER_REV_LEFT = 2475
TICKS_PER_REV_RIGHT = 2470

TWO_PI = 6.283185307179586


def open_quietly(port, baud):
    """Open with DTR/RTS held low, so pyserial does not drive EN and GPIO0.

    The board reboots on open anyway -- see CONNECTING RESETS THE BOARD.
    """
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.05
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def drain(ser):
    """Swallow the boot burst. Returns (bytes, still_streaming).

    Our firmware goes quiet once booted, so waiting for QUIET is what
    separates the two cases -- a byte count cannot, because the reset burst is
    ~500 bytes on its own:

        quiet within the cap -> the ESP32, booted, waiting for a command
        never goes quiet     -> the YDLidar, whose stream never stops

    The second is easy to hit silently: both adapters are 10c4:ea60 reporting
    the same non-unique serial `0001`, so the udev rules match on USB port
    path, and one wrong answer to `make udev` crosses the two names.
    """
    pending = b''
    started = quiet_since = time.monotonic()
    while time.monotonic() - quiet_since < DRAIN_SECS:
        if time.monotonic() - started > DRAIN_CAP:
            return pending, True
        chunk = ser.read(256)
        if chunk:
            pending += chunk
            quiet_since = time.monotonic()
    return pending, False


def read_line(ser, timeout):
    """One stripped line, or None on timeout."""
    deadline = time.monotonic() + timeout
    buf = b''
    while time.monotonic() < deadline:
        buf += ser.read(64)
        cut = min((buf.find(t) for t in (b'\r', b'\n') if t in buf), default=-1)
        if cut >= 0:
            return buf[:cut].decode('utf-8', 'replace').strip()
    return None


def ask(ser, cmd, timeout=REPLY_TIMEOUT):
    """Send one command, return the first line that is an actual answer.

    Lines starting with '#' are the board talking about itself; the only one it
    ever emits is the boot banner, so seeing one mid-run means it rebooted.
    """
    ser.reset_input_buffer()
    ser.write(cmd.encode() + b'\r')
    ser.flush()
    while True:
        line = read_line(ser, timeout)
        if line is None:
            return None
        if line.startswith('#'):
            print(f'[warn] board reset mid-run: {line}', file=sys.stderr)
            continue
        if line:
            return line


def parse_counts(reply):
    """`e` answers "<left> <right>". Returns None if it did not."""
    parts = reply.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(
        description='Watch the ESP32 encoder counts with no ROS running.')
    ap.add_argument('--port', default=PORT, help=f'serial device (default {PORT})')
    ap.add_argument('--baud', type=int, default=BAUD, help=f'default {BAUD}')
    ap.add_argument('--rate', type=float, default=RATE_HZ,
                    help=f'polls per second (default {RATE_HZ:g})')
    ap.add_argument('--seconds', type=float, default=0.0,
                    help='stop after this long (default: until Ctrl-C)')
    ap.add_argument('--zero', action='store_true',
                    help='send `r` first, zeroing both encoders and the PID')
    args = ap.parse_args()

    try:
        ser = open_quietly(args.port, args.baud)
    except serial.SerialException as exc:
        print(f'cannot open {args.port}: {exc}', file=sys.stderr)
        print(f'check: sudo fuser -v {args.port}   (must be empty)',
              file=sys.stderr)
        print('check: id -nG | grep dialout', file=sys.stderr)
        return 1

    period = 1.0 / args.rate
    with ser:
        time.sleep(0.1)
        pending, streaming = drain(ser)
        if streaming:
            print(f'{args.port} never goes quiet ({len(pending)} bytes, no '
                  'command sent) -- that is the lidar, not the ESP32. The '
                  'symlinks are crossed; re-run `make udev`.', file=sys.stderr)
            return 1
        if b'encoders=FAIL' in pending:
            print('[warn] boot banner says encoders=FAIL -- the PCNT units '
                  'would not configure. That is a pin problem (Day 1 step '
                  '5.9), and the counts below will stay at zero.',
                  file=sys.stderr)
        ser.reset_input_buffer()

        if args.zero:
            reply = ask(ser, 'r')
            if reply != 'OK':
                print(f'`r` answered {reply!r}, expected OK', file=sys.stderr)
                return 1

        print(f'{"t":>7}  {"left":>10} {"dL":>7} {"rad/s":>8}   '
              f'{"right":>10} {"dR":>7} {"rad/s":>8}')

        started = time.monotonic()
        prev = None
        prev_t = None
        misses = 0
        next_poll = started
        try:
            while not args.seconds or time.monotonic() - started < args.seconds:
                now = time.monotonic()
                if now < next_poll:
                    time.sleep(next_poll - now)
                next_poll += period

                reply = ask(ser, 'e')
                now = time.monotonic()
                if reply is None:
                    misses += 1
                    print(f'{now - started:7.2f}  no reply to `e`'
                          f'  ({misses} so far)')
                    prev = None      # the next delta would span the gap
                    continue
                counts = parse_counts(reply)
                if counts is None:
                    misses += 1
                    print(f'{now - started:7.2f}  unparsable: {reply!r}')
                    prev = None
                    continue

                left, right = counts
                if prev is None:
                    dl = dr = 0
                    wl = wr = 0.0
                else:
                    dt = now - prev_t
                    dl, dr = left - prev[0], right - prev[1]
                    wl = dl * TWO_PI / TICKS_PER_REV_LEFT / dt
                    wr = dr * TWO_PI / TICKS_PER_REV_RIGHT / dt
                prev, prev_t = counts, now

                print(f'{now - started:7.2f}  {left:10d} {dl:7d} {wl:8.2f}   '
                      f'{right:10d} {dr:7d} {wr:8.2f}')
        except KeyboardInterrupt:
            print()

        if misses:
            print(f'{misses} poll(s) got no usable answer', file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
