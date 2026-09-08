#!/usr/bin/env python3
"""Send one command to the ESP32 over /dev/esp32 and print the reply.

Bypasses ROS entirely. This is the script that answers "is it my code or my
hardware?", which is most of the debugging in a week like this, so it must keep
working when nothing else does -- no ROS running, nothing sourced.

    ./serial_probe.py e            # -> "1234 -5678"   accumulated counts
    ./serial_probe.py r            # -> "OK"           zero encoders + PID
    ./serial_probe.py "o 50 50"    # -> "OK"           raw PWM, ROBOT ON BLOCKS
    ./serial_probe.py "m 20 20"    # -> "OK"           closed loop, ON BLOCKS

The contract (reference/firmware-protocol.md in the docs repo): 57600 8N1, one
command letter plus space-separated arguments, terminated by CARRIAGE RETURN.
A bare \\n is ignored rather than answered, so a host sending CRLF gets one
reply and not two. Unknown commands reply "Invalid Command".

WHAT A REPLY DOES NOT PROVE

  * "OK" only means the command parsed. `o` and `m` answer before the motors
    have done anything, so OK from `m 20 20` is not evidence that the wheels
    turned, and certainly not that they turned *forward*. Watch the wheels --
    that is the whole point of steps 5.5-5.7 on Day 1.
  * A timeout does not separate a dead board from the wrong device on this
    symlink from a port somebody else is holding open. Run
    `sudo fuser -v /dev/esp32` first: on the old board a micro_ros_agent
    running as root produced a generic pyserial error that read exactly like a
    permissions fault and was not one.
  * Encoder counts from `e` are meaningless in isolation; only their *change*
    means anything. Use encoder_report.py for that.

ABOUT THE RESET -- EVERY PROBE REBOOTS THE BOARD

We clear DTR and RTS before open() so pyserial never drives EN and GPIO0
itself, and on this adapter it makes no difference: the kernel asserts both
lines as the tty is opened, and `stty -hupcl` does not change that. Measured
8 Sep 2026 on the CP210x -- three consecutive opens, a ~506-byte boot burst
and the banner every single time.

So the encoder counts restart from zero on every invocation. Two consequences
to know before trusting an output:

  * `./serial_probe.py e` twice cannot show accumulated motion -- each run
    counts from its own reset. The hand-spin checks (Day 1 steps 5.3 and 5.4)
    need ONE persistent connection: that is encoder_report.py, not this.
  * `r` on a fresh connect is very nearly a no-op.

The banner is "# boot reset=1 encoders=ok". Anything unreadable in front of it
is the ROM bootloader talking at 115200 regardless of our baud rate; that is
expected, not a fault. `encoders=FAIL` is a real fault -- the PCNT units would
not configure, which is a pin problem (Day 1 step 5.9).
"""

import argparse
import sys
import time

import serial

PORT = '/dev/esp32'
BAUD = 57600
REPLY_TIMEOUT = 2.0     # s; every reply in the protocol is immediate
DRAIN_SECS = 0.15       # s of quiet before we accept the port as settled
DRAIN_CAP = 1.0         # s; a port that never goes quiet is not the ESP32


def open_quietly(port, baud):
    """Open the port with DTR/RTS held low.

    pyserial stores dtr/rts set before open() and applies them during it, so
    it never drives EN or GPIO0 itself. The kernel still does -- see ABOUT THE
    RESET above -- so this reduces the line toggling, it does not stop the
    reboot.
    """
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def drain(ser):
    """Swallow whatever the board said before we arrived.

    Returns (bytes, streaming). `streaming` means the port was still talking
    when we gave up waiting for quiet -- see check_not_streaming: that is the
    lidar, not the ESP32, and without the cap this waits forever.
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


def report_streaming(pending, port):
    """Explain a port that talks without being asked.

    The ESP32 only speaks when spoken to: a banner on reset, then one line per
    command. A continuous stream at this baud is the YDLidar, whose frames are
    0xAA 0x55 headers at 115200 -- i.e. the udev symlinks are crossed and this
    name points at the wrong adapter. That failure is silent by construction,
    because both adapters are 10c4:ea60 with the same non-unique serial and the
    rules match on USB port path.
    """
    print(f'[warn] {port} never goes quiet: {len(pending)} bytes with no '
          f'command sent.', file=sys.stderr)
    if b'\xaa\x55' in pending or pending.count(b'\xaa') > 4:
        print('[warn] that looks like YDLidar frame headers (0xAA55). This is '
              'the lidar, not the ESP32 -- the symlinks are crossed.',
              file=sys.stderr)
    print('[warn] the other possibility is the historical runaway-output bug '
          '(see reference/firmware-protocol.md); that one only ever followed a '
          'command, so a port already streaming at connect points at the '
          'symlinks first.', file=sys.stderr)
    print('[warn] check: for d in /dev/ttyUSB*; do udevadm info -q path -n $d; '
          'done   and re-run `make udev`.', file=sys.stderr)


def read_reply(ser, timeout):
    """Read one non-empty, non-banner line. Returns None on timeout.

    Replies end in \\r\\n, so split on either and drop the blanks. Lines
    starting with '#' are the board talking about itself (the boot banner),
    never an answer to a command.
    """
    deadline = time.monotonic() + timeout
    buf = b''
    while time.monotonic() < deadline:
        buf += ser.read(64)
        while True:
            cut = min((buf.find(t) for t in (b'\r', b'\n') if t in buf),
                      default=-1)
            if cut < 0:
                break
            line, buf = buf[:cut].strip(), buf[cut + 1:]
            text = line.decode('utf-8', 'replace').strip()
            if text and not text.startswith('#'):
                return text
    return None


def main():
    ap = argparse.ArgumentParser(
        description='Send one raw command to the ESP32 and print the reply.')
    ap.add_argument('command',
                    help='the command, quoted if it has arguments: "m 20 20"')
    ap.add_argument('--port', default=PORT,
                    help=f'serial device (default {PORT})')
    ap.add_argument('--baud', type=int, default=BAUD,
                    help=f'baud rate (default {BAUD})')
    ap.add_argument('--timeout', type=float, default=REPLY_TIMEOUT,
                    help=f'seconds to wait for a reply (default {REPLY_TIMEOUT})')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='show what was already on the port before we sent')
    args = ap.parse_args()

    cmd = args.command.strip()
    if not cmd:
        print('empty command', file=sys.stderr)
        return 2

    try:
        ser = open_quietly(args.port, args.baud)
    except serial.SerialException as exc:
        print(f'cannot open {args.port}: {exc}', file=sys.stderr)
        print('check: sudo fuser -v %s   (must be empty)' % args.port,
              file=sys.stderr)
        print('check: id -nG | grep dialout', file=sys.stderr)
        return 1

    with ser:
        pending, streaming = drain(ser)
        if args.verbose and pending:
            shown = pending.decode('utf-8', 'replace').strip()
            print(f'[before] {shown!r}', file=sys.stderr)
        if streaming:
            report_streaming(pending, args.port)
        if b'encoders=FAIL' in pending:
            print('[warn] boot banner says encoders=FAIL -- PCNT would not '
                  'configure, see Day 1 step 5.9', file=sys.stderr)

        ser.write(cmd.encode() + b'\r')
        ser.flush()
        reply = read_reply(ser, args.timeout)

    if reply is None:
        print(f'no reply to {cmd!r} in {args.timeout:g}s', file=sys.stderr)
        return 1

    # Printed even when the port was streaming: on a crossed symlink the
    # "reply" is a slice of lidar frame, and seeing that is the diagnosis.
    print(reply)
    if reply == 'Invalid Command' or streaming:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
