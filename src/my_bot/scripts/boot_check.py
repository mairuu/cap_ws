#!/usr/bin/env python3
"""Reset the ESP32 repeatedly and check it comes up clean every time.

Day 1 step 5.8. The worry is GPIO12 -- the MTDI strapping pin, which selects
the flash voltage (VDD_SDIO) at reset and is also wired to
RIGHT_MOTOR_BACKWARD. If it is ever read high at reset the chip picks 1.8 V for
a 3.3 V flash and fails to boot. That presents as an intermittently dead board,
not as a motor fault, which is why it gets its own step.

    ./boot_check.py                 # 10 resets, banner + protocol each time
    ./boot_check.py --cycles 30     # chase an intermittent
    ./boot_check.py --raw           # show the ROM garbage too

WHAT THIS IS AND IS NOT A SUBSTITUTE FOR

It drives EN through RTS -- the auto-reset circuit every devkit has -- which is
a **chip reset**, and the strapping pins are re-latched on a chip reset exactly
as they are on power-on. The ESP32 cannot even tell the two apart: an EN reset
reports ESP_RST_POWERON, which is why the banner says `reset=1` either way.
So for the GPIO12 question specifically this is the same test, run more times
than anyone would do by hand.

**It is not a power cycle.** It does not re-run the supply ramp, so it cannot
see a brown-out as the motor rail comes up, or a regulator that only misbehaves
from cold. Do a couple of real ones as well -- this replaces the tedium, not
the last word.

WHAT COUNTS AS A CLEAN BOOT

  banner present         `# boot reset=1 encoders=ok`
  encoders=ok            the PCNT units configured
  exactly 4 pullup errors  the left encoder on input-only 34/35. 4x quadrature
                         gives it two PCNT channels, each pulling up a pulse
                         pin and a control pin. A FIFTH means something moved
                         onto an input-only pad; ZERO means the encoder setup
                         did not run at all.
  `e` answers two ints   the protocol is alive, not just the banner
  counts are 0 0         a fresh boot has not counted anything

A missing banner is the failure being hunted. A banner with `encoders=FAIL` is
a different failure -- pins, step 5.9 -- and is reported separately, because
booting-but-broken and not-booting have nothing to do with each other.
"""

import argparse
import sys
import time

import serial

PORT = '/dev/esp32'
BAUD = 57600
BANNER_WAIT = 3.0       # s to wait for the banner after releasing EN
QUIET = 0.30            # s of silence that ends the boot burst
REPLY_TIMEOUT = 1.0
PULLUP_EXPECTED = 4


def open_quietly(port, baud):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.05
    ser.dtr = False         # GPIO0 stays high -> boots the app, not the ROM loader
    ser.rts = False
    ser.open()
    return ser


def hard_reset(ser):
    """Pulse EN via RTS, leaving GPIO0 alone so the app runs.

    DTR is held false throughout. Pulling it low as well is what puts the chip
    into download mode, and that would look exactly like a boot failure here.
    """
    ser.dtr = False
    ser.rts = True          # EN low -- chip held in reset
    time.sleep(0.12)
    ser.reset_input_buffer()
    ser.rts = False         # EN released -- boots
    return time.monotonic()


def collect(ser, deadline_s):
    """Read until the port goes quiet, or the deadline passes."""
    out = b''
    started = last = time.monotonic()
    while time.monotonic() - started < deadline_s:
        chunk = ser.read(256)
        if chunk:
            out += chunk
            last = time.monotonic()
        elif out and time.monotonic() - last > QUIET:
            break
    return out


def read_reply(ser, timeout):
    deadline = time.monotonic() + timeout
    buf = b''
    while time.monotonic() < deadline:
        buf += ser.read(64)
        while True:
            cut = min((buf.find(t) for t in (b'\r', b'\n') if t in buf), default=-1)
            if cut < 0:
                break
            line, buf = buf[:cut].strip(), buf[cut + 1:]
            text = line.decode('utf-8', 'replace').strip()
            if text and not text.startswith('#'):
                return text
    return None


def ask(ser, cmd, timeout=REPLY_TIMEOUT):
    ser.reset_input_buffer()
    ser.write(cmd.encode() + b'\r')
    ser.flush()
    return read_reply(ser, timeout)


def analyse(blob):
    """Pull the meaningful lines out of a boot burst.

    Everything before the banner at our baud is the ROM bootloader logging at
    115200, so it decodes as garbage. That is expected and is not evidence of
    anything -- drop it rather than showing it.
    """
    text = blob.decode('utf-8', 'replace')
    banner = None
    pullups = 0
    for line in text.replace('\r', '\n').split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('# boot'):
            banner = line
        elif 'gpio_pullup_en' in line:
            pullups += 1
    return banner, pullups


def main():
    ap = argparse.ArgumentParser(
        description='Reset the ESP32 repeatedly and check every boot (Day 1 §5.8).')
    ap.add_argument('--port', default=PORT, help=f'serial device (default {PORT})')
    ap.add_argument('--baud', type=int, default=BAUD, help=f'default {BAUD}')
    ap.add_argument('--cycles', type=int, default=10,
                    help='how many resets (default 10; the checklist asks for '
                         '5, and these are cheap)')
    ap.add_argument('--raw', action='store_true',
                    help='also dump the ROM bootloader garbage')
    args = ap.parse_args()

    try:
        ser = open_quietly(args.port, args.baud)
    except serial.SerialException as exc:
        print(f'cannot open {args.port}: {exc}', file=sys.stderr)
        return 1

    print(f'{args.cycles} EN resets on {args.port}. This re-latches the '
          'strapping pins exactly as a\npower-on does -- but it does not '
          're-run the supply ramp, so do a couple of real\npower cycles too.\n')
    print(f'{"#":>3}  {"boot":>6}  {"banner":<28} {"PU":>3}  {"e":<12} {"r":<4} verdict')

    failures = []
    with ser:
        time.sleep(0.2)
        ser.reset_input_buffer()
        for i in range(1, args.cycles + 1):
            t0 = hard_reset(ser)
            blob = collect(ser, BANNER_WAIT)
            dt = time.monotonic() - t0
            banner, pullups = analyse(blob)

            if args.raw:
                print(f'    raw: {blob!r}')

            counts = ask(ser, 'e')
            zeroed = ask(ser, 'r')

            problems = []
            if banner is None:
                problems.append('no banner')
            elif 'encoders=ok' not in banner:
                problems.append('encoders not ok')
            if banner is not None and pullups != PULLUP_EXPECTED:
                problems.append(f'{pullups} pullup errors, expected {PULLUP_EXPECTED}')
            parts = (counts or '').split()
            if len(parts) != 2 or not all(p.lstrip('-').isdigit() for p in parts):
                problems.append('`e` did not answer two integers')
            elif parts != ['0', '0']:
                problems.append(f'fresh boot already at {counts}')
            if zeroed != 'OK':
                problems.append('`r` did not answer OK')

            verdict = 'ok' if not problems else '; '.join(problems)
            if problems:
                failures.append((i, verdict))
            print(f'{i:3d}  {dt:5.2f}s  {(banner or "-- none --"):<28} '
                  f'{pullups:3d}  {(counts or "--"):<12} {(zeroed or "--"):<4} {verdict}')

    print()
    if not failures:
        print(f'  {args.cycles}/{args.cycles} clean. GPIO12 did not misbehave '
              'across any reset.')
        print('  Still owed: a couple of real power cycles, for the supply ramp.')
        return 0

    print(f'  {len(failures)}/{args.cycles} FAILED:', file=sys.stderr)
    for i, why in failures:
        print(f'    cycle {i}: {why}', file=sys.stderr)
    print('\n  A missing banner across some resets and not others is the GPIO12 '
          'strapping\n  problem: MTDI read high at reset picks 1.8 V for a 3.3 V '
          'flash. It is wired to\n  RIGHT_MOTOR_BACKWARD, so look at what holds '
          'that line during reset.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
