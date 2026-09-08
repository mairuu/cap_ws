#!/usr/bin/env python3
"""Drive the wheels open-loop and check the encoders agree with the motors.

This is Day 1 steps 5.5, 5.6 and 5.7, in that order, in one held-open
connection. ROBOT ON BLOCKS.

    ./motor_check.py                    # 5.5 + 5.6: drive `o`, watch, auto-stop
    ./motor_check.py --reverse          # same, backwards
    ./motor_check.py --closed-loop      # 5.7 as well, but only if 5.5 passed

WHY THIS AND NOT HAND-SPINNING

The checklist reaches 5.3/5.4 by spinning each wheel by hand and watching `e`.
That tests the encoder against your arm. What actually burns the robot out is
the encoder disagreeing with the *motor*: the PID pushes, reads the error as
growing, and winds to full PWM and holds it. So drive the motor and watch its
own encoder -- same measurement, one fewer assumption, and it is the exact
quantity 5.7 is dangerous about.

`o` bypasses the PID entirely (firmware-protocol.md), so nothing here can run
away no matter which way the signs point. That is what makes it safe to do
before 5.7 rather than after.

WHAT IT CANNOT TELL YOU

**Whether the wheels turn the right way round.** It only sees that motor and
encoder agree with each other. If both wheels drive backwards on positive PWM,
the counts still agree and this script prints PASS -- the robot will simply
reverse on every forward command. Only your eyes catch that, and the fix is
swapping that motor's FORWARD/BACKWARD pins, not touching `*_ENC_INVERT`.

So: watch the wheels while it runs. The script reports agreement; you report
direction.

It also cannot check the calibration -- ticks/rev is a recovered number
(records/calibration.md) and hand-driving cannot confirm it. Only
calibrate_straight.py against a tape does that.

READING THE RESULT

  both sides AGREE           -> signs are right, 5.7 is safe to run
  a side DISAGREES           -> flip that side's LEFT_ENC_INVERT /
                                RIGHT_ENC_INVERT in config.h and reflash
                                BEFORE any `m` command
  a side reads zero, motor   -> that encoder is not being read. On the right
  audibly turning               wheel this is the 23/22-vs-32/33 pin conflict
                                (step 5.9), not a sign error.
  neither side moves         -> motor supply is off, or the enables are low.
                                Encoders read zero for a reason that has
                                nothing to do with encoders.

AUTO-STOP (5.6)

The firmware stops the motors 2000 ms after the last `o`/`m`. This drives once
and then keeps polling past that deadline, so the last moving sample says when
it actually stopped. If it never stops, auto-stop is broken and the whole
`cmd_vel`-timeout safety story downstream is broken with it -- do not go on to
Day 2. The script sends `o 0 0` on the way out regardless, including on Ctrl-C
and on any exception, so a broken auto-stop still ends with the wheels stopped.
"""

import argparse
import sys
import time

import serial

PORT = '/dev/esp32'
BAUD = 57600
REPLY_TIMEOUT = 0.5
RATE_HZ = 10.0
DRAIN_SECS = 0.25
DRAIN_CAP = 2.0

AUTO_STOP_MS = 2000     # firmware-protocol.md; what 5.6 is checking for
PID_RATE_HZ = 30.0      # `m` is ticks per PID frame, and the frame is 1/30 s

# records/calibration.md, recovered 2026-09-08.
TICKS_PER_REV_LEFT = 2475
TICKS_PER_REV_RIGHT = 2470

TWO_PI = 6.283185307179586

MOVED = 30          # ticks in one 100 ms poll that count as "turning".
                    # PWM 50 is roughly 250 ticks/poll, so this is well clear
                    # of both electrical noise and one-tick jitter.
RUNAWAY_FACTOR = 3.0    # x the commanded rate, for two consecutive polls,
                        # before we cut power during --closed-loop


def open_quietly(port, baud):
    """Open with DTR/RTS held low. The board still reboots -- see the others."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.05
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def drain(ser):
    """Swallow the boot burst. Returns (bytes, still_streaming)."""
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
    deadline = time.monotonic() + timeout
    buf = b''
    while time.monotonic() < deadline:
        buf += ser.read(64)
        cut = min((buf.find(t) for t in (b'\r', b'\n') if t in buf), default=-1)
        if cut >= 0:
            return buf[:cut].decode('utf-8', 'replace').strip()
    return None


def ask(ser, cmd, timeout=REPLY_TIMEOUT):
    """Send one command, return the first line that is an actual answer."""
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
    parts = (reply or '').split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def stop(ser):
    """Cut PWM. Called on every exit path, including exceptions."""
    try:
        ser.write(b'o 0 0\r')
        ser.flush()
    except Exception:
        pass


def sample(ser, seconds, rate, label):
    """Poll `e` for `seconds`, printing as it goes.

    Returns a list of (t_since_command, left, right) with t measured from the
    moment the drive command was acknowledged -- which is what makes the
    auto-stop timing meaningful rather than approximate.
    """
    print(f'{"t":>6}  {"left":>10} {"dL":>7} {"rad/s":>7}   '
          f'{"right":>10} {"dR":>7} {"rad/s":>7}    {label}')
    period = 1.0 / rate
    rows = []
    started = time.monotonic()
    prev = prev_t = None
    next_poll = started
    while time.monotonic() - started < seconds:
        now = time.monotonic()
        if now < next_poll:
            time.sleep(next_poll - now)
        next_poll += period

        reply = ask(ser, 'e')
        now = time.monotonic()
        counts = parse_counts(reply)
        if counts is None:
            print(f'{now - started:6.2f}  no usable answer: {reply!r}')
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
        rows.append((now - started, left, right, dl, dr))
        print(f'{now - started:6.2f}  {left:10d} {dl:7d} {wl:7.2f}   '
              f'{right:10d} {dr:7d} {wr:7.2f}')
    return rows


def verdict(rows, commanded_sign):
    """Per side: did it move, and did the count go the commanded way?

    Uses the total across the whole driven window rather than any single poll,
    so one dropped reply or a slow start cannot flip the answer.
    """
    if not rows:
        return {}
    net_l = rows[-1][1] - rows[0][1]
    net_r = rows[-1][2] - rows[0][2]
    out = {}
    for name, net in (('left', net_l), ('right', net_r)):
        if abs(net) < MOVED:
            out[name] = ('STILL', net)
        elif (net > 0) == (commanded_sign > 0):
            out[name] = ('AGREE', net)
        else:
            out[name] = ('DISAGREE', net)
    return out


def stopped_at(rows):
    """When the last movement was seen, in seconds since the drive command."""
    last = None
    for t, _l, _r, dl, dr in rows:
        if abs(dl) >= MOVED or abs(dr) >= MOVED:
            last = t
    return last


def main():
    ap = argparse.ArgumentParser(
        description='Drive open-loop and check encoder sign against motor '
                    'sign. ROBOT ON BLOCKS.')
    ap.add_argument('--port', default=PORT, help=f'serial device (default {PORT})')
    ap.add_argument('--baud', type=int, default=BAUD, help=f'default {BAUD}')
    ap.add_argument('--pwm', type=int, default=50,
                    help='open-loop PWM, 0..255 (default 50 -- enough to turn '
                         'a free wheel, low enough to be boring)')
    ap.add_argument('--reverse', action='store_true',
                    help='drive backwards instead')
    ap.add_argument('--seconds', type=float, default=3.5,
                    help='how long to watch, must exceed the 2 s auto-stop '
                         '(default 3.5)')
    ap.add_argument('--rate', type=float, default=RATE_HZ,
                    help=f'polls per second (default {RATE_HZ:g})')
    ap.add_argument('--closed-loop', action='store_true',
                    help='also run 5.7 (`m`), but ONLY if the open-loop check '
                         'above passed on both sides')
    ap.add_argument('--ticks', type=int, default=20,
                    help='closed-loop target, ticks per 1/30 s frame '
                         '(default 20)')
    args = ap.parse_args()

    if not 0 < args.pwm <= 255:
        print('--pwm must be 1..255', file=sys.stderr)
        return 2
    if args.seconds <= AUTO_STOP_MS / 1000.0:
        print(f'--seconds must exceed the {AUTO_STOP_MS/1000:g}s auto-stop, '
              'or 5.6 is not being tested at all', file=sys.stderr)
        return 2

    try:
        ser = open_quietly(args.port, args.baud)
    except serial.SerialException as exc:
        print(f'cannot open {args.port}: {exc}', file=sys.stderr)
        print(f'check: sudo fuser -v {args.port}   (must be empty)', file=sys.stderr)
        print('check: id -nG | grep dialout', file=sys.stderr)
        return 1

    pwm = -args.pwm if args.reverse else args.pwm
    rc = 0

    with ser:
        try:
            time.sleep(0.1)
            pending, streaming = drain(ser)
            if streaming:
                print(f'{args.port} never goes quiet ({len(pending)} bytes, no '
                      'command sent) -- that is the lidar, not the ESP32. The '
                      'symlinks are crossed; re-run `make udev`.', file=sys.stderr)
                return 1
            if b'encoders=FAIL' in pending:
                print('[warn] boot banner says encoders=FAIL -- the PCNT units '
                      'would not configure. Counts will stay at zero and this '
                      'check cannot mean anything. Go to step 5.9.',
                      file=sys.stderr)
                return 1
            ser.reset_input_buffer()

            if ask(ser, 'r') != 'OK':
                print('`r` did not answer OK', file=sys.stderr)
                return 1

            # ---- 5.5 / 5.6: open loop ----------------------------------
            print(f'\n=== 5.5  o {pwm} {pwm}   (open loop, PID bypassed) ===')
            print('WATCH THE WHEELS. This checks that each encoder agrees with '
                  'its own motor;\nonly you can see whether "forward" is '
                  'actually forward.\n')
            if ask(ser, f'o {pwm} {pwm}') != 'OK':
                print('`o` did not answer OK', file=sys.stderr)
                return 1
            rows = sample(ser, args.seconds, args.rate, f'o {pwm} {pwm}')
            stop(ser)

            v = verdict(rows, pwm)
            last = stopped_at(rows)
            print()
            for side in ('left', 'right'):
                state, net = v.get(side, ('NO DATA', 0))
                print(f'  {side:5s}  {state:8s}  net {net:+d} ticks')

            print()
            if last is None:
                print('  5.6  auto-stop: nothing ever moved, so this says '
                      'nothing about auto-stop.')
            elif last < args.seconds - 0.4:
                print(f'  5.6  auto-stop: last movement at {last:.2f}s '
                      f'(expected ~{AUTO_STOP_MS/1000:g}s)  PASS')
            else:
                print(f'  5.6  auto-stop: STILL MOVING at {last:.2f}s -- the '
                      'motors did not stop by themselves.')
                print('       DO NOT PROCEED. Every downstream safety story '
                      '(cmd_vel timeout,')
                print('       twist_mux e-stop) assumes this works.')
                rc = 1

            states = {v.get(s, ('NO DATA',))[0] for s in ('left', 'right')}
            if 'DISAGREE' in states:
                print('\n  A side DISAGREES: its encoder counts down while its '
                      'motor drives up.')
                print('  Flip that side\'s *_ENC_INVERT in config.h and '
                      'reflash BEFORE any `m`.')
                print('  This is the runaway condition. Do not "just try it".')
                rc = 1
            if 'STILL' in states:
                print('\n  A side did not move. If you HEARD that motor turn, '
                      'the encoder is not')
                print('  being read -- on the right wheel that is the '
                      '23/22-vs-32/33 pin conflict')
                print('  (step 5.9). If you heard nothing, look at the motor '
                      'supply and enables.')
                rc = 1

            if not args.closed_loop:
                if rc == 0:
                    print('\n  Open loop is clean. Re-run with --closed-loop '
                          'for 5.7 when you are ready.')
                return rc

            # ---- 5.7: closed loop, gated on the above ------------------
            if rc != 0:
                print('\n  Refusing --closed-loop: the open-loop check did not '
                      'pass. That check is the\n  whole reason 5.7 is safe to '
                      'run.', file=sys.stderr)
                return rc

            ticks = -args.ticks if args.reverse else args.ticks
            expect = abs(ticks) * PID_RATE_HZ          # ticks/s the PID targets
            print(f'\n=== 5.7  m {ticks} {ticks}   (closed loop, PID engaged) ===')
            print(f'  target ~{expect:.0f} ticks/s per side. Cutting power '
                  f'automatically above {RUNAWAY_FACTOR:g}x that.')
            print('  Hand on the power switch anyway.\n')
            if ask(ser, f'm {ticks} {ticks}') != 'OK':
                print('`m` did not answer OK', file=sys.stderr)
                stop(ser)
                return 1

            # Same poll loop, but watching for the wind-up rather than the
            # sign -- so it is inline rather than sample().
            period = 1.0 / args.rate
            started = time.monotonic()
            prev = prev_t = None
            hot = 0
            print(f'{"t":>6}  {"left":>10} {"dL":>7} {"tick/s":>8}   '
                  f'{"right":>10} {"dR":>7} {"tick/s":>8}')
            next_poll = started
            while time.monotonic() - started < args.seconds:
                now = time.monotonic()
                if now < next_poll:
                    time.sleep(next_poll - now)
                next_poll += period
                reply = ask(ser, 'e')
                now = time.monotonic()
                counts = parse_counts(reply)
                if counts is None:
                    print(f'{now - started:6.2f}  no usable answer: {reply!r}')
                    prev = None
                    continue
                left, right = counts
                if prev is None:
                    dl = dr = 0
                    rl = rr = 0.0
                else:
                    dt = now - prev_t
                    dl, dr = left - prev[0], right - prev[1]
                    rl, rr = dl / dt, dr / dt
                prev, prev_t = counts, now
                print(f'{now - started:6.2f}  {left:10d} {dl:7d} {rl:8.0f}   '
                      f'{right:10d} {dr:7d} {rr:8.0f}')
                if max(abs(rl), abs(rr)) > RUNAWAY_FACTOR * expect:
                    hot += 1
                    if hot >= 2:
                        stop(ser)
                        print(f'\n  RUNAWAY: {RUNAWAY_FACTOR:g}x the commanded '
                              'rate for two polls running. Power cut.',
                              file=sys.stderr)
                        print('  Cut the supply, flip that side\'s '
                              '*_ENC_INVERT, reflash, start again at 5.5.',
                              file=sys.stderr)
                        return 1
                else:
                    hot = 0
            stop(ser)
            print('\n  5.7 finished without a wind-up. Compare the settled '
                  f'tick/s against the ~{expect:.0f} target:')
            print('  well short on one side is a PID tuning question, not a '
                  'safety one.')

        except KeyboardInterrupt:
            stop(ser)
            print('\ninterrupted -- PWM zeroed', file=sys.stderr)
            return 130
        except Exception:
            stop(ser)
            raise

    return rc


if __name__ == '__main__':
    sys.exit(main())
