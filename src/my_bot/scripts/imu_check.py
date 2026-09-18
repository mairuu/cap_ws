#!/usr/bin/env python3
"""Characterise the GY-521 (MPU6050) through the firmware's `i` command.

Talks to /dev/esp32 directly. THE ROS STACK MUST BE DOWN -- ros2_control_node
holds the port exclusively and this cannot open it while `make real` runs.
Nothing here commands the motors.

    ./imu_check.py                     # rest characterisation, 10 s
    ./imu_check.py --seconds 30        # longer bias/noise estimate
    ./imu_check.py --axes              # ...then the hands-on axis identification

WHAT IT MEASURES, AND WHY EACH NUMBER MATTERS

1. Boot banner `imu=ok`. `FAIL whoami=0x00` is a bus fault (wiring, pull-ups,
   address); `0x70` / `0x71` is a GY-521 carrying an MPU6500 / MPU9250 die,
   which has a different register map -- the firmware's WHO_AM_I constant is
   not the fix for that.

2. At rest, flat, motors OFF: mean and sigma of all six axes. Expect the z
   accel near +16384 (+1 g at the default +/-2 g range) and everything else
   near zero. The GYRO BIAS is what the EKF subtracts; the GYRO SIGMA is what
   its covariance should say. Both go in records/calibration.md with the
   date. The MPU6050's zero-rate offset is specified to +/-20 deg/s, so a
   bias of a few deg/s is normal and is NOT a fault -- it is a number to
   record, not to chase.

3. Round-trip time of one `i` exchange. The host will issue this inside its
   30 Hz control loop, on the same serial line as the encoder read, so the
   budget is what matters: encoder + motor command already cost ~6 ms of the
   33 ms frame. p95 above ~10 ms means poll the IMU at a divisor of the loop
   rate rather than every cycle.

4. --axes: which RAW axis is which BODY axis, with sign. This is the IMU's
   `reversion` / `inverted` moment. The lidar's two flags cost this project
   days precisely because they were inherited rather than measured, and a
   sign error in yaw rate makes the EKF fight the wheels instead of correct
   them. So: measure it. Do not read it off the silkscreen.

   Three moves, each prompted:
     yaw   -- turn the robot CCW (left, seen from above) ~90 deg. The gyro
              axis that integrates to ~+90 deg is body +z.
     pitch -- tip the nose DOWN ~30 deg and hold. The accel axis that swings
              NEGATIVE is body +x (an axis pointing down reads -g).
     roll  -- tip the LEFT side down ~30 deg and hold. The accel axis that
              swings negative is body +y.
   Body +z is also read from the rest sample (the axis reading +1 g). The
   gyro and accel share axes on this chip, so yaw and rest must name the
   same raw axis with the same sign -- a disagreement is a measurement
   error, redo it.

   The three axes are assembled into a rotation matrix, checked to be a
   proper rotation (determinant +1 -- a -1 means one sign is wrong), and
   printed as the `rpy` for imu_link's joint in description/imu.xacro.
   Mounting orientation belongs in TF, not in code (D-10).

READING THE RESULT

  imu=ok, az ~ +1 g, gyro sigma well under 1 deg/s, p95 under 10 ms
      -> record the numbers, proceed to the axis identification
  az far from 1 g with ax/ay near zero
      -> the range is not +/-2 g. initIMU() writes ACCEL_CONFIG explicitly
         since 18 Sep; an older firmware is flashed.
  ax or ay well off zero at rest
      -> the board is not mounted level. Fine for yaw; the tilt shows up
         as a constant in the accel and the EKF is not using accel.
  gyro sigma of several deg/s with motors OFF
      -> the DLPF is not engaged (older firmware), or the breakout is not
         rigidly mounted and is picking up the floor.
  `IMU Error` on every poll after the banner said ok
      -> the bus is dropping under load. Check the wire run and try
         IMU_I2C_FREQ_HZ = 100000 in config.h before anything else.
"""

import argparse
import math
import statistics
import sys
import time

import serial

PORT = '/dev/esp32'
BAUD = 57600
REPLY_TIMEOUT = 0.5
DRAIN_SECS = 0.25
DRAIN_CAP = 2.0

# Firmware writes GYRO_CONFIG=0 / ACCEL_CONFIG=0 on every boot (mpu6050.cpp).
ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
G = 9.80665

AXES = ('x', 'y', 'z')


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
    """Send one command, return (reply, round_trip_seconds)."""
    ser.reset_input_buffer()
    t0 = time.monotonic()
    ser.write(cmd.encode() + b'\r')
    ser.flush()
    while True:
        line = read_line(ser, timeout)
        if line is None:
            return None, time.monotonic() - t0
        if line.startswith('#'):
            print(f'[warn] board reset mid-run: {line}', file=sys.stderr)
            continue
        if line:
            return line, time.monotonic() - t0


def parse_imu(reply):
    parts = (reply or '').split()
    if len(parts) != 6:
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def banner_status(pending):
    text = pending.decode('utf-8', 'replace')
    for line in text.splitlines():
        if line.startswith('# boot'):
            return line.strip()
    return None


def sample(ser, seconds, rate, label):
    """Poll `i` for `seconds`. Returns (rows, round_trips, errors)."""
    period = 1.0 / rate
    rows, rtts = [], []
    errors = 0
    started = time.monotonic()
    next_poll = started
    last_print = started
    while time.monotonic() - started < seconds:
        now = time.monotonic()
        if now < next_poll:
            time.sleep(next_poll - now)
        next_poll += period
        reply, rtt = ask(ser, 'i')
        vals = parse_imu(reply)
        if vals is None:
            errors += 1
            if reply == 'IMU Error' and errors <= 3:
                print(f'  [{label}] IMU Error at t={time.monotonic() - started:.2f}s',
                      file=sys.stderr)
            elif reply != 'IMU Error':
                print(f'  [{label}] unusable reply {reply!r}', file=sys.stderr)
            continue
        rows.append((time.monotonic() - started, vals))
        rtts.append(rtt)
        if time.monotonic() - last_print > 1.0:
            last_print = time.monotonic()
            ax, ay, az, gx, gy, gz = vals
            print(f'  [{label}] t={rows[-1][0]:5.1f}s  '
                  f'a=({ax:6d},{ay:6d},{az:6d})  g=({gx:5d},{gy:5d},{gz:5d})  '
                  f'rtt={rtt * 1e3:4.1f} ms', file=sys.stderr)
    return rows, rtts, errors


def column_stats(rows):
    cols = list(zip(*[r[1] for r in rows]))
    means = [statistics.fmean(c) for c in cols]
    sigmas = [statistics.pstdev(c) if len(c) > 1 else 0.0 for c in cols]
    return means, sigmas


def report_rest(rows, rtts, errors, seconds):
    means, sigmas = column_stats(rows)
    n = len(rows)
    print()
    print(f'REST  {n} samples over {seconds:.0f} s, {errors} errors, '
          f'{n / seconds:.1f} Hz achieved')
    print(f'  {"axis":>6} {"mean raw":>10} {"sigma raw":>10}   '
          f'{"mean":>10} {"sigma":>10}')
    for i, ax in enumerate(AXES):
        print(f'  {"a" + ax:>6} {means[i]:10.1f} {sigmas[i]:10.1f}   '
              f'{means[i] / ACCEL_LSB_PER_G:8.4f} g {sigmas[i] / ACCEL_LSB_PER_G:8.4f} g')
    for i, ax in enumerate(AXES):
        j = 3 + i
        print(f'  {"g" + ax:>6} {means[j]:10.1f} {sigmas[j]:10.1f}   '
              f'{means[j] / GYRO_LSB_PER_DPS:6.3f} °/s {sigmas[j] / GYRO_LSB_PER_DPS:6.3f} °/s')

    rtts_ms = sorted(r * 1e3 for r in rtts)
    p50 = rtts_ms[len(rtts_ms) // 2]
    p95 = rtts_ms[int(len(rtts_ms) * 0.95)]
    print(f'  `i` round trip: p50 {p50:.1f} ms, p95 {p95:.1f} ms, '
          f'max {rtts_ms[-1]:.1f} ms')

    verdicts = []
    a_g = [m / ACCEL_LSB_PER_G for m in means[:3]]
    up = max(range(3), key=lambda i: abs(a_g[i]))
    mag = math.sqrt(sum(v * v for v in a_g))
    if abs(mag - 1.0) > 0.15:
        verdicts.append(f'FAIL  |a| = {mag:.3f} g, expected ~1.0: range is not '
                        f'+/-2 g, or the board is being moved')
    else:
        verdicts.append(f'PASS  |a| = {mag:.3f} g')
    tilt = math.degrees(math.acos(min(1.0, abs(a_g[up]) / mag)))
    verdicts.append(f'{"PASS" if tilt < 8 else "WARN"}  up is raw '
                    f'{"+" if a_g[up] > 0 else "-"}{AXES[up]}, '
                    f'mount tilt {tilt:.1f} deg from level')
    gsig = max(sigmas[3:]) / GYRO_LSB_PER_DPS
    verdicts.append(f'{"PASS" if gsig < 1.0 else "WARN"}  gyro sigma '
                    f'{gsig:.3f} °/s (motors off; >1 means DLPF off or a '
                    f'loose mount)')
    verdicts.append(f'{"PASS" if p95 < 10.0 else "WARN"}  p95 round trip '
                    f'{p95:.1f} ms against a ~10 ms budget inside the 30 Hz '
                    f'frame')
    if errors:
        verdicts.append(f'WARN  {errors} errors: the bus is dropping reads')
    for v in verdicts:
        print('  ' + v)

    bias_dps = [m / GYRO_LSB_PER_DPS for m in means[3:]]
    bias_rads = [math.radians(b) for b in bias_dps]
    sig_rads = [math.radians(s / GYRO_LSB_PER_DPS) for s in sigmas[3:]]
    print()
    print('  For records/calibration.md and the hardware params:')
    print(f'    gyro bias  (raw)   x {means[3]:8.1f}  y {means[4]:8.1f}  z {means[5]:8.1f}')
    print(f'    gyro bias  (rad/s) x {bias_rads[0]:+.5f}  y {bias_rads[1]:+.5f}  z {bias_rads[2]:+.5f}')
    print(f'    gyro sigma (rad/s) x {sig_rads[0]:.5f}  y {sig_rads[1]:.5f}  z {sig_rads[2]:.5f}')
    print(f'    -> ekf.yaml imu0 vyaw variance ~ {sig_rads[2] ** 2:.2e}  '
          f'(sigma^2; inflate x4 for driving)')
    return means, sigmas, (up, 1 if a_g[up] > 0 else -1)


def prompt(text):
    print()
    print(text)
    try:
        input('  press Enter to start sampling: ')
    except EOFError:
        return False
    return True


def dominant(delta):
    i = max(range(3), key=lambda k: abs(delta[k]))
    others = sorted(abs(delta[k]) for k in range(3) if k != i)
    clear = abs(delta[i]) > 3 * others[-1] if others[-1] > 0 else True
    return i, (1 if delta[i] > 0 else -1), clear


def rpy_from_matrix(r):
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.atan2(-r[2][0], math.hypot(r[0][0], r[1][0]))
    roll = math.atan2(r[2][1], r[2][2])
    yaw = math.atan2(r[1][0], r[0][0])
    return roll, pitch, yaw


def fmt_angle(a):
    for k, name in ((0, '0'), (1, '${pi/2}'), (-1, '${-pi/2}'),
                    (2, '${pi}'), (-2, '${-pi}')):
        if abs(a - k * math.pi / 2) < 0.05:
            return name
    return f'{a:.4f}'


def identify_axes(ser, rest_means, rest_up, rate):
    bias = rest_means[3:]
    body = {}  # body axis -> (raw index, sign)

    body['z'] = rest_up
    print(f'\n  rest says body +z is raw {"+" if rest_up[1] > 0 else "-"}{AXES[rest_up[0]]}')

    if not prompt('YAW: turn the robot CCW (to its LEFT, seen from above) by '
                  'about 90 deg,\n  smoothly, within the 5 s window.'):
        return None
    rows, _, _ = sample(ser, 5.0, rate, 'yaw')
    if len(rows) < 10:
        print('  too few samples')
        return None
    integ = [0.0, 0.0, 0.0]
    prev_t = rows[0][0]
    for t, vals in rows[1:]:
        dt = t - prev_t
        prev_t = t
        for k in range(3):
            integ[k] += (vals[3 + k] - bias[k]) / GYRO_LSB_PER_DPS * dt
    i, s, clear = dominant(integ)
    print(f'  integrated: x {integ[0]:+7.1f}  y {integ[1]:+7.1f}  z {integ[2]:+7.1f} deg')
    print(f'  -> body +z (yaw) is raw {"+" if s > 0 else "-"}{AXES[i]}'
          f'{"" if clear else "   (NOT CLEAR -- other axes moved too, redo)"}')
    if (i, s) != rest_up:
        print('  ** DISAGREES with the rest sample. Gyro and accel share axes on '
              'this chip;\n     one of the two moves was wrong. Redo before trusting '
              'anything below.')
    gyro_z = (i, s)

    if not prompt('PITCH: tip the NOSE DOWN by about 30 deg and HOLD it there '
                  'through the 3 s window.'):
        return None
    rows, _, _ = sample(ser, 3.0, rate, 'pitch')
    means, _ = column_stats(rows)
    delta = [means[k] - rest_means[k] for k in range(3)]
    i, s, clear = dominant(delta)
    body['x'] = (i, -s)  # axis pointing down reads -g
    print(f'  accel delta: x {delta[0] / ACCEL_LSB_PER_G:+.3f}  y {delta[1] / ACCEL_LSB_PER_G:+.3f}  '
          f'z {delta[2] / ACCEL_LSB_PER_G:+.3f} g')
    print(f'  -> body +x (forward) is raw {"+" if -s > 0 else "-"}{AXES[i]}'
          f'{"" if clear else "   (NOT CLEAR, redo)"}')

    if not prompt('ROLL: tip the LEFT side DOWN by about 30 deg and HOLD it '
                  'through the 3 s window.'):
        return None
    rows, _, _ = sample(ser, 3.0, rate, 'roll')
    means, _ = column_stats(rows)
    delta = [means[k] - rest_means[k] for k in range(3)]
    i, s, clear = dominant(delta)
    body['y'] = (i, -s)
    print(f'  accel delta: x {delta[0] / ACCEL_LSB_PER_G:+.3f}  y {delta[1] / ACCEL_LSB_PER_G:+.3f}  '
          f'z {delta[2] / ACCEL_LSB_PER_G:+.3f} g')
    print(f'  -> body +y (left) is raw {"+" if -s > 0 else "-"}{AXES[i]}'
          f'{"" if clear else "   (NOT CLEAR, redo)"}')

    # R[body][raw]: row b is body axis b expressed in raw coordinates.
    r = [[0.0] * 3 for _ in range(3)]
    for b, name in enumerate(AXES):
        i, s = body[name]
        r[b][i] = float(s)
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    used = {body[n][0] for n in AXES}

    print()
    print('AXIS MAP (raw -> body)')
    for name in AXES:
        i, s = body[name]
        print(f'  body +{name}  =  raw {"+" if s > 0 else "-"}{AXES[i]}')
    if len(used) < 3:
        print('  FAIL  two body axes map to the same raw axis. One move was '
              'misread; redo --axes.')
        return None
    if det < 0:
        print(f'  FAIL  determinant {det:+.0f}: this is a reflection, not a '
              f'rotation. One sign is wrong; redo --axes.')
        return None
    print(f'  PASS  proper rotation (det {det:+.0f})')

    # The joint rpy expresses the IMU (child) frame in the body (parent)
    # frame, R_body_raw. Its columns are the raw axes in body coordinates,
    # and its rows are the body axes in raw coordinates -- which is exactly
    # what was measured above, so r IS that matrix. Not its transpose: for
    # a 90 deg yaw the two differ by the sign of the angle, which is the
    # error this script exists to prevent.
    roll, pitch, yaw = rpy_from_matrix(r)
    print()
    print('  For description/imu.xacro:')
    print(f'    <origin xyz="..." rpy="{fmt_angle(roll)} {fmt_angle(pitch)} {fmt_angle(yaw)}"/>')
    print(f'    (rpy = {math.degrees(roll):.1f} {math.degrees(pitch):.1f} '
          f'{math.degrees(yaw):.1f} deg; measured, not read off the silkscreen)')
    return body, (roll, pitch, yaw)


def main():
    ap = argparse.ArgumentParser(
        description='Characterise the MPU6050 through the `i` command. '
                    'ROS stack must be down. Commands no motion.')
    ap.add_argument('--port', default=PORT, help=f'serial device (default {PORT})')
    ap.add_argument('--baud', type=int, default=BAUD, help=f'default {BAUD}')
    ap.add_argument('--seconds', type=float, default=10.0,
                    help='rest sample length (default 10)')
    ap.add_argument('--rate', type=float, default=30.0,
                    help='poll rate; 30 matches the control loop (default 30)')
    ap.add_argument('--axes', action='store_true',
                    help='after the rest sample, run the hands-on yaw/pitch/'
                         'roll identification')
    args = ap.parse_args()

    try:
        ser = open_quietly(args.port, args.baud)
    except serial.SerialException as e:
        print(f'cannot open {args.port}: {e}\n'
              f'  - is `make real` running? it holds the port exclusively\n'
              f'  - is /dev/esp32 missing? the adapter moved sockets: `make udev`',
              file=sys.stderr)
        return 1

    try:
        time.sleep(0.1)
        pending, streaming = drain(ser)
        if streaming:
            print(f'{args.port} never goes quiet ({len(pending)} bytes, no '
                  'command sent) -- that is the lidar, not the ESP32. The '
                  'symlinks are crossed; re-run `make udev`.', file=sys.stderr)
            return 1
        banner = banner_status(pending)
        if banner:
            print(f'banner: {banner}')
            if 'imu=FAIL' in banner:
                print('  imu=FAIL. whoami=0x0 is a bus fault (wiring, address, '
                      'pull-ups); 0x70/0x71 is an MPU6500/MPU9250 die on the '
                      'breakout, not an MPU6050.', file=sys.stderr)
                return 1
        else:
            print('no boot banner seen (PRINT_BOOT_BANNER off, or the board did '
                  'not reset on open); continuing on the strength of `i` alone')
        ser.reset_input_buffer()

        reply, rtt = ask(ser, 'i')
        if parse_imu(reply) is None:
            print(f'first `i` got {reply!r} -- nothing to measure', file=sys.stderr)
            return 1

        print(f'\nREST: keep the robot still and level for {args.seconds:.0f} s, '
              f'motors off.')
        rows, rtts, errors = sample(ser, args.seconds, args.rate, 'rest')
        if len(rows) < 5:
            print('too few samples to report', file=sys.stderr)
            return 1
        means, sigmas, up = report_rest(rows, rtts, errors, args.seconds)

        if args.axes:
            if identify_axes(ser, means, up, args.rate) is None:
                return 1
        return 0
    finally:
        ser.close()


if __name__ == '__main__':
    sys.exit(main())
