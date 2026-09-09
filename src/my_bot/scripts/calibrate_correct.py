#!/usr/bin/env python3
"""Turn floor measurements from calibrate_straight.py into config changes.

Pure arithmetic, no ROS. Reads the CURRENT values out of
config/my_controllers.yaml and description/ros2_control.xacro so the suggested
numbers are always relative to what is actually installed.

TWO INDEPENDENT ERRORS, TWO INDEPENDENT KNOBS.

  scale     both wheels wrong by the same factor. Odom distance is wrong, the
            map comes out uniformly too big or too small, headings are fine.
            Fixed by wheel_radius. Needs --tape-distance.

  asymmetry the two wheels disagree. Odom yaw is biased, so the robot curves
            while odom reports straight (or the reverse). Fixed by the per-side
            radius multipliers. Needs --floor-lateral.

The asymmetry maths, for a run of length D with wheel separation L. Odom sees
heading change t_odom = (sr_hat - sl_hat)/L from estimated wheel arcs; the
floor shows t_phys. Writing the true arcs as s = k * s_hat,

    t_phys - t_odom = (D/L) * (k_r - k_l)      so   k_r - k_l = L*(t_phys - t_odom)/D

and t_phys comes from the lateral offset, which is what you can actually
measure: for near-constant curvature, lateral = D * t_phys / 2. The split is
applied symmetrically (k_l = 1 - d/2, k_r = 1 + d/2) so that correcting the
asymmetry does not disturb the distance scale you just calibrated.

    ros2 run my_bot calibrate_correct.py --distance 3.0 \
        --tape-distance 2.94 --odom-distance 3.00 --floor-lateral 0.08

--floor-lateral is signed: POSITIVE means the robot ended up LEFT of the line,
matching the sign convention calibrate_straight.py prints. Pass --odom-yaw if
the run reported a non-zero yaw drift (an --open-loop run will); it defaults to
0, which is what a converged closed-loop run gives.
"""

import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CONTROLLERS = os.path.join(PKG, 'config', 'my_controllers.yaml')
ROS2_CONTROL = os.path.join(PKG, 'description', 'ros2_control.xacro')


def read_scalar(path, key, default=None):
    """Pull `key: value` out of a YAML file without a yaml dependency."""
    pat = re.compile(r'^\s*%s\s*:\s*([0-9.eE+-]+)\s*(?:#.*)?$' % re.escape(key),
                     re.M)
    m = pat.search(open(path).read())
    if m:
        return float(m.group(1))
    if default is not None:
        return default
    sys.exit('ERROR: could not find %s in %s' % (key, path))


def read_param(path, name):
    """Pull <param name="...">value</param> out of the xacro."""
    m = re.search(r'<param\s+name="%s"\s*>\s*([0-9.eE+-]+)\s*</param>'
                  % re.escape(name), open(path).read())
    if not m:
        sys.exit('ERROR: could not find param %s in %s' % (name, path))
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(
        description='Compute wheel_radius and per-side corrections from a '
                    'calibrate_straight.py run plus floor measurements.')
    ap.add_argument('--distance', type=float,
                    help='the run length you asked for, in metres. Required '
                         'only for the straight-line analyses.')
    ap.add_argument('--odom-distance', type=float,
                    help='"straight-line chord" printed by calibrate_straight')
    ap.add_argument('--tape-distance', type=float,
                    help='metres between the floor marks')
    ap.add_argument('--floor-lateral', type=float,
                    help='metres the end mark sits off the line, + = LEFT')
    ap.add_argument('--odom-yaw', type=float, default=0.0,
                    help='"yaw drift" in degrees printed by calibrate_straight '
                         '(default 0, a converged closed-loop run)')
    ap.add_argument('--floor-heading', type=float,
                    help='measured physical heading change in degrees, if you '
                         'have it. More accurate than inferring it from the '
                         'lateral offset; overrides that inference.')
    ap.add_argument('--spin-odom-deg', type=float,
                    help='"rotated" printed by calibrate_spin.py')
    ap.add_argument('--spin-error-deg', type=float,
                    help='how far past its start heading the robot actually '
                         'finished, + = OVER-rotated')
    ap.add_argument('--spin-lever', type=float,
                    help='instead of --spin-error-deg: distance in metres '
                         'from the centre of rotation to a marked point')
    ap.add_argument('--spin-offset', type=float,
                    help='how far sideways that marked point ended up, in '
                         'metres, + = over-rotated')
    args = ap.parse_args()

    if args.spin_lever and args.spin_offset is not None:
        args.spin_error_deg = math.degrees(
            math.atan2(args.spin_offset, args.spin_lever))

    radius = read_scalar(CONTROLLERS, 'wheel_radius')
    sep = read_scalar(CONTROLLERS, 'wheel_separation')
    mult_l = read_scalar(CONTROLLERS, 'left_wheel_radius_multiplier', 1.0)
    mult_r = read_scalar(CONTROLLERS, 'right_wheel_radius_multiplier', 1.0)
    cpr_l = read_param(ROS2_CONTROL, 'enc_counts_per_rev_left')
    cpr_r = read_param(ROS2_CONTROL, 'enc_counts_per_rev_right')

    print('=' * 66)
    print('CURRENTLY INSTALLED')
    print('  wheel_radius                   %.5f m' % radius)
    print('  wheel_separation               %.5f m' % sep)
    print('  left/right radius multiplier   %.5f / %.5f' % (mult_l, mult_r))
    print('  enc_counts_per_rev  L / R      %.0f / %.0f' % (cpr_l, cpr_r))
    print()

    did_something = False

    # --- scale ------------------------------------------------------------
    if args.tape_distance and args.odom_distance:
        did_something = True

        # The tape measures the CHORD between two floor marks. Odom reports the
        # ARC the wheels rolled. Those are the same number only on a straight
        # run: a path that curved by t has arc/chord = t / (2*sin(t/2)), which
        # is +0.7 % at 24 deg and grows fast. Comparing the two directly makes
        # the wheels look smaller than they are, so correct the chord up to an
        # arc before taking the ratio.
        rolled = args.tape_distance
        curve_note = None
        if args.floor_lateral:
            t = 2.0 * args.floor_lateral / args.tape_distance
            if abs(t) > 1e-6:
                rolled = args.tape_distance * t / (2.0 * math.sin(t / 2.0))
                curve_note = (math.degrees(t), rolled - args.tape_distance)

        ratio = rolled / args.odom_distance
        new_radius = radius * ratio
        err = 100.0 * (args.odom_distance - rolled) / rolled
        print('SCALE  (wheel_radius)')
        if curve_note:
            print('  tape chord %.4f m, but the path curved %.1f deg, so the'
                  % (args.tape_distance, curve_note[0]))
            print('  wheels actually rolled %.4f m (+%.0f mm of arc)'
                  % (rolled, 1000.0 * curve_note[1]))
        print('  odom %.4f m vs rolled %.4f m  ->  odom over-reports by %+.2f %%'
              % (args.odom_distance, rolled, err))
        print('  corrected wheel_radius = %.5f * %.5f' % (radius, ratio))
        print()
        print('    config/my_controllers.yaml   wheel_radius: %.5f' % new_radius)
        print('    description/robot_core.xacro wheel_radius = %.5f' % new_radius)
        print()
        print('  Over a 10 m room that %+.2f %% is %+.0f cm of map error.'
              % (err, 10.0 * err))
        if curve_note and abs(curve_note[0]) > 5.0:
            print()
            print('  *** DO NOT APPLY THIS YET. The run curved %.0f deg, which'
                  % abs(curve_note[0]))
            print('  means the wheels were slipping and scrubbing sideways for')
            print('  most of it, and the arc correction above only undoes the')
            print('  geometry, not the scrub. Fix the ASYMMETRY below first,')
            print('  then re-run straight and re-measure the radius. ***')
        print()
    elif args.tape_distance or args.odom_distance:
        print('SCALE  skipped: need BOTH --tape-distance and --odom-distance.')
        print()

    # --- asymmetry --------------------------------------------------------
    if args.floor_lateral is not None:
        if not args.distance:
            ap.error('--floor-lateral needs --distance')
        did_something = True
        d = args.distance
        if args.floor_heading is not None:
            t_phys = math.radians(args.floor_heading)
            src = 'measured heading'
        else:
            # Near-constant curvature: lateral = D * theta / 2.
            t_phys = 2.0 * args.floor_lateral / d
            src = 'inferred from lateral offset'
        t_odom = math.radians(args.odom_yaw)

        delta = sep * (t_phys - t_odom) / d
        k_l = 1.0 - delta / 2.0
        k_r = 1.0 + delta / 2.0

        print('ASYMMETRY  (per-side radius multipliers)')
        print('  floor lateral %+.4f m over %.2f m' % (args.floor_lateral, d))
        print('  physical heading change  %+.3f deg   (%s)'
              % (math.degrees(t_phys), src))
        print('  odom heading change      %+.3f deg' % args.odom_yaw)
        print('  uncorrected yaw bias     %+.3f deg over this run'
              % math.degrees(t_phys - t_odom))
        print('  k_r - k_l = L*(t_phys - t_odom)/D = %+.6f' % delta)
        print()
        print('  OPTION A, the controller (preferred: one place, and')
        print('  diff_drive_controller applies it to BOTH odometry and the')
        print('  wheel commands, so the robot drives straighter too):')
        print()
        print('    config/my_controllers.yaml, under diff_cont:')
        print('      left_wheel_radius_multiplier:  %.6f' % (mult_l * k_l))
        print('      right_wheel_radius_multiplier: %.6f' % (mult_r * k_r))
        print()
        print('  OPTION B, the hardware interface. Same correction pushed into')
        print('  the tick conversion instead. Do ONE of these, never both:')
        print()
        print('    description/ros2_control.xacro:')
        print('      enc_counts_per_rev_left    %.0f  ->  %.0f'
              % (cpr_l, round(cpr_l / k_l)))
        print('      enc_counts_per_rev_right   %.0f  ->  %.0f'
              % (cpr_r, round(cpr_r / k_r)))
        print()
        if abs(math.degrees(t_phys - t_odom)) < 1.0:
            print('  NOTE: this bias is under 1 deg over the whole run. It is')
            print('  real and worth fixing, but it is NOT big enough to be why')
            print('  a map comes out badly wrong: slam_toolbox corrects yaw by')
            print('  scan matching, and absorbs errors far larger than this.')
            print()

    # --- wheel_separation -------------------------------------------------
    if args.spin_odom_deg and args.spin_error_deg is not None:
        did_something = True
        # Odom yaw = (right arc - left arc) / separation, so odom yaw scales as
        # 1/separation. The robot was driven until ODOM read spin_odom_deg; it
        # physically turned that plus the residual you measured. Hence
        #   true/param = odom/physical.
        physical = args.spin_odom_deg + args.spin_error_deg
        ratio = args.spin_odom_deg / physical
        new_sep = sep * ratio
        print('HEADING  (wheel_separation)')
        print('  odom rotated       %+.2f deg' % args.spin_odom_deg)
        print('  robot rotated      %+.2f deg   (odom %+.2f, residual %+.2f)'
              % (physical, args.spin_odom_deg, args.spin_error_deg))
        print('  odom under-reports yaw by %+.2f %%'
              % (100.0 * (physical - args.spin_odom_deg) / physical))
        print('  corrected wheel_separation = %.5f * %.5f' % (sep, ratio))
        print()
        print('    config/my_controllers.yaml   wheel_separation: %.5f'
              % new_sep)
        print('    description/robot_core.xacro wheel_offset_y = %.5f'
              % (new_sep / 2.0))
        print()
        print('  A %+.2f %% heading error is %+.1f deg over a single 90 deg'
              % (100.0 * (physical - args.spin_odom_deg) / physical,
                 90.0 * (physical - args.spin_odom_deg) / physical))
        print('  turn. That is the error slam_toolbox has to absorb by scan')
        print('  matching every time the robot corners.')
        print()
    elif args.spin_odom_deg or args.spin_error_deg is not None:
        print('HEADING  skipped: need BOTH --spin-odom-deg and '
              '--spin-error-deg (or --spin-lever with --spin-offset).')
        print()

    if not did_something:
        ap.error('nothing to compute: pass --tape-distance/--odom-distance, '
                 '--floor-lateral, and/or --spin-odom-deg with '
                 '--spin-error-deg')

    print('Rebuild with `make build` before the next run.')
    print('=' * 66)


if __name__ == '__main__':
    main()
