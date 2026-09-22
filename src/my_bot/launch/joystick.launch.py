# Gamepad teleop: joy_node (reads the pad) + teleop_twist_joy (turns it into a
# Twist). The keyboard equivalent of this file is just a `ros2 run` line in the
# Makefile; the joystick needs a launch file because it is two nodes and three
# layers of parameters.
#
# WHERE IT PUBLISHES IS AN ARGUMENT, because this robot has two teleop entry
# points and picking the wrong one is the difference between an e-stop and a
# fight. See the Makefile's teleop-joy / teleop-joy-nav targets:
#
#   cmd_vel_topic:=/diff_cont/cmd_vel_unstamped   standalone, no Nav2 running
#   cmd_vel_topic:=/cmd_vel_teleop_raw            during `make nav`/`make explore`
#
# The second one goes through teleop_speed_guard and then twist_mux at priority
# 100, which is what makes it outrank Nav2.
#
# ---------------------------------------------------------------------------
# THE DEADMAN IS NOT A HOLD-TO-STOP. Releasing it makes teleop_twist_joy send
# one zero Twist and then stop publishing. During an autonomous run that means
# twist_mux's 0.5 s teleop timeout expires and NAV2 GETS THE ROBOT BACK. The
# joystick is a momentary override, exactly like the keyboard -- to actually
# end an autonomous run, Ctrl-C `make explore` / `make nav`.
#
# Standalone it is safe in the other direction: nothing else publishes to
# /diff_cont/cmd_vel_unstamped, so silence means diff_drive_controller's
# cmd_vel_timeout halts the wheels.
#
# AUTOREPEAT IS LOAD-BEARING. joy_node republishes the pad state at
# autorepeat_rate even when nothing moves. Without it, holding the stick
# perfectly still produces no new /joy messages, teleop_twist_joy publishes
# nothing, and twist_mux times the human out and hands control back to Nav2
# mid-manoeuvre. 20 Hz is well inside the 0.5 s timeout.
# ---------------------------------------------------------------------------

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    joy_share = get_package_share_directory('teleop_twist_joy')

    use_sim_time = LaunchConfiguration('use_sim_time')
    joy_config = LaunchConfiguration('joy_config')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the /clock topic instead of wall time. true only '
                    'against launch_sim.launch.py.',
    )

    # Buttons and axes only -- the stock files also carry scales, and every one
    # of those scales is wrong for this robot (see below), so our own layer
    # overwrites them afterwards.
    #
    # Which file to use is a property of the PAD, not of the robot. xbox is the
    # default because that is what is plugged in here. To find the numbers for
    # an unlisted pad: run this, then `ros2 topic echo /joy` and watch which
    # index in buttons[] / axes[] moves.
    joy_config_arg = DeclareLaunchArgument(
        'joy_config',
        default_value='xbox',
        description='Basename of a teleop_twist_joy config supplying the '
                    'button and axis INDICES: xbox, ps3, ps5, atk3, xd3. Its '
                    'speed scales are overridden.',
    )

    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev',
        default_value='0',
        description="SDL device index, not a /dev path -- Humble's joy_node "
                    'enumerates through SDL2. `ros2 run joy '
                    'joy_enumerate_devices` lists what it can see.',
    )

    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel_teleop_raw',
        description='Where the Twist goes. Defaults to the guarded channel '
                    '(/cmd_vel_teleop_raw -> teleop_speed_guard -> twist_mux), '
                    'which is the one that overrides Nav2. Use '
                    '/diff_cont/cmd_vel_unstamped to drive with no nav stack.',
    )

    # THE STOCK SCALES DO NOT FIT THIS ROBOT. xbox.config.yaml asks for 0.7 m/s
    # and 1.5 m/s on turbo; diff_cont clamps at 0.30 (config/my_controllers.yaml,
    # D-18 as amended by D-26). Shipping them would mean the top 57% of stick
    # throw does nothing at all, which reads as a broken pad.
    #
    # Instead the two speeds this project actually has are put on the two
    # positions the pad already has:
    #
    #   stick alone   full deflection -> 0.10 m/s, the MAPPING speed
    #   stick + turbo full deflection -> 0.30 m/s, the ceiling, for repositioning
    #
    # 0.10 is not a preference. The X3 Pro sweeps 360 deg in ~86 ms and
    # slam_toolbox does not deskew, so each scan shears by (speed x sweep):
    # 0.9 cm at 0.10, 2.6 cm at 0.30. Sheared scans enter the pose graph and
    # optimisation cannot un-shear them -- that is how the 10 Sep map was lost.
    # Putting 0.30 behind a held button makes the expensive speed deliberate,
    # which is the same problem teleop_speed_guard solves for the keyboard's `q`.
    scale_linear_arg = DeclareLaunchArgument(
        'scale_linear',
        default_value='0.10',
        description='m/s at full stick deflection without turbo. The mapping '
                    'speed -- leave it alone for any run that builds a map.',
    )

    scale_linear_turbo_arg = DeclareLaunchArgument(
        'scale_linear_turbo',
        default_value='0.30',
        description='m/s at full deflection with the turbo button HELD. 0.30 '
                    'is the hard ceiling; diff_cont clamps there regardless.',
    )

    scale_angular_arg = DeclareLaunchArgument(
        'scale_angular',
        default_value='0.50',
        description='rad/s at full stick deflection, both modes. Matches the '
                    "Makefile's TURN and the speed guard's angular limit.",
    )

    # deadzone is a FRACTION of the axis range, and joy_node rescales what is
    # left back onto 0..1, so this only widens the centre dead spot -- it costs
    # no top end. 0.10 rather than the stock 0.30: a worn stick still rests at
    # zero, but fine control near centre survives, and fine control near centre
    # is the whole point of driving a mapping run with a stick.
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'device_id': ParameterValue(
                LaunchConfiguration('joy_dev'), value_type=int),
            'deadzone': 0.10,
            'autorepeat_rate': 20.0,
            'use_sim_time': use_sim_time,
        }],
        # If joy_node ever exits, take teleop_twist_joy down with it rather
        # than leave it subscribed to a topic that will never produce another
        # message. Both downstream topics fail safe on silence -- see the
        # header -- so this cannot strand a moving robot.
        #
        # This is NOT the missing-pad case. Verified 22 Sep with nothing
        # plugged in: joy_node starts, logs NOTHING, stays up, and publishes
        # no /joy at all, so the launch looks perfectly healthy and ignores
        # every input. There is no error to catch, which is exactly why
        # `make joy-check` runs the SDL enumeration before either target.
        on_exit=Shutdown(reason='joy_node exited'),
    )

    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        output='screen',
        parameters=[
            # Layer 1: the pad's button/axis indices.
            PathJoinSubstitution([
                joy_share, 'config',
                [joy_config, TextSubstitution(text='.config.yaml')],
            ]),
            # Layer 2: our speeds, overwriting layer 1's. Later files win.
            {
                'scale_linear.x': ParameterValue(
                    LaunchConfiguration('scale_linear'), value_type=float),
                'scale_linear_turbo.x': ParameterValue(
                    LaunchConfiguration('scale_linear_turbo'),
                    value_type=float),
                'scale_angular.yaw': ParameterValue(
                    LaunchConfiguration('scale_angular'), value_type=float),
                'scale_angular_turbo.yaw': ParameterValue(
                    LaunchConfiguration('scale_angular'), value_type=float),
                # The deadman stays mandatory: no button, no motion.
                'require_enable_button': True,
                # diff_cont is use_stamped_vel:false and teleop_speed_guard
                # subscribes to a plain Twist. A TwistStamped here would be
                # dropped by both, silently.
                'publish_stamped_twist': False,
                'use_sim_time': use_sim_time,
            },
        ],
        remappings=[('/cmd_vel', cmd_vel_topic)],
    )

    return LaunchDescription([
        use_sim_time_arg,
        joy_config_arg,
        joy_dev_arg,
        cmd_vel_topic_arg,
        scale_linear_arg,
        scale_linear_turbo_arg,
        scale_angular_arg,
        joy_node,
        teleop_node,
    ])
