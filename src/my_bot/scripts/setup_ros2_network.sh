#!/usr/bin/env bash
# Configure this machine for multi-machine ROS 2 over the phone hotspot.
#
# Run it on EVERY machine that takes part -- the Jetson and any laptop running
# RViz or teleop. It is idempotent: re-running rewrites its own block in
# ~/.bashrc and nothing else.
#
#   ./setup_ros2_network.sh                    # defaults below
#   ./setup_ros2_network.sh --domain 42 --peers 172.20.10.2,172.20.10.5
#
# WHAT IT SETS, AND WHY EACH ONE IS NEEDED
#
#   ROS_DOMAIN_ID       Deliberately NOT 0. Domain 0 is what every other ROS 2
#                       machine on a shared hotspot also defaults to, and the
#                       failure is silent: a classmate's nodes appear in
#                       `ros2 topic list`, their /tf fights ours, and it reads
#                       as a broken TF tree rather than as a second robot.
#
#   ROS_LOCALHOST_ONLY  Must be 0 or nothing leaves the machine at all. Humble
#                       still honours it; it is gone in Iron+.
#
#   FASTRTPS_DEFAULT_PROFILES_FILE
#                       Points at the XML written below. Phone hotspots are
#                       access points, and many drop client-to-client MULTICAST
#                       while forwarding unicast fine. Fast DDS discovers by
#                       multicast (239.255.0.1) by default, so on such a link
#                       two machines that can ping each other still see zero of
#                       each other's topics. The profile adds each peer's
#                       address as a unicast initial peer, so discovery no
#                       longer depends on multicast surviving the AP.
#
#                       The multicast locator is kept FIRST in the list on
#                       purpose. It is what nodes on the same machine use to
#                       find each other, and same-host discovery must not
#                       regress -- the robot stack is ~15 participants talking
#                       locally, and unicast initial peers only probe
#                       participant ids 0..4 by default.
#
# NOT set here: RMW_IMPLEMENTATION. The stack that works today runs on Humble's
# default rmw_fastrtps_cpp. Swapping to Cyclone four days from the demo buys
# nothing that this profile does not, and costs a new set of failure modes.
set -euo pipefail

DOMAIN=42
PEERS="172.20.10.2,172.20.10.5"

while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --peers)  PEERS="$2";  shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

XML="$HOME/.ros2/fastdds_hotspot.xml"
mkdir -p "$(dirname "$XML")"

{
  echo '<?xml version="1.0" encoding="UTF-8" ?>'
  echo '<dds xmlns="http://www.eprosima.com">'
  echo '  <profiles>'
  echo '    <participant profile_name="hotspot" is_default_profile="true">'
  echo '      <rtps>'
  echo '        <builtin>'
  echo '          <initialPeersList>'
  echo '            <!-- keep multicast first: this is same-host discovery -->'
  echo '            <locator><udpv4><address>239.255.0.1</address></udpv4></locator>'
  IFS=','
  for p in $PEERS; do
    printf '            <locator><udpv4><address>%s</address></udpv4></locator>\n' "$p"
  done
  unset IFS
  echo '          </initialPeersList>'
  echo '        </builtin>'
  echo '      </rtps>'
  echo '    </participant>'
  echo '  </profiles>'
  echo '</dds>'
} > "$XML"

BEGIN='# >>> cap_ws ros2 network >>>'
END='# <<< cap_ws ros2 network <<<'
RC="$HOME/.bashrc"
touch "$RC"

# Drop any previous block, then append the current one.
if grep -qF "$BEGIN" "$RC"; then
  sed -i "/$(printf '%s' "$BEGIN" | sed 's/[]\/$*.^[]/\\&/g')/,/$(printf '%s' "$END" | sed 's/[]\/$*.^[]/\\&/g')/d" "$RC"
fi

cat >> "$RC" <<RC_EOF
$BEGIN
# Written by my_bot/scripts/setup_ros2_network.sh -- re-run it to change these.
export ROS_DOMAIN_ID=$DOMAIN
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=$XML
$END
RC_EOF

echo "wrote   $XML"
echo "peers   239.255.0.1 (multicast), $PEERS"
echo "bashrc  block updated in $RC"
echo
echo "ROS_DOMAIN_ID=$DOMAIN  ROS_LOCALHOST_ONLY=0"
echo
echo "Open a NEW shell (or: source ~/.bashrc) before running any ros2 command."
