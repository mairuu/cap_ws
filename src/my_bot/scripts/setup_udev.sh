#!/usr/bin/env bash
#
# Generate and install udev rules that pin the robot's two USB serial adapters
# to stable names:
#
#   /dev/esp32    -> the ESP32 base controller (DiffDriveSerial)
#   /dev/ydlidar  -> the YDLidar X2
#
# Both are USB serial adapters, so whichever enumerates first becomes
# /dev/ttyUSB0 and the assignment flips across reboots. The symlinks above
# follow the hardware instead of the enumeration order.
#
# Run this ONCE, with both devices plugged in:
#
#   ./scripts/setup_udev.sh
#
# It asks which /dev/ttyUSB* is which, then picks a matching key:
#
#   * ATTRS{serial}  -- preferred. Survives being moved to another USB port.
#   * KERNELS        -- the USB port path, used when the adapters are
#                       indistinguishable (same VID:PID, no serial, which is
#                       common for CH340s and for cloned CP2102s). The device
#                       must then stay in the SAME physical port.
#
# Re-run it after swapping hardware or moving a device between ports.

set -euo pipefail

RULES_FILE=/etc/udev/rules.d/99-my-bot-serial.rules
REPO_COPY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/udev/99-my-bot-serial.rules"

# --- collect candidate devices -----------------------------------------------

mapfile -t DEVS < <(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true)

if [ "${#DEVS[@]}" -eq 0 ]; then
  echo "No /dev/ttyUSB* or /dev/ttyACM* found. Plug in the ESP32 and the lidar," >&2
  echo "then run this again." >&2
  exit 1
fi

# Walk up the sysfs tree from a tty to the USB *device* node (the one that owns
# idVendor), which is the level udev rules match on.
usb_parent() {
  local p="/sys$(udevadm info -q path -n "$1")"
  while [ "$p" != "/sys" ] && [ "$p" != "/" ]; do
    [ -f "$p/idVendor" ] && { echo "$p"; return 0; }
    p="$(dirname "$p")"
  done
  return 1
}

declare -A VID PID SER PORT
USB_DEVS=()   # only devices we could resolve; the menu numbers index THIS array
for d in "${DEVS[@]}"; do
  parent="$(usb_parent "$d")" || { echo "warn: no USB parent for $d, skipping" >&2; continue; }
  VID[$d]="$(cat "$parent/idVendor")"
  PID[$d]="$(cat "$parent/idProduct")"
  SER[$d]="$(cat "$parent/serial" 2>/dev/null || true)"
  PORT[$d]="$(basename "$parent")"
  USB_DEVS+=("$d")
done

if [ "${#USB_DEVS[@]}" -lt 2 ]; then
  echo "Found ${#USB_DEVS[@]} USB serial device(s); need both the ESP32 and the lidar" >&2
  echo "plugged in at the same time." >&2
  exit 1
fi

echo "Serial devices found:"
echo
for i in "${!USB_DEVS[@]}"; do
  d="${USB_DEVS[$i]}"
  printf "  [%d] %-14s %s:%s  serial=%-18s usb-port=%s\n" \
    "$((i + 1))" "$d" "${VID[$d]}" "${PID[$d]}" "${SER[$d]:-<none>}" "${PORT[$d]}"
done
echo

# --- ask which is which -------------------------------------------------------
#
# The human still answers, because a wrong answer here is the one failure this
# script cannot fail loudly on: both adapters are 10c4:ea60 with the same
# non-unique serial, so the rules match on USB port path and crossed names look
# exactly like working names. It happened on 2026-09-08 -- see
# udev/99-my-bot-serial.rules.
#
# But the answer IS checkable, which the earlier comment here denied:
#
#   YDLidar -- streams ~4.5 kB/s of 0xAA55 frames from the moment it has power
#              and never stops. Reading it does not spin the motor: the motor is
#              already spinning, which is why there is a stream to read. It has
#              no command interface, so an `e` sent at it is inert.
#   ESP32   -- silent until spoken to, and answers `e` with two integers.
#
# Volume alone settles it. `# boot reset=1 encoders=ok` is a second signal when
# the open happens to reset the board, but that is not dependable: whether an
# open resets depends on the DTR/RTS state the last close left behind (`hupcl`),
# so classify() asks `e` rather than waiting for a banner.
#
# Advisory, not authoritative -- it says "unknown" for a board that is
# unpowered, held by another process, or mid-flash, and it must never override a
# human who can see the cables. It refuses only on a straight contradiction.

pick() {
  local prompt="$1" ans
  while true; do
    read -rp "$prompt" ans
    ans="${ans#[}"; ans="${ans%]}"
    if [[ "$ans" =~ ^[0-9]+$ ]] && [ "$ans" -ge 1 ] && [ "$ans" -le "${#USB_DEVS[@]}" ]; then
      echo "${USB_DEVS[$((ans - 1))]}"
      return 0
    fi
    echo "Enter a number between 1 and ${#USB_DEVS[@]}." >&2
  done
}

echo "Tip: unplug one device and re-run to see which entry disappears."
echo
ESP_DEV="$(pick 'Which number is the ESP32 base controller? ')"
LIDAR_DEV="$(pick 'Which number is the YDLidar? ')"

if [ "$ESP_DEV" = "$LIDAR_DEV" ]; then
  echo "Those are the same device. Aborting." >&2
  exit 1
fi

# --- check those answers against the wire -------------------------------------

# Say what a port sounds like: esp32, lidar or unknown.
LISTEN_SECS=2
LIDAR_BYTES=3000        # the X2 clears this in well under a second
classify() {
  local dev="$1" tmp size reply
  tmp="$(mktemp)"
  # One open, held on fd 3, so the termios settings and the reply belong to the
  # same session -- closing between them can reset both the port and the board.
  exec 3<>"$dev" || { rm -f "$tmp"; echo unknown; return; }
  stty -F "$dev" 57600 raw -echo 2>/dev/null || true

  timeout "$LISTEN_SECS" cat <&3 >"$tmp" 2>/dev/null || true
  size="$(stat -c%s "$tmp")"
  if [ "$size" -gt "$LIDAR_BYTES" ]; then
    exec 3<&-; rm -f "$tmp"; echo lidar; return
  fi
  if grep -aq '# boot' "$tmp"; then
    exec 3<&-; rm -f "$tmp"; echo esp32; return
  fi
  rm -f "$tmp"

  # Quiet so far. Ask it something only the firmware answers.
  printf 'e\r' >&3
  reply=''
  read -r -t 1 -u 3 reply 2>/dev/null || true
  exec 3<&-
  reply="${reply%$'\r'}"
  if [[ "$reply" =~ ^-?[0-9]+[[:space:]]+-?[0-9]+$ ]]; then
    echo esp32; return
  fi
  echo unknown
}

echo
echo "Checking those answers against the wire (${LISTEN_SECS}s each)..."
ESP_HEARD="$(classify "$ESP_DEV")"
LIDAR_HEARD="$(classify "$LIDAR_DEV")"
printf '  %s answered as: %s\n' "$ESP_DEV" "$ESP_HEARD"
printf '  %s answered as: %s\n' "$LIDAR_DEV" "$LIDAR_HEARD"

CONTRADICTED=0
[ "$ESP_HEARD" = lidar ] && CONTRADICTED=1
[ "$LIDAR_HEARD" = esp32 ] && CONTRADICTED=1

if [ "$CONTRADICTED" -eq 1 ]; then
  echo
  echo "That is backwards. $ESP_DEV sounds like the lidar and/or $LIDAR_DEV" >&2
  echo "sounds like the ESP32 -- swap your two answers and run this again." >&2
  echo "Installing as answered would cross /dev/esp32 and /dev/ydlidar, and" >&2
  echo "nothing downstream would report an error; the lidar driver would just" >&2
  echo "see silence and ros2_control would see noise." >&2
  exit 1
fi

if [ "$ESP_HEARD" != esp32 ]; then
  echo
  echo "warn: $ESP_DEV did not answer \`e\` with two counts (heard:" >&2
  echo "  $ESP_HEARD). Unpowered, held by another process, or running firmware" >&2
  echo "  that does not speak this protocol. Proceeding on your answer." >&2
fi
if [ "$LIDAR_HEARD" != lidar ]; then
  echo
  echo "warn: $LIDAR_DEV is not streaming, so it does not sound like a powered" >&2
  echo "  lidar (heard: $LIDAR_HEARD). Proceeding on your answer." >&2
fi

# --- choose a matching key ----------------------------------------------------

emit_rule() {
  local dev="$1" name="$2" other="$3"

  if [ -n "${SER[$dev]}" ] && [ "${SER[$dev]}" != "${SER[$other]:-}" ]; then
    printf 'SUBSYSTEM=="tty", ATTRS{idVendor}=="%s", ATTRS{idProduct}=="%s", ATTRS{serial}=="%s", SYMLINK+="%s", GROUP="dialout", MODE="0660"\n' \
      "${VID[$dev]}" "${PID[$dev]}" "${SER[$dev]}" "$name"
  else
    # No usable serial: fall back to the physical USB port path.
    printf '# %s has no unique serial -- matched by USB port. Keep it in this port.\n' "$name"
    printf 'SUBSYSTEM=="tty", SUBSYSTEMS=="usb", KERNELS=="%s", SYMLINK+="%s", GROUP="dialout", MODE="0660"\n' \
      "${PORT[$dev]}" "$name"
  fi
}

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
{
  echo "# my_bot serial device names. Generated by scripts/setup_udev.sh on $(date -Is)."
  echo "# Regenerate after swapping hardware or moving a device to another USB port."
  echo
  emit_rule "$ESP_DEV" esp32 "$LIDAR_DEV"
  emit_rule "$LIDAR_DEV" ydlidar "$ESP_DEV"
} >"$TMP"

echo
echo "Rules to install at $RULES_FILE:"
echo
sed 's/^/    /' "$TMP"
echo

read -rp "Install? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted, nothing written."; exit 1; }

# --- install ------------------------------------------------------------------

sudo install -m 0644 "$TMP" "$RULES_FILE"
mkdir -p "$(dirname "$REPO_COPY")"
install -m 0644 "$TMP" "$REPO_COPY"

sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --action=add
sleep 1

echo
echo "Installed. A copy is checked in at udev/99-my-bot-serial.rules."
echo
ok=0
for name in esp32 ydlidar; do
  if [ -e "/dev/$name" ]; then
    printf '  /dev/%-8s -> %s\n' "$name" "$(readlink -f "/dev/$name")"
    ok=$((ok + 1))
  else
    printf '  /dev/%-8s MISSING\n' "$name"
  fi
done

if [ "$ok" -ne 2 ]; then
  echo
  echo "A symlink did not appear. Unplug and replug that device -- some adapters" >&2
  echo "do not re-trigger on 'udevadm trigger'. If it still does not show up:" >&2
  echo "  udevadm test \$(udevadm info -q path -n /dev/ttyUSB0)" >&2
  exit 1
fi
