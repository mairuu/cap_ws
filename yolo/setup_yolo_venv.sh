#!/usr/bin/env bash
# Rebuild the YOLO inference environment from nothing.
#
# This is the file whose loss cost the most last time. requirements-frozen.txt
# beside it lists the resulting versions, but a freeze cannot express ORDER or
# EXCLUSIONS, and on this platform both matter more than the version numbers.
# Read the comments before changing anything.
#
# NEVER run `uv sync` against this venv. The recovered uv.lock pins generic PyPI
# torch 2.13.0 / torchvision 0.28.0 and omits tensorrt entirely; syncing it
# leaves a venv with no CUDA and no .engine support. Upstream yolo_ros's
# yolo_bringup/launch/yolo.launch.py runs exactly that sync on every start,
# which is why this workspace has its own launch file.
#
#   ./setup_yolo_venv.sh [venv_path]        default: ~/yolo/venv
set -euo pipefail

VENV="${1:-$HOME/yolo/venv}"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
# NOTE the .io. The .dev host that older notes reference does not resolve.

TORCH_WHL="$JETSON_INDEX/+f/46b/b8b13f844b211/torch-2.11.0-cp310-cp310-linux_aarch64.whl"
TV_WHL="$JETSON_INDEX/+f/d11/6f08d3d62417d/torchvision-0.26.0-cp310-cp310-linux_aarch64.whl"

command -v uv >/dev/null || { echo "uv not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "== 0. system prerequisites =="
# CUDA/cuDNN/TensorRT are NOT pip packages here, they are apt. If they are
# missing, the L4T apt sources are probably commented out -- see D-15.
for lib in libcudnn.so libnvinfer.so; do
    ldconfig -p | grep -q "$lib" || { echo "MISSING $lib -- install the JetPack userspace first, see records/decisions.md D-15"; exit 1; }
done
# TensorRT's PYTHON binding links the DLA compiler even though nothing here uses
# the DLA, and the file needs an ldconfig after install or it stays unfound.
python3 -c 'import tensorrt' 2>/dev/null || {
    echo "system 'import tensorrt' fails."
    echo "  sudo apt install nvidia-l4t-dla-compiler=36.4.0-20240912212859 && sudo ldconfig"
    exit 1; }
# torch 2.11 links libcudss, which is NOT in the NVIDIA Jetson apt repo at all.
# It comes from the PyPI wheel nvidia-cudss-cu12, but that wheel drags in CUDA
# 12.9 cublas/nvrtc which must never shadow JetPack's 12.6 -- so we take only
# the .so files, put them on the system path, and drop the wheel again.
ldconfig -p | grep -q libcudss.so.0 || {
    echo "MISSING libcudss.so.0 -- install it system-side, without keeping the CUDA 12.9 wheels:"
    echo "  uv pip install --python \$VENV/bin/python nvidia-cudss-cu12==0.7.1.6"
    echo "  sudo mkdir -p /usr/local/lib/cudss"
    echo "  sudo cp -a \$VENV/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss*.so* /usr/local/lib/cudss/"
    echo "  echo /usr/local/lib/cudss | sudo tee /etc/ld.so.conf.d/cudss.conf && sudo ldconfig"
    echo "  uv pip uninstall --python \$VENV/bin/python cuda-toolkit nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudss-cu12"
    exit 1; }

echo "== 1. venv =="
# --system-site-packages is MANDATORY: rclpy is a system apt package and will
# not install into a sealed venv. Same interpreter the ROS nodes use, so only
# PYTHONPATH needs setting at launch -- there is no `activate` step.
uv venv --system-site-packages --python /usr/bin/python3.10 "$VENV"
PY="$VENV/bin/python"

echo "== 2. JetPack torch FIRST, by direct URL =="
# Order matters. Installing ultralytics first would resolve a generic PyPI torch
# (CPU-only or x86) and win. Direct URLs so no index can substitute a wheel.
uv pip install --python "$PY" "$TORCH_WHL" "$TV_WHL"

echo "== 3. pin numpy < 2 =="
# torch's own dependency resolution installs numpy 2.x, which SHADOWS the system
# numpy 1.21 and breaks the system cv2 (built against 1.x) with
# "numpy.core.multiarray failed to import". This must come after torch.
uv pip install --python "$PY" "numpy<2"

echo "== 4. ultralytics, without its dependency list =="
uv pip install --python "$PY" ultralytics --no-deps
# Then its real deps by hand, MINUS three:
#   opencv-python  -- the system cv2 is used via system-site-packages; the PyPI
#                     wheel shadows it and is not CUDA-aware
#   torch/torchvision -- already installed above, from JetPack
#   numpy          -- already pinned <2 above
uv pip install --python "$PY" \
    cloudpickle filelock matplotlib pillow pyyaml requests \
    psutil polars nvidia-ml-py ultralytics-thop "numpy<2"

echo "== 5. verify =="
source /opt/ros/humble/setup.bash
"$PY" - <<'EOF'
import torch, torchvision, cv2, numpy, rclpy, tensorrt, ultralytics
assert torch.cuda.is_available(), "torch cannot see the GPU"
(torch.randn(256, 256, device='cuda') @ torch.randn(256, 256, device='cuda')).sum().item()
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  tensorrt     {tensorrt.__version__}")
print(f"  cudnn        {torch.backends.cudnn.version()}")
print(f"  cv2          {cv2.__version__}   numpy {numpy.__version__}")
print(f"  ultralytics  {ultralytics.__version__}")
print("  rclpy ok")
EOF
echo "== done: $VENV =="
