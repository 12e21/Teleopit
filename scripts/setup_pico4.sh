#!/usr/bin/env bash
# Setup script for Pico4 xrobotoolkit_sdk Python bindings.
#
# Prerequisites:
#   1. Ubuntu 22.04
#   2. Install PC Service deb package:
#        wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
#        sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
#   3. Install pybind11: conda install -c conda-forge pybind11
#
# Usage:
#   bash scripts/setup_pico4.sh
#
set -euo pipefail

echo "=== Pico4 SDK (xrobotoolkit_sdk) Setup ==="

# ---------- Check PC Service is installed ----------
SO_SEARCH_PATHS=(
    "/opt/apps/roboticsservice/SDK/x64"
    "/usr/local/lib"
    "/usr/lib"
)
SO_FOUND=""
for dir in "${SO_SEARCH_PATHS[@]}"; do
    if [ -f "$dir/libPXREARobotSDK.so" ]; then
        SO_FOUND="$dir"
        break
    fi
done

if [ -z "$SO_FOUND" ]; then
    echo "ERROR: libPXREARobotSDK.so not found."
    echo "Please install the PC Service deb package first:"
    echo "  wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb"
    echo "  sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb"
    exit 1
fi
echo "Found libPXREARobotSDK.so in $SO_FOUND"

# ---------- Ensure .so is on the linker path ----------
if ! ldconfig -p 2>/dev/null | grep -q libPXREARobotSDK; then
    echo "--- Registering libPXREARobotSDK.so in system library path (requires sudo) ---"
    sudo ln -sf "$SO_FOUND/libPXREARobotSDK.so" /usr/local/lib/libPXREARobotSDK.so
    sudo ldconfig
    echo "Done: symlinked to /usr/local/lib/"
fi

# ---------- Build & install Python bindings ----------
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
echo "Working in $WORKDIR"

echo "--- Cloning XRoboToolkit-PC-Service-Pybind ---"
git clone --depth 1 \
    https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git \
    "$WORKDIR/pybind"
cd "$WORKDIR/pybind"

# Use the .so and headers from PC Service installation
mkdir -p lib include
cp "$SO_FOUND/libPXREARobotSDK.so" lib/

# Headers: prefer PC Service install path, fall back to building from source
HEADER_DIR="/opt/apps/roboticsservice/SDK/x64"
if [ -f "$HEADER_DIR/PXREARobotSDK.h" ]; then
    cp "$HEADER_DIR/PXREARobotSDK.h" include/
    [ -d "$HEADER_DIR/nlohmann" ] && cp -r "$HEADER_DIR/nlohmann" include/
else
    # Fall back: clone and copy headers only (no build needed)
    echo "--- Fetching headers from source ---"
    git clone --depth 1 \
        https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
        "$WORKDIR/pc-service"
    cp "$WORKDIR/pc-service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" include/
    cp -r "$WORKDIR/pc-service/RoboticsService/PXREARobotSDK/nlohmann" include/nlohmann/
fi

echo "--- Installing xrobotoolkit_sdk ---"
pip install .

echo ""
echo "=== Done! ==="
python -c 'import xrobotoolkit_sdk; print("Verify: import xrobotoolkit_sdk OK")'
