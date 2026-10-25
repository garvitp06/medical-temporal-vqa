#!/usr/bin/env bash
# Download the pretrained lungmask U-Net R231 (Hofmanninger et al. 2020) used for lung masks.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
f=weights/unet_r231-d5d2fc3d.pth
[ -s "$f" ] || curl -fL -o "$f" https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth
sha256sum "$f"
