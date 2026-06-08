#!/usr/bin/env bash
#
# Overnight diffusion retraining with the "new input style"
# (range + null space unfiltered; full_fbp conditioning channel = sharpened ramp FBP).
# This change lives in scripts/prep_DLR.py build_measurement_components and is inherited
# by the diffusion scripts via the shared SpectralFeatureBuilder.
#
# Stages (run INSIDE the recon-web-server container):
#   1) Base diffusion training : 200 resume epochs, n_source 80 + 240          -> 200*2 = 400
#   2) Per-exposure fine-tuning : 20 epochs x {0.1,1.0,10.0,100.0} mAs x {80,240} -> 20*4*2 = 160
#   Total = 560 epochs.
#
# All output (this script + both python runs) goes to ONE log file.
#
# Run (detached) and tail:
#   docker exec -d recon-web-server bash /workspace/static_ct_recon_demo/scripts/run_diffusion_overnight.sh
#   docker exec recon-web-server tail -f /workspace/static_ct_recon_demo/logs/diffusion_train_newstyle.log
#
# Override the GPU with:  DEVICE=cuda:1 docker exec -d ... (see README of this run)

set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
REPO=/workspace/static_ct_recon_demo
LOG="$REPO/logs/diffusion_train_newstyle.log"

mkdir -p "$REPO/logs"
cd "$REPO"

# Redirect EVERYTHING below to the single log file (overwrite for a fresh run).
exec > "$LOG" 2>&1

echo "=================================================================="
echo "DIFFUSION RETRAIN (new input style)   device=$DEVICE"
echo "start: $(date -u)"
echo "=================================================================="

echo
echo "##### STAGE 1/2: base diffusion training (200 epochs, resume, n_source 80 + 240) #####"
python scripts/prep_diffusion_train.py --n-sources 80 240 --epochs 200 --device "$DEVICE"

echo
echo "##### STAGE 2/2: per-exposure fine-tuning (20 epochs x 4 exposures x {80,240}) #####"
python scripts/prep_diffusion_finetuning.py --n-sources 80 240 --exposures 0.1 1.0 10.0 100.0 --epochs 20 --device "$DEVICE"

echo
echo "=================================================================="
echo "DIFFUSION RETRAIN COMPLETE   end: $(date -u)"
echo "=================================================================="
