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
# Safe to launch NOW: the script first waits (polling every POLL_INTERVAL seconds) until the
# DLR training (prep_DLR.py) on GPU 0 has finished, so the two never compete for the GPU.
#
# Run (detached) and tail:
#   docker exec -d recon-web-server bash /workspace/static_ct_recon_demo/scripts/run_diffusion_overnight.sh
#   docker exec recon-web-server tail -f /workspace/static_ct_recon_demo/logs/diffusion_train_newstyle.log
#
# Override the GPU with:       DEVICE=cuda:1 docker exec -d ...
# Override the poll interval:  POLL_INTERVAL=120 docker exec -d ...

set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"   # seconds between checks for the DLR run to finish
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
echo "##### WAIT: holding until the DLR run (prep_DLR.py) on GPU 0 finishes #####"
# The '[p]' trick keeps this grep from matching its own process line.
while ps -eo args 2>/dev/null | grep -q "[p]rep_DLR\.py"; do
    echo "[$(date -u)] prep_DLR.py still running on GPU 0; waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done
echo "[$(date -u)] prep_DLR.py is no longer running -> starting diffusion training on $DEVICE."

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
