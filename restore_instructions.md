# Weight Restore Instructions

Before the "new input style" retraining (DLR + diffusion), the existing weights were
backed up in place with a `.bak_prenewstyle` suffix next to each original `.pt` file.
This document explains how to restore them if the retraining results are worse.

All paths are relative to the repo root:
`/home/staticct/matt/workspace/static_ct_recon_demo`
(inside the `recon-web-server` container this is `/workspace/static_ct_recon_demo`).

## What was backed up

Model dir: `backend/app/api/deep_learning/models/main/n_source_{80,240}/`

**DLR / NeuralSpark (canonical `main`):**
- `main.pt` → `main.pt.bak_prenewstyle`

**Diffusion base + per-exposure (canonical `main`):**
- `diffusion_{80,240}.pt` → `…​.bak_prenewstyle`
- `diffusion_{80,240}_{0.1,1.0,10.0,100.0}mas.pt` → `…​.bak_prenewstyle`

(Note: `diffusion_{80,240}_original_backup.pt` are older backups created by the
fine-tuning script on a previous run — not the pre-new-style snapshot. Prefer the
`.bak_prenewstyle` files for restoring the state from just before this retraining.)

The DLR per-epoch checkpoints under `…/n_source_{80,240}/checkpoints/epoch_*.pt`
are also preserved by the trainer and can be used to roll back to a specific epoch.

## Restore everything (DLR + diffusion, both geometries)

Run from the repo root (host):

```bash
cd /home/staticct/matt/workspace/static_ct_recon_demo
find backend/app/api/deep_learning/models/main -name '*.bak_prenewstyle' | while read -r b; do
  orig="${b%.bak_prenewstyle}"
  cp -f "$b" "$orig"
  echo "restored $orig"
done
```

Then restart the web server so it reloads the restored weights:

```bash
docker restart recon-web-server
```

## Restore only one model

```bash
cd /home/staticct/matt/workspace/static_ct_recon_demo
D=backend/app/api/deep_learning/models/main

# DLR (NeuralSpark) 80-view:
cp -f $D/n_source_80/main.pt.bak_prenewstyle            $D/n_source_80/main.pt

# Diffusion base 80-view:
cp -f $D/n_source_80/diffusion_80.pt.bak_prenewstyle    $D/n_source_80/diffusion_80.pt

# Diffusion 0.1 mAs fine-tuned, 80-view:
cp -f $D/n_source_80/diffusion_80_0.1mas.pt.bak_prenewstyle  $D/n_source_80/diffusion_80_0.1mas.pt
```

Swap `n_source_80`/`80` for `n_source_240`/`240`, and the exposure for
`1.0`, `10.0`, or `100.0` as needed. Restart the container afterward.

## Notes

- The backups were made with `cp -n` (no-clobber), so re-running the backup step
  will NOT overwrite these `.bak_prenewstyle` snapshots.
- Restoring does not touch the code change to the input style
  (`scripts/prep_DLR.py` `build_measurement_components`). If you also want to revert
  the input-style change itself, do that in git separately.
