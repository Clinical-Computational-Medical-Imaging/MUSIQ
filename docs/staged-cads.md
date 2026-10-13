# Large-scale staged CADS

The `cads` workflow task runs the three CADS stages (preprocess → inference → restore+combine) in
sequence. For large cohorts (~100–1000+ scans) the stages can instead be run as **separate jobs** so
CPU and GPU resources are used efficiently (e.g. CPU vs GPU SLURM partitions). All three must point at
the same processed tree (the staging dir defaults to `<processed>/cads_staging`, so they agree
automatically):

```bash
# 1. Preprocess (CPU only)
musiq_cads_preprocess --input-dirpath-processed /data/processed --cads-tasks all
# 2. Inference (GPU; add --cpu to force CPU)
musiq_cads_inference  --input-dirpath-processed /data/processed --cads-tasks all
# 3. Restore to original geometry + combine into CTcads.nii.gz (CPU only)
musiq_cads_restore    --input-dirpath-processed /data/processed --cads-tasks all
```

Each stage is idempotent: a study is skipped once its `CTcads.nii.gz` exists. Intermediates live in
the staging dir and are auto-removed once each `CTcads.nii.gz` is written.

> **Note:** `musiq_cads_inference` runs **only** the GPU inference stage — the full run-everything
> path is `musiq --tasks cads` or the three scripts above in order.
