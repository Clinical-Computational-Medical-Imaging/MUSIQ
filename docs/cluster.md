# Running BOA on an HPC cluster (Apptainer/Singularity)

The `boa` task normally runs via the `shipai/boa-cli` Docker image. On clusters without Docker,
run it under Apptainer/Singularity instead:

```bash
musiq ... --tasks boa --boa-runtime apptainer --boa-sif /path/to/boa-cli.sif
```

## Building the SIF

Build the SIF once on a node that has `apptainer` (not the scheduler) from the provided
[`boa-cli.def`](../boa-cli.def):

```bash
apptainer build --fakeroot boa-cli.sif boa-cli.def   # build node needs internet to pull docker://shipai/boa-cli
```

`boa-cli.def` bootstraps the stock `shipai/boa-cli` image and stubs out TotalSegmentator's
`preview.py`. This is **required** on hosts with glibc ≥ 2.38: `--nv` binds the host's GL libraries
(`libGLdispatch.so.0`) into the container, and the stock image (glibc 2.31) then fails at import time
with `` ImportError: ... version `GLIBC_2.38' not found ``. MUSIQ never uses that preview image, so
removing the `fury`/OpenGL import fixes it without touching BOA's pinned CUDA stack.

If the build node has no internet, edit the def's header to `Bootstrap: docker-archive` /
`From: boa-cli-all.tar` and point it at a saved image archive.

## Cluster-specific notes

- **File ownership:** `apptainer exec` runs as your own user, so outputs come out owned by you and no
  `DOCKER_USER` chown is needed.
- **Weights:** the SIF filesystem is **read-only**, so BOA cannot download weights at runtime — you
  **must** pass `--boa-weights-path` to a pre-populated, writable, shared directory (unless the image
  already bundles the weights).
- **GPU:** request a GPU in your sbatch script (`--gres=gpu:1`); MUSIQ adds `--nv` automatically
  unless `--boa-device cpu`.
