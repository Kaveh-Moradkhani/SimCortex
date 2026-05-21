# SimCortex v2.0 Docker Guide

This document explains how to build and run the **SimCortex v2.0** Docker image.
It focuses on containerized execution of the `scpp` CLI for **Stage 2 (Segmentation)**, **Stage 3 (InitSurf)**, and **Stage 4 (Deform)**. For the full pipeline itself—stage logic, expected inputs and outputs, naming conventions, and workflow order—see the repository root `README.md`.

The Docker image is intended to provide a reproducible runtime for SimCortex, including the Python / CUDA / PyTorch / PyTorch3D stack used by the project.

---

## Table of Contents

- [Overview](#overview)
- [Published Resources](#published-resources)
- [Image Tags](#image-tags)
- [Build the Image](#build-the-image)
- [Quick Validation](#quick-validation)
- [Recommended Runtime Pattern](#recommended-runtime-pattern)
- [Mounting Datasets and Outputs](#mounting-datasets-and-outputs)
- [Hydra Configuration from Docker](#hydra-configuration-from-docker)
- [Inspect Packaged Config Files](#inspect-packaged-config-files)
- [Use Your Own YAML Config](#use-your-own-yaml-config)
- [Examples by Stage](#examples-by-stage)
- [GPU Support](#gpu-support)
- [Shared Server and HPC Notes](#shared-server-and-hpc-notes)
- [Docker Hub Publication](#docker-hub-publication)
- [Apptainer / Singularity Notes](#apptainer--singularity-notes)

---

## Overview

The Docker image allows users to run SimCortex without recreating the full local environment manually. This is useful for:

- reproducibility across workstations and servers
- simpler setup for collaborators
- preserving a validated PyTorch / PyTorch3D stack
- CLI-based workflows where datasets and outputs are mounted from the host

At the moment, the main published Docker image is intentionally focused on the **PyTorch / PyTorch3D-based stages** of the pipeline:

- **Stage 2 — Segmentation**
- **Stage 3 — InitSurf**
- **Stage 4 — Deform**

**Stage 1 — FreeSurfer to MNI152 preprocessing** is typically run outside Docker because it depends on external tools such as **FreeSurfer** and **NiftyReg**, and packaging those tools into the main image would substantially increase image size and maintenance burden.

The image is not intended to replace the project README. Instead, it provides a container runtime for the **Segmentation**, **InitSurf**, and **Deform** commands documented there. In the current recommended workflow, **Preprocessing (Stage 1)** is run outside the main Docker image.

---

## Published Resources

- **Docker Hub:** [kavehmoradkhani/simcortex](https://hub.docker.com/r/kavehmoradkhani/simcortex)
- **Zenodo pre-trained weights and splits:** [SimCortex v2.0: Pre-trained Models and Dataset Splits](https://zenodo.org/records/18974730)

---

## Image Tags

Examples below use the local image tag:

```text
simcortex:2.0.0
```

After publishing to Docker Hub, the tag can be:

```text
kavehmoradkhani/simcortex:2.0.0
```

Official repository:

```text
https://hub.docker.com/r/kavehmoradkhani/simcortex
```

Keep versioned tags even if you later publish `latest`, so users can pin an exact image for reproducibility.

---

## Build the Image

From the repository root:

```bash
docker build -f docker/Dockerfile -t simcortex:2.0.0 .
```

The image is expected to bundle the SimCortex runtime stack, including the packaged `scpp` CLI and Hydra YAML configs.

---

## Quick Validation

Show the main CLI:

```bash
docker run --rm simcortex:2.0.0 simcortex --help
```

Show stage help:

```bash
docker run --rm simcortex:2.0.0 simcortex seg --help
docker run --rm simcortex:2.0.0 simcortex initsurf --help
docker run --rm simcortex:2.0.0 simcortex deform --help
```

Verify key Python packages:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import torch, pytorch3d, pymeshlab; print(torch.__version__)"
```

Verify GPU visibility:

```bash
docker run --rm --gpus all simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

---

## Recommended Runtime Pattern

On shared Linux systems, Docker writes files as `root` by default unless told otherwise. For ordinary SimCortex runs, the safest pattern is to run as the host user and mount data and outputs explicitly.

Recommended runtime block:

```bash
--user $(id -u):$(id -g) \
-e HOME=/tmp \
-e UMASK=002
```

Recommended day-to-day command pattern:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex --help
```

This gives you:

- host-owned output files instead of root-owned files
- explicit mounted paths for datasets and run outputs
- GPU access when Docker and the NVIDIA runtime are configured correctly

If you do not need GPU access, omit `--gpus all`.

---

## Mounting Datasets and Outputs

Docker containers do not automatically see host files. Mount them explicitly with `-v`:

```bash
-v /host/path:/container/path
```

A simple and maintainable layout is:

```bash
-v /home/<user>/datasets:/data \
-v /home/<user>/runs:/runs
```

Inside the container:

- `/data` points to datasets and derivatives
- `/runs` points to experiment outputs and logs

Example host layout:

```text
/home/<user>/datasets/
  hcpya-u100/
  oasis-1/
  splits/
/home/<user>/runs/
  seg/
  initsurf/
  deform/
```

Example mounted layout inside the container:

```text
/data/hcpya-u100
/data/oasis-1
/data/splits
/runs/seg
/runs/initsurf
/runs/deform
```

Keeping datasets and run outputs mounted separately makes commands easier to read and reduces mistakes.

---

## Hydra Configuration from Docker

SimCortex uses Hydra configs. You can pass overrides directly through the CLI inside `docker run`.

General pattern:

```bash
docker run --rm [docker-options] simcortex:2.0.0 \
  simcortex <stage> <command> key=value key=value
```

Example:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex initsurf generate \
  dataset.split_file=/data/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/data/hcpya-u100/derivatives/scpp-preproc-0.1 \
  dataset.seg_roots.HCP_YA=/data/hcpya-u100/derivatives/scpp-seg-0.1 \
  outputs.out_roots.HCP_YA=/data/hcpya-u100/derivatives/scpp-initsurf-0.1 \
  outputs.log_dir=/runs/initsurf/exp01/logs
```

This is usually the simplest approach for quick tests and one-off runs.

---

## Inspect Packaged Config Files

If you want to inspect the packaged Hydra configs inside the image, you can locate the installed package and print a config file directly.

Print the package location:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import simcortexpp, pathlib; print(pathlib.Path(simcortexpp.__file__).resolve().parent)"
```

Print an InitSurf config to stdout:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import simcortexpp, pathlib; p=pathlib.Path(simcortexpp.__file__).resolve().parent/'configs'/'initsurf'/'generate.yaml'; print(p.read_text())"
```

Print a Deform train config to stdout:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import simcortexpp, pathlib; p=pathlib.Path(simcortexpp.__file__).resolve().parent/'configs'/'deform'/'train.yaml'; print(p.read_text())"
```

Save a packaged config to the host:

```bash
mkdir -p /tmp/scpp_cfg

docker run --rm \
  -v /tmp/scpp_cfg:/out \
  simcortex:2.0.0 \
  python -c "import simcortexpp, pathlib; p=pathlib.Path(simcortexpp.__file__).resolve().parent/'configs'/'initsurf'/'generate.yaml'; open('/out/generate.yaml','w').write(p.read_text())"
```

After that, edit `/tmp/scpp_cfg/generate.yaml` on the host.

---

## Use Your Own YAML Config

If a stage supports a `user_config` pattern, mount your custom YAML file and pass it explicitly.

Example:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  -v /home/<user>/myconfigs:/cfg \
  simcortex:2.0.0 \
  simcortex deform train user_config=/cfg/train.yaml
```

If your stage does not use a `user_config` field, override individual values directly on the CLI instead.

---

## Examples by Stage

> Note: In the current recommended setup, **Stage 1 (FreeSurfer to MNI152 preprocessing)** is run outside the main Docker image. The examples below therefore start from **Stage 2** and assume preprocessing outputs already exist under `scpp-preproc-*`.

### Stage 2 — Segmentation train

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex seg train \
  dataset.path=/data/<dataset>/derivatives/scpp-preproc-0.1 \
  dataset.split_file=/data/splits/<dataset>_split.csv \
  outputs.root=/runs/seg/exp01
```

### Stage 2 — Segmentation inference

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  simcortex:2.0.0 \
  simcortex seg infer \
  dataset.path=/data/<dataset>/derivatives/scpp-preproc-0.1 \
  dataset.split_file=/data/splits/<dataset>_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/data/checkpoints/seg_best_dice.pt \
  outputs.out_root=/data/<dataset>/derivatives/scpp-seg-0.1
```

### Stage 3 — InitSurf

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex initsurf generate \
  dataset.split_file=/data/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/data/hcpya-u100/derivatives/scpp-preproc-0.1 \
  dataset.seg_roots.HCP_YA=/data/hcpya-u100/derivatives/scpp-seg-0.1 \
  outputs.out_roots.HCP_YA=/data/hcpya-u100/derivatives/scpp-initsurf-0.1 \
  outputs.log_dir=/runs/initsurf/exp01/logs
```

### Stage 4 — Deformation train

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex deform train \
  outputs.root=/runs/deform/exp01
```

### Stage 4 — Deformation eval

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex deform eval
```

---

## GPU Support

If Docker and the NVIDIA runtime are configured correctly on the host, enable GPU access with:

```bash
--gpus all
```

First validate the host Docker GPU setup:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
  nvidia-smi
```

Then validate GPU access inside the SimCortex image:

```bash
docker run --rm --gpus all simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If the CUDA test container fails, the problem is with the host Docker / NVIDIA runtime setup rather than SimCortex.

---

## Shared Server and HPC Notes

On some managed Linux systems, the Docker daemon cannot bind-mount arbitrary paths such as:

```text
/project/...
```

Even if the path exists for the user, Docker may return an error like:

```text
error while creating mount source path ... permission denied
```

This is typically a host-side Docker policy issue, not an SimCortex issue.

Practical workarounds:

- mount from `$HOME/...`
- mount from `/tmp/...`
- use another host path explicitly allowed by your Docker configuration
- create a symlink, copy, or temporary view under an allowed path before running Docker

---

## Docker Hub Publication

Tag the local image:

```bash
docker tag simcortex:2.0.0 kavehmoradkhani/simcortex:2.0.0
```

Push it:

```bash
docker push kavehmoradkhani/simcortex:2.0.0
```

Then users can pull it with:

```bash
docker pull kavehmoradkhani/simcortex:2.0.0
```

If you later publish a `latest` tag, keep the versioned tag as well.

---

## Apptainer / Singularity Notes

Many neuroimaging and HPC systems prefer **Apptainer / Singularity** rather than Docker. A common pattern is to convert the Docker image into a `.sif` file.

Build from the local Docker image:

```bash
apptainer build simcortex_2.0.0.sif docker-daemon://simcortex:2.0.0
```

Build from Docker Hub after publication:

```bash
apptainer build simcortex_2.0.0.sif docker://kavehmoradkhani/simcortex:2.0.0
```

After conversion, re-check:

- CLI behavior
- writable output directories
- environment variables
- GPU access on the target system

Apptainer runtime behavior is often close to Docker, but it is not always identical.

---

