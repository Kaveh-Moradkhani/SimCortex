<p align="center">
  <img src="docs/assets/simcortex-logo.png" alt="SimCortex logo" width="420"/>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.22710781">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22710781.svg" alt="Zenodo DOI"/>
  </a>
</p>

SimCortex v2.0.0 is the implementation accompanying the journal manuscript of SimCortex: a modular and reproducible framework for cortical surface reconstruction in **MNI152 space**. It provides four practical stages that can be run independently or as a full pipeline:

> **Previous version:** The original ShapeMI/MICCAI 2025 conference implementation is preserved as:
> - [SimCortex v1.0.0 release](https://github.com/Neuro-iX/SimCortex/releases/tag/v1.0.0)
> - [Legacy v1 branch](https://github.com/Neuro-iX/SimCortex/tree/legacy/v1-shapemi2025)
> - [Conference paper](https://arxiv.org/abs/2507.06955)


1. **Preprocessing (FreeSurfer to MNI152)**  
   Export key FreeSurfer volumes and surfaces, register them to MNI152, and write outputs in a **BIDS-derivatives-style** layout.

2. **Segmentation (3D U-Net, MNI space)**  
   Train and apply a 3D U-Net to predict a **9-class segmentation** in **MNI152 space**, with inference and evaluation utilities.

3. **Initial Surfaces (InitSurf)**  
   Generate initial White Matter and Pial surfaces from saved segmentation predictions, together with hemisphere SDFs and ribbon outputs.

4. **Deformation (Deform)**  
   Deform the initial surfaces toward MNI-aligned FreeSurfer target surfaces using geometric losses and optional collision-aware evaluation, and write **deformed surfaces** as BIDS derivatives.

This README focuses on **how to run the pipeline correctly**: expected inputs, produced outputs, folder and file naming conventions, and representative commands for each stage.

The **project and main Docker image support all four stages**.

---

## Table of Contents

- [Installation](#installation)
- [Pre-trained Weights and Official Splits](#pre-trained-weights-and-official-splits)
- [Configuration](#configuration)
- [Data and Folder Conventions](#data-and-folder-conventions)
- [Coordinate Convention](#coordinate-convention)
- [Split File Format](#split-file-format)
- [Stage 1 - Preprocessing (FreeSurfer to MNI152)](#stage-1---preprocessing-freesurfer-to-mni152)
- [Stage 2 - Segmentation (3D U-Net, MNI space)](#stage-2---segmentation-3d-u-net-mni-space)
- [Stage 3 - Initial Surfaces (InitSurf)](#stage-3---initial-surfaces-initsurf)
- [Stage 4 - Deformation (Deform)](#stage-4---deformation-deform)
- [Docker](#docker)
- [Citation](#citation)
- [License](#license)

---

## Installation

SimCortex is developed and release-tested on **Linux with Python 3.10**. The
validated SimCortex v2 environment uses **Python 3.10.19**, **PyTorch 2.1.0**
with a CUDA 12.1 build, **torchvision 0.16.0**, and **PyTorch3D 0.7.8**.

For the complete four-stage workflow, the **Docker image is the recommended
installation path** because it preserves the validated PyTorch / CUDA /
PyTorch3D stack. See [`docker/README.md`](docker/README.md).

### Local source installation

PyTorch3D is intentionally not installed through the SimCortex project metadata
because its compatible build depends on the selected PyTorch and CUDA versions.
For a local installation, first create a Python 3.10 environment and install a
mutually compatible PyTorch / PyTorch3D stack. Then, from the repository root,
install SimCortex with all declared stage extras:

```bash
python -m pip install -e ".[preproc,seg,initsurf,deform-metrics]"
```

The declared extras provide the following stage-specific dependencies:

- `preproc`: ANTsPy (`antspyx`) for N4 correction, registration, and resampling
- `seg`: MONAI
- `initsurf`: PyTorch, SciPy, scikit-image, Numba, and python-fcl
- `deform-metrics`: python-fcl and pymeshlab
- `torch`: PyTorch, for installations that need the standalone PyTorch extra

Deformation **training and evaluation** additionally require a compatible
**PyTorch3D** installation. PyTorch3D is not bundled or automatically selected
by `pip install -e .`.

FreeSurfer is also not installed by SimCortex. Stage 1 consumes existing
FreeSurfer outputs such as `orig.mgz`, `aseg.mgz`, `aparc+aseg.mgz`,
`filled.mgz`, and the cortical surfaces under `surf/`.

After installation, verify the public CLI:

```bash
simcortex --version
simcortex --help
simcortex fs-to-mni --help
simcortex seg --help
simcortex initsurf --help
simcortex deform --help
```

A base editable install is also available for development or inspection:

```bash
python -m pip install -e .
```

This base install does **not** install all dependencies required to execute every
pipeline stage.

---

## Pre-trained Weights and Official Splits

The finalized SimCortex v2 pre-trained models, configuration files, dataset
splits, and reproducibility metadata are archived on Zenodo:

- **Pre-trained models and reproducibility files:**
  [SimCortex v2: Pre-trained Models and Reproducibility Files](https://doi.org/10.5281/zenodo.22710781)

The associated reproducibility dataset, including the processed development
data and the multi-dataset evaluation cohort used in the SimCortex v2 study,
is archived separately in the Federated Research Data Repository (FRDR):

- **Reproducibility dataset:**
  [SimCortex Reproducibility Dataset for Cortical Surface Reconstruction](https://doi.org/10.20383/103.01769)

The validated SimCortex v2 release checkpoint identities are:

```text
Segmentation checkpoint:
  Filename: seg_best_dice.pt
  SHA256:   2cea6dc1606958dbdb2aea50748b73ff73e604b0c367b3b1588cef7a112c8e06

Primary deformation checkpoint:
  Filename: deform_best_rmse.pth
  SHA256:   3fb84c917953e479e72d59291837bd6b08f40fb197534046b95dc5a452e86b95

Secondary deformation checkpoint:
  Filename: deform_best_model.pth
  SHA256:   d30d3218069a6eb395dacd3ff0dd085a93f6f0102616eed809b90c362b885b40
```

`deform_best_rmse.pth` is the recommended deformation checkpoint and is the
checkpoint used for the final SimCortex v2 paper evaluation.
`deform_best_model.pth` is retained as an alternative checkpoint for
reproducibility and comparison.

The Zenodo archive also contains the frozen segmentation, InitSurf, and
deformation YAML configurations, the official development split, the
560-case evaluation cohort definition, SHA256 checksums, and release
provenance metadata.

The checkpoint hashes above can be used to verify local copies before running
inference.

---

## Configuration

All configurable stages use Hydra YAML files shipped with the package under:

```text
src/simcortex/configs/
  seg/
  initsurf/
  deform/
```

You can configure runs in two main ways.

### 1. Edit the stage YAML

This is recommended for stable experiments and longer runs.

Examples:

```bash
simcortex seg train
simcortex initsurf generate
simcortex deform eval
```

### 2. Use Hydra overrides directly on the CLI

This is recommended for quick tests or one-off experiments.

Examples:

```bash
simcortex seg train outputs.root=/tmp/simcortex_runs/seg/exp01
```

```bash
simcortex deform eval dataset.split_name=test outputs.out_dir=/tmp/deform_eval
```

### 3. Use a separate user config file

If a stage supports a `user_config` field, you can point it to a separate YAML file and keep the packaged defaults unchanged.

Example pattern:

```bash
simcortex deform train user_config=/path/to/my_train.yaml
```

---

## Data and Folder Conventions

You will typically work with **two roots**:

1. **Code repository**  
   This repository contains code, configs, scripts, and package metadata.

2. **Dataset root**  
   Each dataset has its own BIDS-style root with raw data, derivatives, and split files.

Recommended structure:

```text
datasets/<dataset-name>/
  bids/                 # raw BIDS dataset
  derivatives/          # processed outputs (BIDS derivatives)
    freesurfer-7.4.1/
    sc-preproc/
    sc-seg/
    sc-initsurf/
    sc-deform/
  splits/
    <dataset>_split.csv
```

SimCortex reads inputs from `derivatives/` and writes outputs back to `derivatives/` using **BIDS-derivatives-style naming**.

### Typical naming principles

- subject IDs follow `sub-XXXX`
- sessions are typically written as `ses-01`
- MNI outputs are labeled with `space-MNI152`
- segmentation outputs use `desc-seg9_dseg`
- InitSurf produces both mesh outputs and SDF / ribbon outputs
- Deform writes final surface meshes under `sc-deform`

> Important: keep dataset naming and folder organization consistent across stages. In practice, this makes multi-stage and multi-dataset workflows much easier to maintain.

---


## Coordinate Convention

SimCortex reconstructs cortical surfaces in a common **1 mm MNI152 space**. The
validated reference grid has shape `(182, 218, 182)`, and all exported
`space-MNI152` surfaces are stored in physical **RAS coordinates in
millimetres**.

During Stage 1, FreeSurfer surfaces are converted from **tkRAS** to native
scanner RAS and then transformed to MNI152 space. For surface coordinates, use
the explicit transforms written under `xfm/`:

```text
*_from-native_to-MNI152_mode-surface_xfm.txt
*_from-MNI152_to-native_mode-surface_xfm.txt
```

Do not apply the raw ANTs/ITK `*_desc-antsAffine.mat` file directly to PLY
vertices; it follows the ANTs image-transform convention rather than the
surface-point convention used by SimCortex.

InitSurf and Deform convert between world and voxel coordinates internally as
needed, but exported `space-MNI152` meshes are always converted back to
**MNI152 RAS-mm** coordinates.

## Split File Format

A split CSV is required for Segmentation, InitSurf, and Deform.

### Session handling

The current split-file contract is **subject-based**: session is not a split
CSV column. Segmentation, InitSurf, and Deform instead use one
`dataset.session_label` for the entire run (default: `"01"`), and resolve input
and output paths under the corresponding `ses-<label>` directory.

Therefore, a single downstream run does not represent multiple sessions of the
same subject independently through the split CSV. For a multi-session dataset,
run the stage separately for each session and override
`dataset.session_label=<label>` so that it matches the session written by the
preceding stage.

Stage 1 follows the same explicit-session pattern through
`--session-label` (default: `01`) and writes session-qualified derivatives.
Its FreeSurfer input resolver supports both flat subject directories and nested
`sub-*/ses-*` layouts, but the selected session label still defines the
derivative paths produced by that invocation.

### Single-dataset split

Minimum columns:

- `subject` (for example `sub-0001`)
- `split` in `{train, val, test}`

Example:

```csv
subject,split
sub-0001,train
sub-0002,val
sub-0003,test
```

### Multi-dataset split

Add one more column:

- `dataset` (must match the keys used in Hydra config overrides, such as `HCP_YA` or `OASIS1`)

Example:

```csv
subject,split,dataset
sub-100307,test,HCP_YA
sub-101915,test,HCP_YA
sub-0001,test,OASIS1
```

### Important note

In multi-dataset workflows, the `dataset` values in the CSV must match the names used in overrides such as:

```text
dataset.roots.HCP_YA
dataset.seg_roots.HCP_YA
outputs.out_roots.HCP_YA
```

---


## Stage 1 - Preprocessing (FreeSurfer to MNI152)

This stage converts key **FreeSurfer 7.4.1 outputs** into a **BIDS-derivatives-style** layout, applies optional **N4 bias-field correction** to the T1 image, estimates a **linear registration** (**rigid** or **affine**) from native T1w space to **MNI152**, resamples the main volumetric outputs into MNI space, and writes both **native/scanner-space** and **MNI-space** cortical surfaces as ASCII PLY files.

The current implementation is fully Python-based for preprocessing and no longer depends on external command-line tools.


### Inputs

- A FreeSurfer derivatives root containing subject folders with at least:
  - `mri/orig.mgz`
  - `mri/aseg.mgz`
  - `mri/aparc+aseg.mgz`
  - `mri/filled.mgz`
  - `surf/lh.white`, `surf/rh.white`
  - `surf/lh.pial`, `surf/rh.pial` (or `*.pial.T1` if present)
- An MNI152 1 mm reference template image supplied by the user.

The MNI template is an external scientific input and is not bundled with the
SimCortex Python package. Pass its path explicitly with `--mni-template`.

For exact reproducibility of the SimCortex preprocessing setup used during
development and evaluation, the reference image had the following properties:

```text
Shape:      182 × 218 × 182
Resolution: 1 × 1 × 1 mm
SHA256:     9c255b63e668242b94621b446be0c9a04a1216bd25423b746a0c6f9443ea0fc0
```

Use the same template bytes when reproducing previously generated SimCortex
derivatives. The repository does not make a provenance or redistribution claim
for this external reference image.

### Python dependencies

Stage 1 requires Python packages including:

- `antspyx`
- `nibabel`
- `numpy`
- `typer`

### Notes

* **`--transform-type`**: Can be either `rigid` or `affine`.
* **`--n4`**: Enables N4 bias-field correction before registration.
* **`--with-aparc-aseg`** and **`--with-filled`**: Control whether those optional FreeSurfer outputs are also exported and resampled.
* **Surface outputs**: Are written as `.surf.ply`.
* **`--overwrite`**: Recompute outputs even if they already exist.

### Run for all discovered subjects

```bash
simcortex fs-to-mni \
  --freesurfer-root /path/to/datasets/<dataset>/derivatives/freesurfer-7.4.1 \
  --out-deriv-root /path/to/datasets/<dataset>/derivatives/sc-preproc \
  --mni-template /path/to/MNI152_T1_1mm.nii.gz \
  --transform-type affine \
  --n4 \
  --with-aparc-aseg \
  --with-filled \
  -v
```

### Run for selected subjects

```bash
simcortex fs-to-mni \
  --freesurfer-root /path/to/datasets/<dataset>/derivatives/freesurfer-7.4.1 \
  --out-deriv-root /path/to/datasets/<dataset>/derivatives/sc-preproc \
  --mni-template /path/to/MNI152_T1_1mm.nii.gz \
  --participant-label sub-0001 \
  --participant-label sub-0019 \
  --transform-type affine \
  --n4 \
  --with-aparc-aseg \
  --with-filled \
  -v
```

### Output layout
A typical subject output looks like:

```text
sc-preproc/
  dataset_description.json
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_desc-fsraw_T1w.nii.gz
        sub-XXXX_ses-01_desc-preproc_T1w.nii.gz
        sub-XXXX_ses-01_desc-aseg_dseg.nii.gz
        sub-XXXX_ses-01_desc-aparcaseg_dseg.nii.gz
        sub-XXXX_ses-01_desc-filled_dseg.nii.gz

        sub-XXXX_ses-01_space-MNI152_desc-preproc_T1w.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-aseg_dseg.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-aparcaseg_dseg.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-filled_dseg.nii.gz

        sub-XXXX_ses-01_from-T1w_to-MNI152_mode-image_xfm.txt
        sub-XXXX_ses-01_from-MNI152_to-T1w_mode-image_xfm.txt
        sub-XXXX_ses-01_from-T1w_to-MNI152_mode-image_xfm.json

      xfm/
        sub-XXXX_ses-01_from-native_to-MNI152_mode-image_desc-antsAffine.mat
        sub-XXXX_ses-01_from-native_to-MNI152_mode-surface_xfm.txt
        sub-XXXX_ses-01_from-MNI152_to-native_mode-surface_xfm.txt

      surfaces/
        sub-XXXX_ses-01_hemi-L_white.surf.ply
        sub-XXXX_ses-01_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_hemi-R_white.surf.ply
        sub-XXXX_ses-01_hemi-R_pial.surf.ply

        sub-XXXX_ses-01_space-MNI152_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_pial.surf.ply
```


## Stage 2 - Segmentation (3D U-Net, MNI space)

This stage trains and applies a 3D U-Net to predict a **9-class segmentation** in **MNI152 space** using Stage 1 preprocessing outputs.

### Expected inputs from Stage 1

For each subject under `sc-preproc`:

- `..._space-MNI152_desc-preproc_T1w.nii.gz`
- `..._space-MNI152_desc-aparcaseg_dseg.nii.gz`
- `..._space-MNI152_desc-filled_dseg.nii.gz`

### T1w intensity normalization

Before the T1w image is passed to the segmentation network, it is converted to
`float32` and non-finite values are replaced with zero. The 99th percentile is
computed from strictly positive voxels (`T1w > 0`); intensities are then clipped
to `[0, p99]` and divided by `p99`. For standard MNI-space T1w inputs, this keeps
the zero-valued background at zero and scales foreground intensities to `[0, 1]`.
The same normalization routine is used during segmentation training and
inference. If no valid positive foreground percentile is available, the
sanitized volume is returned without percentile scaling.

### Output prediction naming

Segmentation predictions are written under `sc-seg` as:

```text
sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz
```

### Single-dataset training
Use dataset.path and a split CSV for that dataset.

```bash
simcortex seg train \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  outputs.root=/path/to/simcortex-runs/seg/exp01 \
  trainer.use_ddp=false
```
### Multi-dataset training
Use a combined split CSV with a dataset column and provide one root per dataset.

```bash
simcortex seg train \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  outputs.root=/path/to/simcortex-runs/seg/exp01_hcpya+oasis1 \
  trainer.use_ddp=false
```

### Multi-GPU DDP training

```bash
simcortex seg train --torchrun --nproc-per-node 2 \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  outputs.root=/path/to/simcortex-runs/seg/exp01_hcpya+oasis1 \
  trainer.use_ddp=true
```

### Inference

Segmentation inference supports both **single-dataset** and **multi-dataset** execution.

#### Single-dataset inference

Use `dataset.path` and `outputs.out_root` when running inference for one dataset only.

```bash
simcortex seg infer \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/sc-seg
```
In this mode, predictions are written under:
```text
/path/to/datasets/<dataset>/derivatives/sc-seg/sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz
```
Note: for single-dataset inference, dataset.split_file should normally refer to a split CSV for that dataset only.

#### Multi-dataset inference

Use dataset.roots and outputs.out_roots when running inference across multiple datasets from one combined split file.

```bash
simcortex seg infer \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-seg \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-seg
```

### Evaluation

For one dataset:

```bash
simcortex seg eval \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  outputs.pred_root=/path/to/datasets/<dataset>/derivatives/sc-seg \
  outputs.eval_csv=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.csv \
  outputs.eval_xlsx=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.xlsx
```
For multiple datasets:

```bash
simcortex seg eval \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  outputs.pred_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-seg \
  outputs.pred_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-seg \
  outputs.eval_csv=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.csv \
  outputs.eval_xlsx=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.xlsx
```

---

## Stage 3 - Initial Surfaces (InitSurf)

This stage generates initial cortical surfaces from **saved segmentation predictions**. It is not an end-to-end segmentation-to-surface training stage; instead, it consumes Stage 1 and Stage 2 outputs.

### Inputs

- Preprocessing derivatives (`sc-preproc`) for the MNI-aligned T1 image
- Segmentation derivatives (`sc-seg`) for `..._desc-seg9_dseg.nii.gz`
- split CSV

### Output layout

```text
sc-initsurf/
  dataset_description.json
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg_used.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg_cleaned.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-lh_white_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-rh_white_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-lh_pial_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-rh_pial_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-ribbon_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-ribbon_prob.nii.gz
      surfaces/
        sub-XXXX_ses-01_space-MNI152_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_pial.surf.ply
```
### Single-dataset example

```bash
simcortex initsurf generate \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.seg_root=/path/to/datasets/<dataset>/derivatives/sc-seg \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=all \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/sc-initsurf \
  outputs.log_dir=/path/to/simcortex-runs/initsurf/exp01/logs_generate
```

### Multi-dataset example

```bash
simcortex initsurf generate \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  dataset.seg_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-seg \
  dataset.seg_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-seg \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-initsurf \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-initsurf \
  outputs.log_dir=/path/to/simcortex-runs/initsurf/exp01/logs_generate
```

### Typical runtime

A typical runtime is approximately 70–110 s / subject with `n_workers: 8`,
depending on hardware, I/O speed, and dataset characteristics.
---

## Stage 4 - Deformation (Deform)

This stage deforms the InitSurf meshes toward the MNI-aligned FreeSurfer target surfaces.

### Inputs

- Preprocessing derivatives (`sc-preproc`) containing:
  - MNI T1
  - target FreeSurfer surfaces in MNI space
- InitSurf derivatives (`sc-initsurf`) containing:
  - initial surfaces
  - ribbon probability volumes
- split CSV

### Deformation input normalization

The deformation stage normalizes the MRI and ribbon-probability channels
separately. For the MRI channel, the mean and standard deviation are computed
from nonzero voxels when at least 100 nonzero voxels are available; otherwise,
they are computed from the full volume. The normalized MRI is

```text
(T1w - mean) / max(std, 1e-6)
```

and is stored as `float32`. This same MRI normalization is used for deformation
training and inference, before the center crop/pad step.

The ribbon-probability channel is **not** z-score normalized. Non-finite values
are replaced, optional values below `prob_clip_min` are set to zero, values are
clipped to `[0, prob_clip_max]`, and `prob_gamma` is applied when it differs from
1.0. With the shipped deformation defaults, `prob_clip_min=0.0`,
`prob_clip_max=1.0`, and `prob_gamma=1.0`.

These stage-specific normalization steps are separate from the optional N4
bias-field correction performed during Stage 1 preprocessing.

### Outputs

During **inference**, the stage writes deformed surfaces under `sc-deform`:

```text
sc-deform/
  dataset_description.json
  sub-XXXX/
    ses-01/
      surfaces/
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-R_pial.surf.ply
```

### Training example
Use dataset.path and dataset.initsurf_root for one dataset.
```bash
simcortex deform train \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.initsurf_root=/path/to/datasets/<dataset>/derivatives/sc-initsurf \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  outputs.root=/path/to/simcortex-runs/deform/exp01
```
### Multi-dataset training
Use a combined split CSV with a dataset column and provide one preprocessing root and one InitSurf root per dataset.
```bash
simcortex deform train --torchrun --nproc-per-node 2 \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  dataset.initsurf_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-initsurf \
  dataset.initsurf_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-initsurf \
  outputs.root=/path/to/simcortex-runs/deform/exp01_hcpya+oasis1
```

### Inference

For one dataset:

```bash
simcortex deform infer \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.initsurf_root=/path/to/datasets/<dataset>/derivatives/sc-initsurf \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/deform_best_rmse.pth \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/sc-deform
```
For multiple datasets:
```bash
simcortex deform infer \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  dataset.initsurf_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-initsurf \
  dataset.initsurf_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-initsurf \
  model.ckpt_path=/path/to/deform_best_rmse.pth \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-deform \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-deform
```

### Evaluation

For one dataset:

```bash
simcortex deform eval \
  dataset.path=/path/to/datasets/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  outputs.pred_root=/path/to/datasets/<dataset>/derivatives/sc-deform \
  outputs.out_dir=/path/to/simcortex-runs/deform/exp01/eval_test
```
For multiple datasets:
```bash
simcortex deform eval \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-preproc \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-preproc \
  outputs.pred_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/sc-deform \
  outputs.pred_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/sc-deform \
  outputs.out_dir=/path/to/simcortex-runs/deform/exp01_hcpya+oasis1/eval_test
```

### Evaluation outputs

This stage writes the following Excel reports:

- `surface_metrics.xlsx`
- `collision_metrics.xlsx`
- `collision_metrics_enhanced.xlsx`
- `collision_summary.xlsx`

---

## Docker

The official SimCortex v2.0.0 container is published on Docker Hub:

- **Docker Hub:** [kavehmoradkhani/simcortex](https://hub.docker.com/r/kavehmoradkhani/simcortex)
- **Versioned image:** `kavehmoradkhani/simcortex:2.0.0`
- **Latest alias:** `kavehmoradkhani/simcortex:latest`

Pull the versioned release:

```bash
docker pull kavehmoradkhani/simcortex:2.0.0
```

For exact reproducibility, the SimCortex v2.0.0 image can be pulled by its
immutable registry digest:

```bash
docker pull kavehmoradkhani/simcortex@sha256:b4dfdfa36a10eeedadd1eb4d62adfe751025e45336e6fdfeab717a2f6245ec25
```

The published `2.0.0` and `latest` tags currently resolve to this same image
digest.

The release image was built from SimCortex source revision:

```text
59f167e02ebe9680f9d9f2293827f7983d3325a3
```

The container supports all four stages:

- **Stage 1 — Preprocessing**
- **Stage 2 — Segmentation**
- **Stage 3 — InitSurf**
- **Stage 4 — Deform**

Basic CLI check:

```bash
docker run --rm \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex --help
```

GPU visibility check:

```bash
docker run --rm --gpus all \
  kavehmoradkhani/simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

The validated release environment includes Python 3.10, PyTorch 2.1.0 with
CUDA 12.1, PyTorch3D 0.7.8, and ANTsPy 0.6.1.

For full Docker usage, including:

- mounting datasets, outputs, checkpoints, and templates;
- running with the host UID/GID;
- passing Hydra overrides;
- inspecting packaged YAML configurations;
- Stage 1–4 Docker examples;
- GPU execution;
- shared-server and HPC usage;
- Apptainer/Singularity usage;

see [`docker/README.md`](docker/README.md).

---

## Citation

If you use SimCortex in your research, please cite the published SimCortex
paper:

```bibtex
@inproceedings{Moradkhani2025SimCortex,
  author    = {Moradkhani, Kaveh and Rushmore, R. Jarrett and Bouix, Sylvain},
  title     = {{SimCortex}: Collision-free Simultaneous Cortical Surfaces Reconstruction},
  booktitle = {International Workshop on Shape in Medical Imaging},
  series    = {Lecture Notes in Computer Science},
  volume    = {16171},
  pages     = {347--359},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  year      = {2025},
  doi       = {10.1007/978-3-032-06774-6_26}
}
```

The SimCortex v2 journal manuscript, *SimCortex v2: Joint Cortical Surface
Reconstruction with Near-Zero Collisions and Self-Intersections*, is currently
submitted to *Medical Image Analysis*. Its final citation will be added when a
stable public bibliographic record is available.

### Model and data releases

The finalized SimCortex v2 model artifacts are available on Zenodo:

- Moradkhani, K. & Bouix, S. (2026). *SimCortex v2: Pre-trained Models and
  Reproducibility Files* (Version v2). Zenodo.
  https://doi.org/10.5281/zenodo.22710781

The associated reproducibility dataset is available through FRDR:

- Moradkhani, K. & Bouix, S. (2026). *SimCortex Reproducibility Dataset for
  Cortical Surface Reconstruction*. Federated Research Data Repository.
  https://doi.org/10.20383/103.01769

Repository citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

## License

The SimCortex source code in this repository is licensed under the
[Apache License 2.0](LICENSE).

The separately archived SimCortex v2 pre-trained model and reproducibility
files on Zenodo are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Dataset access and reuse remain subject to the terms associated with the
corresponding source datasets and the FRDR record.
