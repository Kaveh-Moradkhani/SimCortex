# SimCortex v2.0

SimCortex v2.0 is the journal-version implementation of SimCortex: a modular and reproducible framework for cortical surface reconstruction in **MNI152 space**. It provides four practical stages that can be run independently or as a full pipeline:

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

The **project itself** provides all four stages. However, the **main Docker image** is intended primarily for **Stage 1 (FreeSurfer to MNI152 preprocessing)**, **Stage 2 (Segmentation)**, **Stage 3 (InitSurf)**, and **Stage 4 (Deform)**.

---

## Table of Contents

- [Installation](#installation)
- [Pre-trained Weights and Official Splits](#pre-trained-weights-and-official-splits)
- [Configuration](#configuration)
- [Data and Folder Conventions](#data-and-folder-conventions)
- [Split File Format](#split-file-format)
- [Recommended Workflow Order](#recommended-workflow-order)
- [Stage 1 - Preprocessing (FreeSurfer to MNI152)](#stage-1---preprocessing-freesurfer-to-mni152)
- [Stage 2 - Segmentation (3D U-Net, MNI space)](#stage-2---segmentation-3d-u-net-mni-space)
- [Stage 3 - Initial Surfaces (InitSurf)](#stage-3---initial-surfaces-initsurf)
- [Stage 4 - Deformation (Deform)](#stage-4---deformation-deform)
- [Docker](#docker)
- [License](#license)

---

## Installation

From the repository root:

```bash
pip install -e .
simcortex --help
simcortex fs-to-mni --help
simcortex seg --help
simcortex initsurf --help
simcortex deform --help
```

### Recommended environment

- Python 3.10+
- PyTorch
- PyTorch3D
- MONAI
- `nibabel`, `numpy`, `scipy`, `scikit-image`
- `pandas`, `openpyxl`
- `trimesh`, `tqdm`, `python-fcl`
- `hydra-core`, `omegaconf`, `typer`

### Optional / stage-specific dependencies

- **ANTsPy** (`antspyx`) for **Stage 1 preprocessing** (N4 bias correction, linear registration, and image resampling in Python)
- **FreeSurfer outputs** are required as **inputs** to Stage 1 (for example `orig.mgz`, `aseg.mgz`, `aparc+aseg.mgz`, `filled.mgz`, and cortical surfaces under `surf/`)
- **python-fcl** for collision-related metrics
- **pymeshlab** for optional mesh-processing utilities

---

## Pre-trained Weights and Official Splits

Official pre-trained checkpoints and dataset split files are available on Zenodo:

- **Zenodo record:** [SimCortex v2.0: Pre-trained Models and Dataset Splits](https://zenodo.org/records/18974730)

This record currently provides the packaged segmentation weights, deformation weights, and split CSV files used for evaluation and reproducible experiments.

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
    simcortex-preproc-0.1/
    simcortex-seg-0.1/
    simcortex-initsurf-0.1/
    simcortex-deform-0.1/
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
- Deform writes final surface meshes under `simcortex-deform-*`

> Important: keep dataset naming and folder organization consistent across stages. In practice, this makes multi-stage and multi-dataset workflows much easier to maintain.

---

## Split File Format

A split CSV is required for Segmentation, InitSurf, and Deform.

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

## Recommended Workflow Order

A typical full workflow is:

1. Run **Preprocessing** for each dataset to create `simcortex-preproc-0.1`
2. Train **Segmentation** and select a checkpoint
3. Run **Segmentation inference** to create `simcortex-seg-0.1`
4. Run **InitSurf** to create `simcortex-initsurf-0.1`
5. Train, infer, and evaluate **Deformation** to create `simcortex-deform-0.1`

This staged design is intentional and makes debugging, ablation, and evaluation easier.

---

## Stage 1 - Preprocessing (FreeSurfer to MNI152)

This stage converts key **FreeSurfer 7.4.1 outputs** into a **BIDS-derivatives-style** layout, applies optional **N4 bias-field correction** to the T1 image, estimates a **linear registration** (**rigid** or **affine**) from native T1w space to **MNI152**, resamples the main volumetric outputs into MNI space, and writes both **native/scanner-space** and **MNI-space** cortical surfaces as ASCII PLY files.

The current implementation is fully Python-based for preprocessing and no longer depends on external command-line tools.

### What this stage does

For each FreeSurfer subject, Stage 1 performs the following steps:

1. Export native FreeSurfer volumes from MGZ to NIfTI using `nibabel`
2. Optionally apply **N4 bias-field correction** to `orig.mgz` using **ANTsPy**
3. Estimate a **linear transform** (`rigid` or `affine`) from native T1w to the MNI template using **ANTsPy**
4. Resample FreeSurfer-derived volumes into **MNI152 space**
5. Read FreeSurfer cortical surfaces directly in Python, convert them from **surface/tkRAS** to **scanner/world RAS**, and write:
   - native/scanner-space PLY surfaces
   - MNI-space PLY surfaces

### Inputs

- A FreeSurfer derivatives root containing subject folders with at least:
  - `mri/orig.mgz`
  - `mri/aseg.mgz`
  - `mri/aparc+aseg.mgz`
  - `mri/filled.mgz`
  - `surf/lh.white`, `surf/rh.white`
  - `surf/lh.pial`, `surf/rh.pial` (or `*.pial.T1` if present)
- An MNI template image, for example:

```text
src/MNI152_T1_1mm.nii.gz
```

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
  --out-deriv-root /path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  --mni-template /path/to/SimCortex/src/MNI152_T1_1mm.nii.gz \
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
  --out-deriv-root /path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  --mni-template /path/to/SimCortex/src/MNI152_T1_1mm.nii.gz \
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
simcortex-preproc-0.1/
  dataset_description.json
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_desc-fsraw_T1w.nii.gz
        sub-XXXX_ses-01_desc-preproc_T1w.nii.gz
        sub-XXXX_ses-01_desc-aseg_dseg.nii.gz
        sub-XXXX_ses-01_desc-aparc+aseg_dseg.nii.gz
        sub-XXXX_ses-01_desc-filled_T1w.nii.gz

        sub-XXXX_ses-01_space-MNI152_desc-preproc_T1w.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-aseg_dseg.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-aparc+aseg_dseg.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-filled_T1w.nii.gz

        sub-XXXX_ses-01_from-T1w_to-MNI152_mode-image_xfm.txt
        sub-XXXX_ses-01_from-MNI152_to-T1w_mode-image_xfm.txt
        sub-XXXX_ses-01_from-T1w_to-MNI152_mode-image_xfm.json
        sub-XXXX_ses-01_from-T1w_to-MNI152_mode-image_desc-antsAffine.mat

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

### What this stage provides to later stages

Stage 1 provides the MNI-aligned T1w image and MNI-aligned FreeSurfer-derived target volumes and surfaces used by later stages of the SimCortex pipeline.

---

## Stage 2 - Segmentation (3D U-Net, MNI space)

This stage trains and applies a 3D U-Net to predict a **9-class segmentation** in **MNI152 space** using Stage 1 preprocessing outputs.

### Expected inputs from Stage 1

For each subject under `simcortex-preproc-*`:

- `..._space-MNI152_desc-preproc_T1w.nii.gz`
- `..._space-MNI152_desc-aparc+aseg_dseg.nii.gz`
- `..._space-MNI152_desc-filled_T1w.nii.gz`

### Output prediction naming

Segmentation predictions are written under `simcortex-seg-*` as:

```text
sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz
```

### Single-dataset training
Use dataset.path and a split CSV for that dataset.

```bash
simcortex seg train \
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  outputs.root=/path/to/simcortex-runs/seg/exp01 \
  trainer.use_ddp=false
```
### Multi-dataset training
Use a combined split CSV with a dataset column and provide one root per dataset.

```bash
simcortex seg train \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  outputs.root=/path/to/simcortex-runs/seg/exp01_hcpya+oasis1 \
  trainer.use_ddp=false
```

### Multi-GPU DDP trainingP

```bash
simcortex seg train --torchrun --nproc-per-node 2 \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  outputs.root=/path/to/simcortex-runs/seg/exp01_hcpya+oasis1 \
  trainer.use_ddp=true
```

### Inference

Segmentation inference supports both **single-dataset** and **multi-dataset** execution.

#### Single-dataset inference

Use `dataset.path` and `outputs.out_root` when running inference for one dataset only.

```bash
simcortex seg infer \
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/simcortex-seg-0.1
```
In this mode, predictions are written under:
```text
/path/to/datasets/<dataset>/derivatives/simcortex-seg-0.1/sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz
```
Note: for single-dataset inference, dataset.split_file should normally refer to a split CSV for that dataset only.

### Multi-dataset inference 

Use dataset.roots and outputs.out_roots when running inference across multiple datasets from one combined split file.

```bash
simcortex seg infer \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-seg-0.1 \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-seg-0.1
```

### Evaluation

For one dataset:

```bash
simcortex seg eval \
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  outputs.pred_root=/path/to/datasets/<dataset>/derivatives/simcortex-seg-0.1 \
  outputs.eval_csv=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.csv \
  outputs.eval_xlsx=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.xlsx
```
For multiple datasets:

```bash
simcortex seg eval \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  outputs.pred_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-seg-0.1 \
  outputs.pred_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-seg-0.1 \
  outputs.eval_csv=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.csv \
  outputs.eval_xlsx=/path/to/simcortex-runs/seg/exp01/evals/seg_eval_test.xlsx
```

---

## Stage 3 - Initial Surfaces (InitSurf)

This stage generates initial cortical surfaces from **saved segmentation predictions**. It is not an end-to-end segmentation-to-surface training stage; instead, it consumes Stage 1 and Stage 2 outputs.

### Inputs

- Preprocessing derivatives (`simcortex-preproc-*`) for the MNI-aligned T1 image
- Segmentation derivatives (`simcortex-seg-*`) for `..._desc-seg9_dseg.nii.gz`
- split CSV

### Output layout

```text
simcortex-initsurf-0.1/
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
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.seg_root=/path/to/datasets/<dataset>/derivatives/simcortex-seg-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=all \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/simcortex-initsurf-0.1 \
  outputs.log_dir=/path/to/simcortex-runs/initsurf/exp01/logs_generate
```

### Multi-dataset example

```bash
simcortex initsurf generate \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  dataset.seg_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-seg-0.1 \
  dataset.seg_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-seg-0.1 \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-initsurf-0.1 \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-initsurf-0.1 \
  outputs.log_dir=/path/to/simcortex-runs/initsurf/exp01/logs_generate
```

### Typical runtime

A typical runtime is approximately 70–110 s / subject with `n_workers: 8`,
depending on hardware, I/O speed, and dataset characteristics.
---

## Stage 4 - Deformation (Deform)

This stage deforms the InitSurf meshes toward the MNI-aligned FreeSurfer target surfaces.

### Inputs

- Preprocessing derivatives (`simcortex-preproc-*`) containing:
  - MNI T1
  - target FreeSurfer surfaces in MNI space
- InitSurf derivatives (`simcortex-initsurf-*`) containing:
  - initial surfaces
  - ribbon probability volumes
- split CSV

### Outputs

During **inference**, the stage writes deformed surfaces under `simcortex-deform-*`:

```text
simcortex-deform-0.1/
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
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.initsurf_root=/path/to/datasets/<dataset>/derivatives/simcortex-initsurf-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  outputs.root=/path/to/simcortex-runs/deform/exp01
```
### Multi-dataset training
Use a combined split CSV with a dataset column and provide one preprocessing root and one InitSurf root per dataset.
```bash
simcortex deform train --torchrun --nproc-per-node 2 \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  dataset.initsurf_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-initsurf-0.1 \
  dataset.initsurf_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-initsurf-0.1 \
  outputs.root=/path/to/simcortex-runs/deform/exp01_hcpya+oasis1
```

### Inference

For one dataset:

```bash
simcortex deform infer \
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.initsurf_root=/path/to/datasets/<dataset>/derivatives/simcortex-initsurf-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/deform_best_rmse.pth \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/simcortex-deform-0.1
```
For multiple datasets:
```bash
simcortex deform infer \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  dataset.initsurf_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-initsurf-0.1 \
  dataset.initsurf_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-initsurf-0.1 \
  model.ckpt_path=/path/to/deform_best_rmse.pth \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-deform-0.1 \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-deform-0.1
```

### Evaluation

For one dataset:

```bash
simcortex deform eval \
  dataset.path=/path/to/datasets/<dataset>/derivatives/simcortex-preproc-0.1 \
  dataset.split_file=/path/to/datasets/<dataset>/splits/dataset_split.csv \
  dataset.split_name=test \
  outputs.pred_root=/path/to/datasets/<dataset>/derivatives/simcortex-deform-0.1 \
  outputs.out_dir=/path/to/simcortex-runs/deform/exp01/eval_test
```
For multiple datasets:
```bash
simcortex deform eval \
  dataset.split_file=/path/to/datasets/splits/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-preproc-0.1 \
  outputs.pred_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/simcortex-deform-0.1 \
  outputs.pred_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/simcortex-deform-0.1 \
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


Docker support is provided as an **execution environment** for the SimCortex pipeline.

The main Docker image is intended to support **all four stages**:

- **Stage 1 — Preprocessing**
- **Stage 2 — Segmentation**
- **Stage 3 — InitSurf**
- **Stage 4 — Deform**

Basic CLI check:

```bash
docker run --rm simcortex:2.0.0 simcortex --help
```

GPU visibility check:

```bash
docker run --rm --gpus all simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

For full Docker usage, including:

- the published Docker image on Docker Hub: [kavehmoradkhani/simcortex](https://hub.docker.com/r/kavehmoradkhani/simcortex)
- running as the host user with `--user $(id -u):$(id -g)`
- mounting datasets and outputs with `-v`
- passing Hydra overrides from the CLI
- extracting packaged YAML configs from inside the container
- using custom edited YAML files
- stage-specific Docker command examples for **Stage 1–4**
- shared server and Apptainer notes

see:

```text
docker/README.md
```

---

## License

See the repository `LICENSE` file.
