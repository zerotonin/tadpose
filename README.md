# TadPose

[![tests](https://github.com/zerotonin/tadpose/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/tadpose/actions/workflows/tests.yml)
[![docs](https://github.com/zerotonin/tadpose/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/tadpose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](pyproject.toml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pose estimation: DeepLabCut](https://img.shields.io/badge/pose%20estimation-DeepLabCut-1f4b99.svg)](https://github.com/DeepLabCut/DeepLabCut)
[![Statistics: reRandomStats](https://img.shields.io/badge/statistics-reRandomStats-009E73.svg)](https://github.com/zerotonin/reRandomStats)

**Automated behavioural phenotyping of *Xenopus laevis* tadpoles from
24-well plate video.**

TadPose provides a pipeline for extracting posture dynamics and velocity
features from multi-well plate recordings of tadpoles, enabling
unsupervised behavioural clustering to quantify seizure phenotypes in
models of developmental and epileptic encephalopathies (DEE).

> This pipeline was developed at the Department of Zoology, University
> of Otago.  Early development took place under
> [alexrhmatthews/tadpole_wells](https://github.com/alexrhmatthews/tadpole_wells) and
> [zerotonin/24well_pipe](https://github.com/zerotonin/24well_pipe);
> the codebase was reorganised and renamed to TadPose for publication.

## Pipeline overview

1. **Well detection** — Hough circle transform with eigenvector-corrected
   centres to localise all 24 wells despite lens distortion
   (`tadpose.well_detection`).
2. **Video segmentation** — Split full-plate recordings into per-well
   clips for downstream pose estimation (`tadpose.video_segmentation`).
3. **Pose estimation** — Seven anatomical landmarks tracked via
   DeepLabCut: left eye, right eye, tail base, three tail segments,
   tail tip (`tadpose.dlc_runner`).
4. **Feature extraction** — Body-centric velocity decomposition
   (thrust, yaw, slip) and posture dynamics in a frons-aligned
   coordinate system (`tadpose.feature_extraction`).
5. **Feature cleaning & normalisation** — Artefact removal via
   distribution-based thresholds, z-score standardisation
   (`tadpose.feature_cleaning`, `tadpose.normalisation`).
6. **Behavioural clustering** — GPU-accelerated k-means via
   [STAG](https://github.com/zerotonin/stag) on combined velocity +
   posture dynamics features, yielding 36 stable behavioural prototypes
   (`tadpose.clustering`).
7. **Post-clustering analysis** — Proportion statistics, significance
   testing, centroid visualisation (`tadpose.analysis`).

## Installation

```bash
conda env create -f environment.yml
conda activate tadpose
pip install -e .
```

### Configure machine-specific paths (required first step)

TadPose hard-codes no absolute paths.  Every data root, interpreter, and
HPC setting is read from a gitignored `local_paths.json`, resolved against
the committed template.  Copy it and edit the `local` profile (and, on the
cluster, the `hpc` profile) **before running anything**:

```bash
cp local_paths.template.json local_paths.json
# then edit local_paths.json: set data_root, code_root, python_interpreter, …
```

Resolution order for the data root: `$TADPOSE_DATA_ROOT` → the active
profile's `data_root` in `local_paths.json` → an in-repo `data/` symlink.
A missing `local_paths.json` fails loudly and names the template to copy.
SLURM submit scripts read the same file via `slurm/load_paths.sh`.

## Citation

If you use TadPose in your research, please cite:

> Matthews, A.R.H., Beck, C., & Geurten, B.R.H. (2026). TadPose:
> Automated behavioural phenotyping of Xenopus laevis tadpoles from
> 24-well plate video. [Software].
> https://github.com/zerotonin/tadpose

## License

MIT — see [LICENSE](LICENSE).
