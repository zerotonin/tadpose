# TadPose

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

## Citation

If you use TadPose in your research, please cite:

> Matthews, A.R.H., Beck, C., & Geurten, B.R.H. (2026). TadPose:
> Automated behavioural phenotyping of Xenopus laevis tadpoles from
> 24-well plate video. [Software].
> https://github.com/zerotonin/tadpose

## License

MIT — see [LICENSE](LICENSE).
