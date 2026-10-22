# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — moseq_fit                                             ║
# ║  « PCA and the kappa scan, inside the kpms env »                 ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  tadpose moseq-fit pca | kappa-scan: thin driver around          ║
# ║  keypoint_moseq (imported lazily, never a package dependency).   ║
# ║  pca: project config, PCA on the fit set, latent dim from the    ║
# ║  scree.  kappa-scan: one AR-only fit per kappa, reporting the    ║
# ║  median syllable duration that kappa produces.  calibrate | full ║
# ║  continue an AR-only start as the full keypoint-SLDS.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``tadpose moseq-fit``: PCA, kappa scan and full-model fits on a fit set (kpms env)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose import moseq_constants as mc
from tadpose.moseq_export import load_fit_set

log = logging.getLogger("tadpose.moseq_fit")

SCREE_NAME: str = "pca_scree.csv"
SCAN_DIR: str = "kappa_scan"
FULL_DIR: str = "full_fits"


def main(argv: list[str] | None = None) -> None:
    """Run one step on one fit set: ``pca``, ``kappa-scan``, ``calibrate`` or ``full``."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.step == "pca":
        fit_project_pca(args.fit_set, args.project)
    elif args.step == "kappa-scan":
        kappas = [args.kappa] if args.kappa is not None else _pick(mc.KAPPA_SCAN, args.slurm_array)
        for kappa in kappas:
            scan_kappa(args.fit_set, args.project, kappa, args.iters or mc.AR_ONLY_ITERS)
    elif args.step == "calibrate":
        for factor in _pick(mc.FULL_KAPPA_FACTORS, args.slurm_array):
            name = build_full_model_name(args.ar_kappa, args.ar_kappa * factor, args.iters or mc.CALIBRATION_ITERS)
            if (args.project / FULL_DIR / f"{name}.json").exists():
                log.info("%s already fitted, skipped", name)
                continue
            fit_full_model(args.fit_set, args.project, args.ar_kappa, args.ar_kappa * factor,
                           args.iters or mc.CALIBRATION_ITERS)
    else:
        fit_full_model(args.fit_set, args.project, args.ar_kappa, args.kappa,
                       args.iters or mc.FULL_MODEL_ITERS)


def _pick(values: tuple[float, ...], is_array: bool) -> list[float]:
    """All *values*, or only the one of this SLURM array task."""
    return [values[int(os.environ["SLURM_ARRAY_TASK_ID"])]] if is_array else list(values)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tadpose moseq-fit",
        description="keypoint-MoSeq steps on a fit set; run inside the keypoint_moseq env.",
    )
    parser.add_argument("step", choices=("pca", "kappa-scan", "calibrate", "full"))
    parser.add_argument("--fit-set", type=Path, required=True, help="fitset_seed<n>.npz")
    parser.add_argument("--project", type=Path, required=True, help="kpms project folder")
    parser.add_argument("--kappa", type=float,
                        help="kappa-scan: one kappa (default: KAPPA_SCAN); full: the full-model kappa")
    parser.add_argument("--ar-kappa", type=float, help="calibrate, full: kappa of the AR-only start")
    parser.add_argument("--slurm-array", action="store_true",
                        help="fit only the SLURM_ARRAY_TASK_ID-th KAPPA_SCAN / FULL_KAPPA_FACTORS entry")
    parser.add_argument("--iters", type=int,
                        help="iterations (default: AR_ONLY_ITERS, CALIBRATION_ITERS or FULL_MODEL_ITERS)")
    args = parser.parse_args(argv)
    if args.step in ("calibrate", "full") and args.ar_kappa is None:
        parser.error(f"{args.step} needs --ar-kappa")
    if args.step == "full" and args.kappa is None:
        parser.error("full needs --kappa")
    return args


# ┌──────────────────────────────────────────────────────────────┐
# │ Project  « config, PCA, scree »                              │
# └──────────────────────────────────────────────────────────────┘


def fit_project_pca(fit_set: Path, project: Path) -> int:
    """Create the kpms project, fit and save the PCA; return the latent dim.

    The latent dim is ``LATENT_DIM`` when set, else the scree rule.
    """
    import keypoint_moseq as kpms

    kpms.setup_project(str(project), overwrite=True, **build_project_options())
    data, _ = _format(kpms, fit_set, project)
    # Filled points carry no evidence of the pose: fit on confident frames only.
    pca = kpms.fit_pca(**data, **kpms.load_config(str(project)), exclude_outliers_for_pca=True)
    kpms.save_pca(pca, str(project))
    scree = build_scree(pca.explained_variance_ratio_)
    scree.to_csv(project / SCREE_NAME, index=False, encoding="utf-8")
    by_scree = choose_latent_dim(scree.cumulative.to_numpy())
    latent_dim = mc.LATENT_DIM if mc.LATENT_DIM is not None else by_scree
    kpms.update_config(str(project), latent_dim=latent_dim)
    log.info("PCA: scree rule (%.0f %%) gives %d; latent_dim %d explains %.1f %%",
             100 * mc.PCA_VARIANCE_TARGET, by_scree, latent_dim,
             100 * scree.cumulative.iloc[latent_dim - 1])
    return latent_dim


def build_project_options() -> dict:
    """kpms config entries for the tadpole skeleton."""
    return {
        "bodyparts": list(mc.BODYPARTS),
        "use_bodyparts": list(mc.BODYPARTS),
        "skeleton": [list(edge) for edge in mc.SKELETON],
        "anterior_bodyparts": list(mc.ANTERIOR_BODYPARTS),
        "posterior_bodyparts": list(mc.POSTERIOR_BODYPARTS),
        "fps": mc.FPS,
    }


def build_scree(explained_ratio: np.ndarray) -> pd.DataFrame:
    ratio = np.asarray(explained_ratio, dtype=float)
    return pd.DataFrame({"component": np.arange(1, ratio.size + 1),
                         "explained": ratio, "cumulative": np.cumsum(ratio)})


def choose_latent_dim(cumulative: np.ndarray) -> int:
    """Components needed for PCA_VARIANCE_TARGET, within MIN/MAX_LATENT_DIM."""
    reached = int(np.searchsorted(cumulative, mc.PCA_VARIANCE_TARGET)) + 1
    return max(min(reached, mc.MAX_LATENT_DIM, len(cumulative)), mc.MIN_LATENT_DIM)


def _format(kpms, fit_set: Path, project: Path) -> tuple[dict, dict]:
    # Exports are float32 to halve their size; kpms runs JAX in 64-bit.
    coords, conf = (
        {name: array.astype(np.float64) for name, array in d.items()} for d in load_fit_set(fit_set)
    )
    config = kpms.load_config(str(project))
    kpms.check_nan_proportions(coords, config["bodyparts"])
    return kpms.format_data(coords, conf, **config)


# ┌──────────────────────────────────────────────────────────────┐
# │ Kappa scan  « AR-only fits, syllable durations »             │
# └──────────────────────────────────────────────────────────────┘


def scan_kappa(fit_set: Path, project: Path, kappa: float, iters: int) -> dict:
    """Fit the AR-HMM at one kappa; write and return its syllable statistics."""
    import keypoint_moseq as kpms

    data, metadata = _format(kpms, fit_set, project)
    name = build_scan_model_name(kappa)
    model = _fit_ar_only(kpms, data, metadata, project, kappa, iters, name)
    durations = compute_syllable_durations(np.asarray(model["states"]["z"]), np.asarray(data["mask"]))
    summary = summarise_durations(durations, kappa, iters)
    _save_summary(summary, project / SCAN_DIR / f"{name}.json")
    log.info("kappa %.0e: median syllable %.1f frames (%.0f ms), %d syllables >= 1 %% of frames",
             kappa, summary["median_frames"], summary["median_ms"], summary["n_syllables_1pct"])
    return summary


def build_scan_model_name(kappa: float) -> str:
    return f"ar_kappa_{kappa:.2g}"


def _fit_ar_only(kpms, data: dict, metadata: tuple, project: Path, kappa: float, iters: int,
                 name: str) -> dict:
    config = kpms.load_config(str(project))
    model = kpms.init_model(data, pca=kpms.load_pca(str(project)), **config)
    model = kpms.update_hypparams(model, kappa=kappa)
    model, _ = kpms.fit_model(model, data, metadata, str(project), model_name=name, ar_only=True,
                              num_iters=iters, generate_progress_plots=False)
    return model


def _save_summary(summary: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")


def compute_syllable_durations(z: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Lengths of constant-label runs inside the valid frames of each segment.

    ``z`` (segments, T - lags) is aligned to the end of ``mask`` (segments, T):
    the AR model has no state for the first lag frames.  Runs break at
    invalid frames and at segment ends.
    """
    valid = mask[:, mask.shape[1] - z.shape[1]:].astype(bool)
    lengths, labels = [], []
    for seg_z, seg_valid in zip(z, valid):
        labelled = np.where(seg_valid, seg_z, -1)
        edges = np.flatnonzero(np.diff(labelled)) + 1
        starts = np.concatenate([[0], edges])
        stops = np.concatenate([edges, [labelled.size]])
        keep = labelled[starts] >= 0
        lengths.append((stops - starts)[keep])
        labels.append(labelled[starts][keep])
    return pd.DataFrame({"label": np.concatenate(labels), "n_frames": np.concatenate(lengths)})


def summarise_durations(durations: pd.DataFrame, kappa: float, iters: int) -> dict:
    frames_per_label = durations.groupby("label").n_frames.sum()
    share = frames_per_label / frames_per_label.sum()
    median = float(durations.n_frames.median())
    return {
        "kappa": kappa,
        "iters": iters,
        "median_frames": median,
        "median_ms": 1000 * median / mc.FPS,
        "mean_frames": float(durations.n_frames.mean()),
        "single_frame_share": float((durations.n_frames == 1).mean()),
        "n_syllables_used": int(share.size),
        "n_syllables_1pct": int((share >= 0.01).sum()),
        "n_bouts": int(len(durations)),
    }


# ┌──────────────────────────────────────────────────────────────┐
# │ Full model  « keypoint-SLDS, calibration and final fit »     │
# └──────────────────────────────────────────────────────────────┘


def fit_full_model(fit_set: Path, project: Path, ar_kappa: float, kappa: float, iters: int) -> dict:
    """Continue an AR-only start as the full model; write and return its syllable statistics.

    The start is the kappa scan's checkpoint at *ar_kappa* when there is one,
    else a fresh AR-only fit saved under this model's own name.
    """
    import keypoint_moseq as kpms

    data, metadata = _format(kpms, fit_set, project)
    name = build_full_model_name(ar_kappa, kappa, iters)
    scan_checkpoint = project / build_scan_model_name(ar_kappa) / "checkpoint.h5"
    if scan_checkpoint.exists():
        model, _, _, start_iter = kpms.load_checkpoint(path=str(scan_checkpoint))
    else:
        model = _fit_ar_only(kpms, data, metadata, project, ar_kappa, mc.AR_ONLY_ITERS, name)
        start_iter = mc.AR_ONLY_ITERS
    model = kpms.update_hypparams(model, kappa=kappa)
    model, _ = kpms.fit_model(model, data, metadata, str(project), model_name=name, ar_only=False,
                              start_iter=start_iter, num_iters=start_iter + iters,
                              generate_progress_plots=False)
    kpms.reindex_syllables_in_checkpoint(str(project), name)
    durations = compute_syllable_durations(np.asarray(model["states"]["z"]), np.asarray(data["mask"]))
    summary = {"ar_kappa": ar_kappa, **summarise_durations(durations, kappa, iters)}
    _save_summary(summary, project / FULL_DIR / f"{name}.json")
    log.info("full model (AR start %.0e) kappa %.0e: mean syllable %.1f frames, median %.1f, "
             "%d syllables >= 1 %% of frames", ar_kappa, kappa, summary["mean_frames"],
             summary["median_frames"], summary["n_syllables_1pct"])
    return summary


def build_full_model_name(ar_kappa: float, kappa: float, iters: int) -> str:
    return f"full_ar{ar_kappa:.2g}_k{kappa:.2g}_{iters}it"
