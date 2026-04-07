# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — normalisation                                       ║
# ║  « z-scoring 10^7 observations without breaking a sweat »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Compute, save, load, and apply z-score normalisation for      ║
# ║  clustering feature matrices.                                   ║
# ║                                                                 ║
# ║  Merged from getMuSigmaForZscore.py + zscore_with_predefined_  ║
# ║  musigma.py (A.R.H. Matthews, 2024).  Removed hardcoded paths  ║
# ║  and module-level script execution.                             ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ┌──────────────────────────────────────────────────────────────┐
# │ Compute  « mu and sigma from data »                          │
# └──────────────────────────────────────────────────────────────┘

def compute_mu_sigma(
    data: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute per-column mean and standard deviation.

    Args:
        data: (N, F) feature matrix.

    Returns:
        (mu, sigma) each of shape (F,).
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return mu, sigma


# ┌──────────────────────────────────────────────────────────────┐
# │ Persist  « save / load mu-sigma to CSV »                     │
# └──────────────────────────────────────────────────────────────┘

def save_mu_sigma(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    path: Path,
) -> None:
    """Write mu and sigma to a two-column CSV.

    Args:
        mu:    (F,) means.
        sigma: (F,) standard deviations.
        path:  Output CSV path.
    """
    path = Path(path)
    pd.DataFrame({"mu": mu, "sigma": sigma}).to_csv(path, index=False)


def load_mu_sigma(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load mu and sigma from a two-column CSV.

    Args:
        path: CSV with 'mu' and 'sigma' columns.

    Returns:
        (mu, sigma) as numpy arrays.
    """
    path = Path(path)
    df = pd.read_csv(path)
    return df["mu"].values, df["sigma"].values


# ┌──────────────────────────────────────────────────────────────┐
# │ Transform  « z-score and inverse »                           │
# └──────────────────────────────────────────────────────────────┘

def z_score(
    data: NDArray[np.floating],
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply z-score normalisation: z = (x - mu) / sigma.

    Args:
        data:  (N, F) feature matrix.
        mu:    (F,) means.
        sigma: (F,) standard deviations.

    Returns:
        (N, F) z-scored matrix.  Columns with sigma == 0 are set to 0.
    """
    safe_sigma = sigma.copy()
    safe_sigma[safe_sigma == 0] = 1.0
    return (data - mu) / safe_sigma


def de_z_score(
    data_z: NDArray[np.floating],
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Invert z-score normalisation: x = z * sigma + mu.

    Args:
        data_z: (N, F) z-scored matrix.
        mu:     (F,) means.
        sigma:  (F,) standard deviations.

    Returns:
        (N, F) matrix in original units.
    """
    return data_z * sigma + mu


# ┌──────────────────────────────────────────────────────────────┐
# │ Convenience  « compute + save in one call »                  │
# └──────────────────────────────────────────────────────────────┘

def compute_and_save_mu_sigma(
    data_path: Path,
    output_path: Path,
    *,
    exclude_columns: Optional[list[str]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load a .npy or .csv feature matrix, compute mu/sigma, save.

    Args:
        data_path:       Path to .npy array or .csv file.
        output_path:     Where to write the mu-sigma CSV.
        exclude_columns: Column names to drop before computing
                         (only for .csv input, e.g. 'time_series_id').

    Returns:
        (mu, sigma) arrays.
    """
    data_path = Path(data_path)

    if data_path.suffix == ".npy":
        data = np.load(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        if exclude_columns:
            df = df.drop(
                columns=[c for c in exclude_columns if c in df.columns]
            )
        data = df.values
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")

    mu, sigma = compute_mu_sigma(data)
    save_mu_sigma(mu, sigma, output_path)
    return mu, sigma
