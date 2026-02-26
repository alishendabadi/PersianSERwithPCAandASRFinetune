"""
PCA dimensionality analysis for Whisper embeddings.

Usage (run from the project root):
    python utils/pca_analysis.py --mode variance [--threshold 0.95]
    python utils/pca_analysis.py --mode scree
"""

import sys
import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# Allow imports from the project root regardless of where the script is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from constants import FEATURES_SAVE_PATH, FLOATING_POINT, SHEMO_PATH  # noqa: E402
from configs import REPRESENTATION_CONFIG                              # noqa: E402
from utils.representation_extraction import extract_features           # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all_features(features_path: str, data_set: str, model_size: str, layer: int) -> np.ndarray:
    """
    Load every pre-extracted .pt feature file from *features_path*.
    If no files exist yet, run extraction first.

    Returns a 2-D float32 NumPy array of shape [N_frames, D] where
    N_frames = total_files * T (sequence length) and D = hidden dim.
    """
    def _pt_files(path):
        return [f for f in os.listdir(path) if f.endswith(".pt")] if os.path.isdir(path) else []

    pt_files = _pt_files(features_path)

    if not pt_files:
        print(f"No pre-extracted features found at '{features_path}'. Extracting now …")
        extract_features(
            DATA_SET=data_set,
            CHECKPOINTS=model_size,
            REPRESENTATION_LAYER=layer,
            FEATURES_SAVE_PATH=features_path,
        )
        pt_files = _pt_files(features_path)

    if not pt_files:
        raise RuntimeError(
            f"Still no .pt feature files found in '{features_path}' after attempting extraction.\n"
            f"  • Make sure SHEMO_PATH (currently '{SHEMO_PATH}') exists and contains .wav files.\n"
            f"  • Check that extract_features() completed without errors above."
        )

    print(f"Loading {len(pt_files)} feature files from '{features_path}' …")
    all_reps = []
    for fname in tqdm(pt_files, desc="Loading features"):
        rep = torch.load(os.path.join(features_path, fname), weights_only=True)  # [1, T, D]
        all_reps.append(rep.cpu().float())

    if not all_reps:
        raise RuntimeError(
            f"All .pt files in '{features_path}' failed to load. "
            "Check that the files are valid PyTorch tensors."
        )

    all_reps = torch.cat(all_reps, dim=0)              # [N, T, D]
    flat = all_reps.view(-1, all_reps.shape[-1])        # [N*T, D]
    return flat.numpy()


def _fit_full_pca(X: np.ndarray) -> PCA:
    """Fit PCA using all possible components."""
    print(f"Fitting full PCA on {X.shape[0]:,} vectors of dimension {X.shape[1]} …")
    pca = PCA()
    pca.fit(X)
    return pca


# ---------------------------------------------------------------------------
# Analysis 1 – Explained Variance Ratio
# ---------------------------------------------------------------------------

def explained_variance_analysis(X: np.ndarray, target_variance: float = 0.95) -> int:
    """
    Find how many PCA components are required to explain *target_variance*
    fraction of the total variance.  Prints results and saves a cumulative
    explained-variance plot to the artifacts directory.

    Returns the number of components needed.
    """
    pca = _fit_full_pca(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= target_variance)) + 1

    print(f"\n--- Explained Variance Analysis ---")
    print(f"  Input dimensionality : {X.shape[1]}")
    print(f"  Target variance      : {target_variance * 100:.1f}%")
    print(f"  Components needed    : {n_components}")
    print(f"  Compression ratio    : {X.shape[1] / n_components:.1f}×")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumulative) + 1), cumulative, linewidth=1.5, label="Cumulative variance")
    ax.axhline(y=target_variance, color="red", linestyle="--",
               label=f"{target_variance * 100:.0f}% threshold")
    ax.axvline(x=n_components, color="green", linestyle="--",
               label=f"n = {n_components} components")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"PCA Cumulative Explained Variance  "
                 f"({target_variance * 100:.0f}% → {n_components} components)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(ARTIFACTS_DIR,
                             f"pca_explained_variance_{int(target_variance * 100)}pct.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")

    return n_components


# ---------------------------------------------------------------------------
# Analysis 2 – Scree Plot / Elbow Method
# ---------------------------------------------------------------------------

def scree_plot(X: np.ndarray) -> None:
    """
    Produce a cumulative scree plot with 90 / 95 / 99 % threshold markers.
    Saved as PDF in the artifacts directory.
    """
    pca = _fit_full_pca(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n = len(cumulative)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n + 1), cumulative, linewidth=1.5)
    thresholds = [(0.90, "orange"), (0.95, "red"), (0.99, "purple")]
    for threshold, color in thresholds:
        idx = int(np.argmax(cumulative >= threshold)) + 1
        ax.axhline(y=threshold, color=color, linestyle="--", alpha=0.8,
                   label=f"{threshold * 100:.0f}%  →  n = {idx}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Scree Plot (Cumulative Explained Variance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(ARTIFACTS_DIR, "pca_scree_plot.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Scree plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA dimensionality analysis for Whisper embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/pca_analysis.py --mode variance
  python utils/pca_analysis.py --mode variance --threshold 0.99
  python utils/pca_analysis.py --mode scree
        """,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["variance", "scree"],
        help="'variance' – explained-variance analysis; 'scree' – scree/elbow plot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        metavar="FLOAT",
        help="Variance threshold for --mode variance (default: 0.95)",
    )
    args = parser.parse_args()

    X = _load_all_features(
        features_path=REPRESENTATION_CONFIG["path"],
        data_set=REPRESENTATION_CONFIG["data_set"],
        model_size=REPRESENTATION_CONFIG["model"],
        layer=REPRESENTATION_CONFIG["layer"],
    )

    if args.mode == "variance":
        explained_variance_analysis(X, target_variance=args.threshold)
    elif args.mode == "scree":
        scree_plot(X)


if __name__ == "__main__":
    main()
