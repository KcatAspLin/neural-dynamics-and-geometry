"""Tutorial 1: linear, intrinsic, and effective dimensions need not agree."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from common import setup, record, save_figure


def local_dimension(x, k):
    """Mean bias-corrected local Poisson k-neighbor estimate; no duplicates."""
    d = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[0][:, 1:]
    if np.any(d <= 0):
        raise ValueError("Remove duplicates before estimating local dimension.")
    return float(np.mean((k - 2) / np.log(d[:, -1, None] / d[:, :-1]).sum(axis=1)))


def main():
    args, rng = setup(__doc__)
    n, p, noise = 600, 24, 0.02
    theta = rng.uniform(0, 2 * np.pi, n)
    q, _ = np.linalg.qr(rng.normal(size=(p, 2)))
    clean = np.column_stack([np.cos(theta), np.sin(theta)]) @ q.T
    noisy = clean + noise * rng.normal(size=clean.shape)
    pc0, pc1 = PCA().fit(clean), PCA().fit(noisy)
    pr = lambda ev: float(ev.sum() ** 2 / np.square(ev).sum())
    ks = [8, 16, 32, 64]
    dims = {name: [local_dimension(x, k) for k in ks]
            for name, x in [("clean", clean), ("noisy", noisy)]}
    projection = pc1.transform(noisy)
    # A squared reconstruction loss measured on new rows still decreases with k;
    # it is not by itself a model-selection criterion for a noise-free rank.
    train, test = noisy[:400], noisy[400:]
    pca_train = PCA().fit(train)
    centered = test - pca_train.mean_
    errors = []
    for k in range(1, p + 1):
        v = pca_train.components_[:k]
        errors.append(float(np.mean((centered - centered @ v.T @ v) ** 2)))
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    ax[0, 0].scatter(*projection[:, :2].T, c=theta, s=6, cmap="twilight")
    ax[0, 0].set(xlabel="PC 1", ylabel="PC 2", title="One latent angle; two linear directions")
    ax[0, 0].set_aspect("equal")
    for name, pc in [("clean", pc0), ("noisy", pc1)]:
        ax[0, 1].semilogy(np.arange(1, p + 1), np.maximum(pc.explained_variance_, 1e-9),
                          ".-", label=name)
        ax[1, 0].plot(ks, dims[name], "o-", label=name)
    ax[0, 1].set(xlabel="Component", ylabel="Covariance eigenvalue", title="Noise fills the ambient space")
    ax[0, 1].legend()
    ax[1, 0].axhline(1, color="gray", ls=":", label="clean intrinsic dimension")
    ax[1, 0].set(xlabel="Neighbor count k", ylabel="Estimated local dimension", title="Scale changes the estimate")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(np.arange(1, p + 1), errors, "o-")
    ax[1, 1].set(xlabel="Retained PCs", ylabel="Test reconstruction MSE", title="New-row reconstruction is not rank selection")
    assert np.linalg.matrix_rank(clean - clean.mean(0)) == 2
    assert errors[-1] < 1e-20
    save_figure(fig, args, "pca_geometry")
    record(args, "01_pca_dimension", {"n": n, "ambient_dimension": p,
           "known_clean_intrinsic_dimension": 1, "clean_rank": 2,
           "clean_participation_ratio": pr(pc0.explained_variance_),
           "noisy_participation_ratio": pr(pc1.explained_variance_),
           "noise_std": noise, "neighbor_counts": ks, "local_estimates": dims,
           "test_reconstruction_mse_k2": errors[1]})


if __name__ == "__main__":
    main()
