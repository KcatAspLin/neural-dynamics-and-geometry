"""Tutorial 2: compare embeddings against the known Swiss-roll metric."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, trustworthiness
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
import umap
from common import setup, record, save_figure


def main():
    args, rng = setup(__doc__)
    n = 650
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    height = rng.uniform(0, 20, n)
    x = np.column_stack([t * np.cos(t), height, t * np.sin(t)])
    # Arc length along the spiral makes a flat rectangle of the induced metric.
    arc = 0.5 * (t * np.sqrt(1 + t * t) + np.arcsinh(t))
    truth = np.column_stack([arc, height])
    x += 0.02 * rng.normal(size=x.shape)
    graph = kneighbors_graph(x, 6, mode="distance")
    components = connected_components(graph.maximum(graph.T), directed=False)[0]
    assert components == 1, "Disconnected neighborhood graph; change the sampling or k."
    options = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                   random_state=args.seed, n_jobs=1)
    if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
        options["compatibility_layout"] = True
    embeddings = {"PCA": PCA(n_components=2).fit_transform(x),
                  "Isomap": Isomap(n_neighbors=6).fit_transform(x),
                  "UMAP": umap.UMAP(**options).fit_transform(x)}
    i, j = rng.integers(0, n, size=(2, 5000))
    keep = i != j
    i, j = i[keep], j[keep]
    true_d = np.linalg.norm(truth[i] - truth[j], axis=1)
    metrics = {}
    fig = plt.figure(figsize=(12, 3.3), layout="constrained")
    ax = fig.add_subplot(141, projection="3d")
    ax.scatter(*x.T, c=t, s=4, cmap="viridis")
    ax.set(title="Observed Swiss roll", xlabel="x", ylabel="y", zlabel="z")
    for panel, (name, z) in enumerate(embeddings.items(), start=2):
        ax = fig.add_subplot(1, 4, panel)
        ax.scatter(*z.T, c=t, s=5, cmap="viridis")
        ax.set(title=name, xlabel="Coordinate 1", ylabel="Coordinate 2")
        distances = np.linalg.norm(z[i] - z[j], axis=1)
        scale = float(distances @ true_d / (distances @ distances))
        metrics[name] = {"trustworthiness_k10": float(trustworthiness(x, z, n_neighbors=10)),
                         "intrinsic_distance_spearman": float(spearmanr(distances, true_d).statistic),
                         "scaled_intrinsic_stress": float(np.linalg.norm(scale * distances - true_d)
                                                           / np.linalg.norm(true_d))}
    # A larger neighborhood can bridge different folds. Record the sensitivity
    # rather than reporting only the favorable known-ground-truth setting.
    z12 = Isomap(n_neighbors=12).fit_transform(x)
    d12 = np.linalg.norm(z12[i] - z12[j], axis=1)
    scale12 = float(d12 @ true_d / (d12 @ d12))
    metrics["Isomap_k12_sensitivity"] = {
        "intrinsic_distance_spearman": float(spearmanr(d12, true_d).statistic),
        "scaled_intrinsic_stress": float(np.linalg.norm(scale12 * d12 - true_d) / np.linalg.norm(true_d))}
    save_figure(fig, args, "embeddings")
    record(args, "02_embedding_comparison", {"n": n, "isomap_k": 6,
           "graph_components": int(components), "umap": options, "embedding_metrics": metrics})


if __name__ == "__main__":
    main()
