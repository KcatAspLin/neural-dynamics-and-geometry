"""Tutorial 6: known manifolds, a disk control, and scale-dependent homology."""
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from common import setup, record, save_figure, circle, sphere, longest_bar


def main():
    args, rng = setup(__doc__)
    ring, _ = circle(rng, 160, 0.02)
    sph = sphere(rng, 150, 0.015)
    # A product of two circles in R^4: an embedded flat torus.
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 14, endpoint=False),
                       np.linspace(0, 2 * np.pi, 14, endpoint=False))
    tor = np.column_stack([np.cos(a.ravel()), np.sin(a.ravel()),
                           np.cos(b.ravel()), np.sin(b.ravel())])
    tor += rng.normal(0, 0.015, tor.shape)
    radius = np.sqrt(rng.uniform(0, 1, 160))
    angle = rng.uniform(0, 2 * np.pi, 160)
    disk = radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    datasets = {"Circle": ring, "Sphere": sph, "Product torus": tor, "Filled disk": disk}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    metrics = {}
    for column, (name, x) in enumerate(datasets.items()):
        dgm = ripser(x, maxdim=2, coeff=2, thresh=2.3)["dgms"]
        np.savez(args.out / (name.lower().replace(" ", "_") + "_diagrams.npz"),
                 points=x, h0=dgm[0], h1=dgm[1], h2=dgm[2])
        # Display only a projection. Computation above uses the full input space.
        axes[0, column].scatter(x[:, 0], x[:, 1], c=x[:, -1], s=8, cmap="viridis")
        axes[0, column].set(title=name, xlabel="Coordinate 1", ylabel="Coordinate 2", aspect="equal")
        for dim, diagram in enumerate(dgm):
            finite = diagram[np.isfinite(diagram[:, 1])]
            axes[1, column].scatter(finite[:, 0], finite[:, 1], s=14, alpha=0.75,
                                    label=f"H{dim}", marker=[".", "o", "^"][dim])
        axes[1, column].plot([0, 2.3], [0, 2.3], ":", color="gray", lw=0.8)
        axes[1, column].set(xlabel="Birth (edge length)", ylabel="Death", xlim=(-0.04, 2.35), ylim=(-0.04, 2.35))
        axes[1, column].legend(fontsize=7)
        metrics[name] = {"n": len(x), "ambient": x.shape[1],
                         "longest_finite_H1": longest_bar(dgm[1]),
                         "longest_finite_H2": longest_bar(dgm[2]),
                         "essential_or_right_censored_counts": [int(np.isinf(d[:, 1]).sum()) for d in dgm]}
    save_figure(fig, args, "topology_clouds")
    record(args, "06_manifold_clouds", {"rips_max_edge_length": 2.3, "coefficient_field": 2,
           "datasets": metrics,
           "interpretation": "Ground truth is known by construction; projections and finite bars alone do not prove manifold type."})


if __name__ == "__main__":
    main()
