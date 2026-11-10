"""Tutorial 7: train/test preprocessing and a prespecified topological null."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ripser import ripser
from common import setup, record, save_figure, longest_bar


def main():
    args, rng = setup(__doc__)
    p, n_train, n_test, permutations = 24, 500, 180, 99
    preferred = np.linspace(0, 2 * np.pi, p, endpoint=False)
    def sample(n):
        angle = rng.uniform(0, 2 * np.pi, n)
        rate = 2 + 4 * np.exp(2 * (np.cos(angle[:, None] - preferred) - 1))
        return rate + rng.normal(0, 0.15, rate.shape), angle
    train, _ = sample(n_train)
    test, angle = sample(n_test)
    pca = PCA(n_components=8).fit(train)
    z = pca.transform(test)
    diagrams = ripser(z, maxdim=1, coeff=2)["dgms"]
    raw_diagrams = ripser(test, maxdim=1, coeff=2)["dgms"]
    observed = longest_bar(diagrams[1])
    surrogate = []
    for _ in range(permutations):
        shuffled = np.column_stack([rng.permutation(test[:, j]) for j in range(p)])
        # The training-set transformation remains frozen for every test surrogate.
        null_z = pca.transform(shuffled)
        surrogate.append(longest_bar(ripser(null_z, maxdim=1, coeff=2)["dgms"][1]))
    surrogate = np.asarray(surrogate)
    pvalue = float((1 + np.count_nonzero(surrogate >= observed)) / (permutations + 1))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    ax[0].scatter(*z[:, :2].T, c=angle, cmap="twilight", s=12)
    ax[0].set(xlabel="PC 1", ylabel="PC 2", title="Held-out rate samples", aspect="equal")
    limit = max(np.max(diagrams[1][:, 1], initial=0.0), 1) * 1.08
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])]
        ax[1].scatter(*finite.T, s=14, label=f"H{dim}")
    ax[1].plot([0, limit], [0, limit], ":", c="gray")
    ax[1].set(xlabel="Birth (edge length)", ylabel="Death", title="Rips persistence in 8 PC coordinates")
    ax[1].legend()
    ax[2].hist(surrogate, bins=15, alpha=0.7, label="column-permutation null")
    ax[2].axvline(observed, color="C1", label="observed maximum H1 lifetime")
    ax[2].set(xlabel="Maximum finite H1 lifetime", ylabel="Count", title=f"Plus-one randomization p = {pvalue:.3f}")
    ax[2].legend(fontsize=7)
    save_figure(fig, args, "persistence")
    np.savez(args.out / "neural_ring_analysis.npz", train=train, test=test, angles=angle,
             test_scores=z, components=pca.components_, mean=pca.mean_,
             h0=diagrams[0], h1=diagrams[1], null_lifetimes=surrogate)
    record(args, "07_neural_persistence", {"n_train": n_train, "n_test": n_test,
           "neurons": p, "pca_components": 8, "permutations": permutations,
           "observed_H1_lifetime": observed,
           "raw_space_H1_lifetime": longest_bar(raw_diagrams[1]),
           "null_max_lifetime": float(surrogate.max()), "randomization_p": pvalue,
           "null_hypothesis": "Independent neuron columns for IID test observations, conditional on the frozen training transform.",
           "caveat": "Rejecting this null does not establish autonomous recurrent generation or an attractor."})


if __name__ == "__main__":
    main()
