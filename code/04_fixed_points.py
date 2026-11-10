"""Tutorial 4: roots, residual validation, Jacobians, and step size."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from common import setup, record, save_figure

W = np.diag([1.5, 0.7])
TAU = 1.0


def field(x):
    return (-x + W @ np.tanh(x)) / TAU


def jacobian(x):
    return (-np.eye(2) + W @ np.diag(1 - np.tanh(x) ** 2)) / TAU


def main():
    args, rng = setup(__doc__)
    fixed = []
    for initial in rng.uniform(-3, 3, (70, 2)):
        result = root(field, initial, jac=jacobian)
        # Solver success alone is insufficient. Always validate the residual.
        if np.linalg.norm(field(result.x)) < 1e-9:
            if not any(np.linalg.norm(result.x - old) < 1e-5 for old in fixed):
                fixed.append(result.x)
    fixed = sorted(fixed, key=lambda x: x[0])
    assert len(fixed) == 3
    rows = []
    for x in fixed:
        lam = np.linalg.eigvals(jacobian(x)).real
        step = 1e-6
        numeric = np.column_stack([(field(x + step * e) - field(x - step * e)) / (2 * step)
                                   for e in np.eye(2)])
        assert np.max(np.abs(numeric - jacobian(x))) < 1e-7
        rows.append({"point": x.tolist(), "residual": float(np.linalg.norm(field(x))),
                     "jacobian_eigenvalues": lam.tolist(),
                     "continuous_stable": bool(np.all(lam < 0)),
                     "euler_dt4_spectral_radius": float(max(abs(1 + 4 * lam)))})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    xx = np.linspace(-2, 2, 400)
    ax[0].plot(xx, -xx + 1.5 * np.tanh(xx), label=r"$f_1(x)$ at $y=0$")
    ax[0].axhline(0, color="gray", lw=0.6)
    for row in rows:
        x = row["point"][0]
        ax[0].plot(x, 0, "o", mfc="C0" if row["continuous_stable"] else "white", mec="C0")
    ax[0].set(xlabel="x", ylabel="dx/dt", title="Three roots; only two attract")
    ax[0].legend()
    dt = np.linspace(0, 4, 250)
    outer = np.linalg.eigvals(jacobian(fixed[-1])).real
    for index, lam in enumerate(outer):
        ax[1].plot(dt, abs(1 + dt * lam), label=f"Euler mode {index + 1}")
    ax[1].axhline(1, color="gray", ls=":")
    ax[1].set(xlabel="Euler step", ylabel="Multiplier magnitude", title="Discretization can destabilize a sink")
    ax[1].legend()
    save_figure(fig, args, "fixed_points")
    record(args, "04_fixed_points", {"tau": TAU, "fixed_points": rows,
           "search_initializations": 70, "global_completeness": "not certified by multistart search"})


if __name__ == "__main__":
    main()
