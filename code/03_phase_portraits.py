"""Tutorial 3: sinks, periodic orbits, basins, and numerical stability."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common import setup, record, save_figure


def hopf(t, z, mu=0.8, omega=1.0):
    x, y = z
    r2 = x * x + y * y
    return np.array([(mu - r2) * x - omega * y,
                     omega * x + (mu - r2) * y])


def double_well(t, z):
    x, y = z
    return np.array([x - x ** 3, -2 * y])


def main():
    args, rng = setup(__doc__)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    grid = np.linspace(-1.7, 1.7, 33)
    xx, yy = np.meshgrid(grid, grid)
    fields = [("Stable spiral", lambda t, z: hopf(t, z, mu=-0.3)),
              ("Stable periodic orbit", hopf), ("Two attracting equilibria", double_well)]
    periodic_radius_errors = []
    for ax, (title, fun) in zip(axes, fields):
        dx, dy = fun(0, np.array([xx, yy]))
        speed = np.hypot(dx, dy)
        ax.streamplot(xx, yy, dx, dy, density=0.9, color=speed, cmap="Greys", linewidth=0.7)
        for initial in [[0.1, 0.1], [1.5, 0.4], [-1.3, -0.5], [0.2, -1.4]]:
            sol = solve_ivp(fun, (0, 30), initial, rtol=1e-9, atol=1e-11, max_step=0.05)
            assert sol.success
            ax.plot(*sol.y, lw=1.5)
            ax.plot(*sol.y[:, 0], "o", ms=3)
            if title == "Stable periodic orbit":
                periodic_radius_errors.append(float(abs(np.linalg.norm(sol.y[:, -1]) - np.sqrt(0.8))))
        ax.set(title=title, xlabel="x", ylabel="y", xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), aspect="equal")
    assert max(periodic_radius_errors) < 1e-5
    save_figure(fig, args, "phase_portraits")
    record(args, "03_phase_portraits", {"hopf_mu": 0.8,
           "expected_cycle_radius": float(np.sqrt(0.8)),
           "max_final_radius_error": max(periodic_radius_errors),
           "euler_multiplier_for_rate_minus5_dt0p5": -1.5,
           "exact_multiplier_same_step": float(np.exp(-2.5))})


if __name__ == "__main__":
    main()
