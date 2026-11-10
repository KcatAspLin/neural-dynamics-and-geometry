"""Tutorial 5: approximate continuous attraction in a finite rate network."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from common import setup, record, save_figure


def main():
    args, rng = setup(__doc__)
    n, tau, dt = 96, 0.1, 0.002
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    w = (-4.0 + 14.0 * np.cos(theta[:, None] - theta[None, :])) / n
    steps = 5000
    def simulate(initial, perturb=None):
        r = initial.copy()
        history = np.empty((steps, n))
        for i in range(steps):
            if i == 1500 and perturb is not None:
                r = perturb(r)
            r += dt * (-r + expit(w @ r)) / tau
            history[i] = r
        return history
    initial = expit(3 * np.cos(theta))
    # First equilibrate, then use exactly the same baseline state in each probe.
    baseline = simulate(initial)[-1]
    radial = simulate(baseline, lambda r: 0.6 * r + 0.4 * np.mean(r))
    phase = simulate(baseline, lambda r: np.roll(r, 12))
    time = dt * np.arange(steps)
    moment = lambda r: r @ np.exp(1j * theta) / n
    amp0 = abs(moment(baseline))
    phase0 = np.angle(moment(baseline))
    sigmoid = expit(w @ baseline)
    jac = (-np.eye(n) + np.diag(sigmoid * (1 - sigmoid)) @ w) / tau
    eig = np.sort(np.linalg.eigvals(jac).real)[::-1]
    residual = float(np.linalg.norm(-baseline + sigmoid))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    im = axes[0, 0].imshow(radial[::10].T, aspect="auto", origin="lower",
                           extent=[time[0], time[-1], -np.pi, np.pi], cmap="viridis")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Preferred angle (rad)", title="Radial perturbation at 3 s")
    fig.colorbar(im, ax=axes[0, 0], label="Rate (normalized)")
    axes[0, 1].plot(theta, baseline, label="baseline")
    axes[0, 1].plot(theta, radial[-1], "--", label="after radial perturbation")
    axes[0, 1].plot(theta, phase[-1], ":", lw=2, label="after phase perturbation")
    axes[0, 1].set(xlabel="Preferred angle (rad)", ylabel="Rate", title="Bump recovers; phase can change")
    axes[0, 1].legend(fontsize=8)
    for label, history in [("radial", radial), ("phase", phase)]:
        z = moment(history)
        axes[1, 0].plot(time, abs(z), label=label)
        axes[1, 1].plot(time, np.angle(np.exp(1j * (np.angle(z) - phase0))), label=label)
    axes[1, 0].set(xlabel="Time (s)", ylabel="First-harmonic magnitude", title="Transverse relaxation")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Phase shift (rad)", title="Tangential memory")
    axes[1, 0].legend(); axes[1, 1].legend()
    assert residual < 1e-6
    assert np.all((radial >= 0) & (radial <= 1))
    assert abs(abs(moment(radial[-1])) - amp0) < 1e-4
    save_figure(fig, args, "ring_attractor")
    record(args, "05_ring_attractor", {"n": n, "tau_seconds": tau, "dt_seconds": dt,
           "baseline_fixed_point_residual": residual,
           "largest_jacobian_eigenvalues": eig[:5].tolist(),
           "baseline_bump_magnitude": float(amp0),
           "final_radial_magnitude_error": float(abs(abs(moment(radial[-1])) - amp0)),
           "final_phase_shift_radians": float(np.angle(moment(phase[-1]) / moment(baseline))),
           "caveat": "Finite neuron grid has discrete rotation symmetry; near-neutral is not a proof of exact continuous symmetry."})


if __name__ == "__main__":
    main()
