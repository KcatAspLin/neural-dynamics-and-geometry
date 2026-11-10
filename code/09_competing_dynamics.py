"""Tutorial 9: identical unperturbed rings, different perturbation responses."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common import setup, record, save_figure


def main():
    args, rng = setup(__doc__)
    omega, kappa, delta = 1.0, 1.0, 0.3
    times = np.linspace(0, 4 * np.pi, 801)
    reference = np.column_stack([np.cos(omega * times), np.sin(omega * times)])
    def recurrent(t, x):
        return kappa * (1 - x @ x) * x + omega * np.array([-x[1], x[0]])
    def driven(t, x):
        # External clock supplies velocity; current state does not affect it.
        return omega * np.array([-np.sin(omega * t), np.cos(omega * t)])
    def integrate(fun, initial):
        sol = solve_ivp(fun, (times[0], times[-1]), initial, t_eval=times,
                        rtol=1e-10, atol=1e-12, max_step=0.025)
        assert sol.success
        return sol.y.T
    rec0, ext0 = integrate(recurrent, [1, 0]), integrate(driven, [1, 0])
    rec1, ext1 = integrate(recurrent, [1 + delta, 0]), integrate(driven, [1 + delta, 0])
    observational_difference = float(np.max(np.linalg.norm(rec0 - ext0, axis=1)))
    # Repeat noisy perturbation measurements at one full cycle. Matched reference
    # phase avoids confusing tangential phase changes with transverse contraction.
    n = 200
    final_time = 2 * np.pi
    radial_rec = integrate(recurrent, [1 + delta, 0])[np.argmin(abs(times - final_time))]
    angle_at_index = omega * times[np.argmin(abs(times - final_time))]
    radial_ext = np.array([np.cos(angle_at_index) + delta, np.sin(angle_at_index)])
    observed_rec = np.linalg.norm(radial_rec + rng.normal(0, 0.03, (n, 2)), axis=1) - 1
    observed_ext = np.linalg.norm(radial_ext + rng.normal(0, 0.03, (n, 2)), axis=1) - 1
    statistic = float(observed_ext.mean() - observed_rec.mean())
    pooled = np.r_[observed_ext, observed_rec]
    permuted = []
    for _ in range(999):
        shuffled = rng.permutation(pooled)
        permuted.append(float(shuffled[:n].mean() - shuffled[n:].mean()))
    pvalue = float((1 + np.count_nonzero(np.asarray(permuted) >= statistic)) / 1000)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    ax[0].plot(*rec0.T, label="autonomous recurrent")
    ax[0].plot(*ext0.T, "--", label="external clock")
    ax[0].set(xlabel="x", ylabel="y", title="Unperturbed trajectories coincide", aspect="equal")
    ax[0].legend(fontsize=8)
    ax[1].plot(times, np.linalg.norm(rec1, axis=1) - 1, label="recurrent")
    ax[1].plot(times, np.linalg.norm(ext1, axis=1) - 1, label="external clock")
    ax[1].set(xlabel="Time", ylabel="Signed radial displacement", title="Same radial intervention")
    ax[1].legend(fontsize=8)
    ax[2].hist(observed_rec, bins=20, alpha=0.65, label="recurrent")
    ax[2].hist(observed_ext, bins=20, alpha=0.65, label="external clock")
    ax[2].set(xlabel="Radial displacement after one cycle", ylabel="Trials", title=f"Prespecified group contrast: p={pvalue:.3f}")
    ax[2].legend(fontsize=8)
    save_figure(fig, args, "null_hypotheses")
    assert observational_difference < 1e-6
    record(args, "09_competing_dynamics", {"max_unperturbed_difference": observational_difference,
           "radial_perturbation": delta, "measurement_noise_std": 0.03,
           "trials_per_model": n, "permutations": 999,
           "radial_mean_contrast": statistic, "randomization_p": pvalue,
           "caveat": "This intervention distinguishes these two specified models, not all possible mechanisms."})


if __name__ == "__main__":
    main()
