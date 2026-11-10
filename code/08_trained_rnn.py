"""Tutorial 8: train delayed binary memory, then analyze the hidden map."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from sklearn.decomposition import PCA
import torch
from torch import nn
from common import setup, record, save_figure
from recurrent_tools import state_jacobian


def main():
    args, rng = setup(__doc__)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hidden, length, batch, updates = 16, 35, 64, 650
    cell = nn.RNNCell(1, hidden, nonlinearity="tanh").double()
    readout = nn.Linear(hidden, 1).double()
    nn.init.orthogonal_(cell.weight_hh, gain=1.1)
    optimizer = torch.optim.Adam(list(cell.parameters()) + list(readout.parameters()), lr=0.006)
    losses = []
    def rollout(cues, steps=length):
        cues = cues.to(dtype=cell.weight_hh.dtype)
        h = torch.zeros(len(cues), hidden, dtype=torch.float64)
        states, outputs = [], []
        for t in range(steps):
            u = cues[:, None] if t < 3 else torch.zeros(len(cues), 1, dtype=torch.float64)
            h = cell(u, h)
            states.append(h)
            outputs.append(readout(h).squeeze(-1))
        return torch.stack(states, dim=1), torch.stack(outputs, dim=1)
    for _ in range(updates):
        signs = 2 * torch.randint(0, 2, (batch,), dtype=torch.float64) - 1
        strength = 0.8 + 0.4 * torch.rand(batch, dtype=torch.float64)
        _, prediction = rollout(signs * strength)
        loss = ((prediction[:, 3:] - signs[:, None]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(cell.parameters()) + list(readout.parameters()), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    test_sign = np.tile([-1.0, 1.0], 64)
    test_strength = rng.uniform(0.7, 1.3, len(test_sign))
    with torch.no_grad():
        ht, yp = rollout(torch.tensor(test_sign * test_strength), steps=150)
        fit_h, _ = rollout(torch.tensor(np.linspace(-1.2, -0.8, 40).tolist()
                                       + np.linspace(0.8, 1.2, 40).tolist()))
    h_array, y_array = ht.numpy(), yp.numpy()
    pca = PCA(n_components=2).fit(fit_h.numpy().reshape(-1, hidden))
    z = pca.transform(h_array.reshape(-1, hidden)).reshape(len(test_sign), 150, 2)
    wh = cell.weight_hh.detach().numpy()
    bias = (cell.bias_ih + cell.bias_hh).detach().numpy()
    def mapping(h):
        return np.tanh(wh @ h + bias)
    def jacobian(h):
        return np.diag(1 - mapping(h) ** 2) @ wh
    fixed = []
    starts = np.concatenate([h_array[:12, 34], rng.normal(0, 0.5, (30, hidden)),
                              np.zeros((1, hidden))])
    for initial in starts:
        sol = root(lambda h: mapping(h) - h, initial,
                   jac=lambda h: jacobian(h) - np.eye(hidden))
        if np.linalg.norm(mapping(sol.x) - sol.x) < 1e-8:
            if not any(np.linalg.norm(sol.x - old) < 1e-5 for old in fixed):
                fixed.append(sol.x)
    fixed_rows = []
    spectra = []
    for h in fixed:
        eig = np.linalg.eigvals(jacobian(h))
        spectra.append(eig)
        with torch.no_grad():
            value = float(readout(torch.tensor(h)).item())
        fixed_rows.append({"residual": float(np.linalg.norm(mapping(h) - h)),
                           "spectral_radius": float(np.max(abs(eig))),
                           "readout": value})
    # Independent derivative verification; includes LSTM's cell state.
    automatic = state_jacobian(cell, ht[0, -1].detach(), torch.zeros(1, dtype=torch.float64)).numpy()
    assert np.max(np.abs(automatic - jacobian(h_array[0, -1]))) < 1e-10
    lstm = nn.LSTMCell(1, 4).double()
    full_jac = state_jacobian(lstm, torch.zeros(8, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    assert full_jac.shape == (8, 8)
    gru = nn.GRUCell(1, 4).double()
    assert state_jacobian(gru, torch.zeros(4, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)).shape == (4, 4)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    ax[0, 0].semilogy(losses)
    ax[0, 0].set(xlabel="Optimizer update", ylabel="Training MSE", title="Learning dynamics (parameters change)")
    for index in [0, 1, 2, 3]:
        ax[0, 1].plot(*z[index, :35].T, color="C0" if test_sign[index] < 0 else "C1", alpha=0.8)
    ax[0, 1].set(xlabel="PC 1", ylabel="PC 2", title="Inference dynamics (parameters frozen)")
    for sign in [-1, 1]:
        group = y_array[test_sign == sign]
        ax[1, 0].plot(group.mean(0), label=f"cue {sign:+d}")
        ax[1, 0].fill_between(np.arange(150), group.min(0), group.max(0), alpha=0.15)
    ax[1, 0].axvline(35, ls=":", color="gray", label="training horizon")
    ax[1, 0].set(xlabel="Recurrent step", ylabel="Readout", title="Prolonged zero-input probe")
    ax[1, 0].legend(fontsize=8)
    angle = np.linspace(0, 2 * np.pi, 300)
    ax[1, 1].plot(np.cos(angle), np.sin(angle), ":", color="gray")
    for index, eig in enumerate(spectra):
        ax[1, 1].scatter(eig.real, eig.imag, s=14, label=f"root {index + 1}")
    ax[1, 1].set(xlabel="Re eigenvalue", ylabel="Im eigenvalue", title="Discrete-map stability boundary", aspect="equal")
    if spectra:
        ax[1, 1].legend(fontsize=7)
    save_figure(fig, args, "rnn_dynamics")
    torch.save({"cell": cell.state_dict(), "readout": readout.state_dict(),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": rng.bit_generator.state, "seed": args.seed,
                "updates": updates, "hidden_size": hidden, "training_length": length},
               args.out / "delayed_memory_checkpoint.pt")
    np.savez(args.out / "rnn_probe_states.npz", states=h_array, readout=y_array,
             cue_sign=test_sign, cue_strength=test_strength, fixed_points=np.asarray(fixed),
             pca_components=pca.components_, pca_mean=pca.mean_, losses=np.asarray(losses))
    accuracy = float(np.mean(np.sign(y_array[:, 34]) == test_sign))
    assert accuracy > 0.95, "This seed did not learn the task; inspect optimization."
    record(args, "08_trained_rnn", {"hidden_units": hidden, "updates": updates,
           "training_steps_per_trial": length, "last_training_mse": losses[-1],
           "test_accuracy_at_step35": accuracy,
           "test_mse_at_step35": float(np.mean((y_array[:, 34] - test_sign) ** 2)),
           "test_mse_at_step150": float(np.mean((y_array[:, -1] - test_sign) ** 2)),
           "fixed_points": fixed_rows,
           "root_search_completeness": "not certified", "lstm_full_state_jacobian_shape": list(full_jac.shape)})


if __name__ == "__main__":
    main()
