# Computational and implementation source audit

Verified 10 November 2026. General mathematical derivations and all synthetic examples are original exposition. Software API descriptions are tied to official documentation, and recorded numerical claims are tied to executed scripts in this project.

| Citation key | Primary source | Verification and use |
|---|---|---|
| root_ripser | https://joss.theoj.org/papers/10.21105/joss.00925 | Christopher Tralie, Nathaniel Saul, Rann Bar-On; exact title *Ripser.py: A Lean Persistent Homology Library for Python*; JOSS 3(29), 925 (2018); DOI verified from publisher BibTeX. |
| root_umap_docs | https://umap-learn.readthedocs.io/en/latest/api.html and https://umap-learn.readthedocs.io/en/latest/reproducibility.html | Version 0.6.0 API and compatibility layout option checked. Script detects this option before setting it and records software version and seed. No cross-version exact-repeatability guarantee. |
| root_torch_rnn | https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNNCell.html | Official RNN update, summed input/recurrent biases, input/state shape, and tanh derivative convention checked. |
| app_torchlstm | https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTMCell.html | Official h,c state and modern gating equations checked. Duplicate source key root_torch_lstm merged into app_torchlstm. |
| API supplement | https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html | maxdim, coeff, thresh, distance_matrix, and cocycle-return options checked. H0 through maxdim are returned; finite threshold requires censoring care. |

## Numerical provenance

Nine numbered Python scripts were executed end to end with seed 17. All result assertions passed. The generated JSON files record package versions and scalar results; NPZ files retain the main data arrays. The delayed-memory model was trained in this execution, and its checkpoint includes model, readout, optimizer, and random-number states. The other simulations use declared analytical or generative models. No experiment on biological recordings was run.

The final full runner completed successfully after corrections. Additional checks compared analytical ODE and RNN Jacobians with finite differences or automatic differentiation, checked a known oscillator radius, and inspected the distinction between intrinsic-distance quality and local trustworthiness. Changing the Isomap neighborhood exposes a documented failure mode. These checks are limited to the concrete included examples.
