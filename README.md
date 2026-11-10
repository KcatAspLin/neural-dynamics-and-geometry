# Mathematical Foundations of Neural Dynamics and Geometry

**From Dimensionality Reduction to Dynamical Systems, Manifolds, and Topology**  
Foundations edition, 10 November 2026.

A textbook-style course for readers with first-year undergraduate linear algebra, calculus, probability, and basic Python. It connects exact mathematical definitions, reproducible numerical methods, and the strength of scientific inference in neuroscience and artificial neural networks.

## Read or compile the book

The delivered PDF can be read without installing anything. This source project contains all figures already generated, so rebuilding the book does not require rerunning Python.

With a reasonably complete TeX Live or MiKTeX installation (including standard `latex-extra`, fonts, and science packages), run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Without `latexmk`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. The project uses the `book` class, pdfLaTeX, BibTeX, `natbib`, `plainnat`, AMS math/theorem packages, `graphicx`, `booktabs`, `longtable`, `tabularx`, `tikz`, `tcolorbox`, `listings`, `microtype`, `hyperref`, and Latin Modern fonts. Shell escape is unnecessary. On Overleaf, upload the source ZIP, select `main.tex`, and choose pdfLaTeX.

## Project structure

| Location | Contents |
|---|---|
| `main.tex` | Modular book assembly, parts, appendices, bibliography |
| `preamble.tex` | Typography, notation, theorem and explanatory-box definitions |
| `frontmatter.tex` | Title page, scope, curriculum dependencies, notation, reading routes |
| `chapters/chapter01.tex`–`chapter13.tex` | Thirteen fully written introductory chapters |
| `appendices/appendix_a.tex`–`appendix_e.tex` | Four prerequisite references and alphabetical glossary |
| `references.bib` | Deduplicated, externally verified bibliography |
| `figures/` | Vector PDF figures plus PNG previews |
| `code/` | Nine standalone tutorial entry points, shared helpers, runner, dependencies |
| `results/` | Recorded metrics, generated arrays, environment, and trained checkpoint |
| `research/` | Bibliographic source audits and citation-alias mapping |
| `VALIDATION.md` | Compilation and numerical verification summary |

## Curriculum

| Chapter | Topic | Essential prerequisites |
|---|---|---|
| 1 | Data, vector spaces, PCA, and nonlinear dimensionality reduction | Basic linear algebra |
| 2 | ODEs, maps, stability, attractors, bifurcations, and noise | Basic calculus |
| 3 | Topology prerequisites, smooth manifolds, embeddings, and metrics | Chapters 1–2 as needed |
| 4 | Simplicial complexes, homology, persistence, and circular coordinates | Chapter 3; elementary algebra introduced locally |
| 5 | Invariant/slow manifolds, observation maps, and latent dynamics | Chapters 2–4 |
| 6 | Neural population latent-variable analysis | Chapters 1–2; probability |
| 7 | Neural recurrent dynamics and attractor hypotheses | Chapters 2 and 5 |
| 8 | Neural representation geometry and alignment | Chapters 1, 3, 5 |
| 9 | Neural TDA pipelines and statistical controls | Chapters 4, 6, 8 |
| 10 | RNN, GRU, LSTM, representation, and learning dynamics | Chapters 2, 5–8 |
| 11 | Nine computational tutorials | Relevant topic chapters |
| 12 | What mathematical claims neural data support | Foundations plus applications |
| 13 | Critical literature case studies | Chapter 12 framework |

The preface marks the depth of the advanced extensions. The book is an introductory course, not an exhaustive monograph on stochastic analysis, singular perturbation, integral algebraic topology, or Koopman spectral theory. General proofs of those extensions are deferred to the references. Persistent-cohomology circular coordinates are explained, but no circular-coordinate decoder is implemented in the nine tutorials. All tutorial data are synthetic; no biological dataset has been reanalyzed.

## Run the computational tutorials

Create a clean environment, install the dependencies, then run all examples:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/run_all.py --seed 17
```

For a close reproduction of the recorded environment, use Python 3.12 and `code/requirements-tested.txt`. The recorded PyTorch build is the Linux CPU wheel; a CPU index is provided in that file. On macOS or another platform, install the platform-appropriate PyTorch wheel of the same release instead of the `+cpu` build. A GPU is not required. Only the exact recorded environment was executed for this edition; the broader ranges are API compatibility guidance, not a claim that all combinations were tested.

By default, scripts save metrics and arrays to `results/`, and figures to `figures/`. For an isolated run that preserves the delivered results:

```bash
python code/run_all.py --seed 18 --out new_results --figures new_figures
```

Run one tutorial in the same way:

```bash
python code/07_neural_persistence.py --seed 17 --out my_results --figures my_figures
```

`run_all.py` sets common numerical-library thread counts to one, runs scripts sequentially, and stops if any script fails. The first UMAP call may spend extra time compiling numerical kernels. Each script independently resets its random seed. Results can change across software versions even with the same seed.

### Tutorial map

| Script | Research question | Main output |
|---|---|---|
| `01_pca_dimension.py` | Why can a circle have intrinsic dimension 1, linear rank 2, and larger noisy local estimates? | `pca_geometry.pdf` |
| `02_embedding_comparison.py` | Which properties do PCA, Isomap, and UMAP preserve on a known Swiss roll? | `embeddings.pdf` and neighborhood sensitivity metrics |
| `03_phase_portraits.py` | How do sinks, cycles, and competing basins differ? | `phase_portraits.pdf` |
| `04_fixed_points.py` | How should root residuals and Jacobian stability be validated? | `fixed_points.pdf` |
| `05_ring_attractor.py` | What differs between radial relaxation and phase shifts? | `ring_attractor.pdf` |
| `06_manifold_clouds.py` | What do circle, sphere, torus, and disk filtrations look like? | `topology_clouds.pdf` and persistence arrays |
| `07_neural_persistence.py` | Is coordinated angular tuning distinguishable from a declared independence null? | `persistence.pdf` and `neural_ring_analysis.npz` |
| `08_trained_rnn.py` | Does a small delayed-memory RNN exhibit stable states? | `rnn_dynamics.pdf`, checkpoint, and hidden states |
| `09_competing_dynamics.py` | Can identical ring trajectories come from mechanisms separated by an intervention? | `null_hypotheses.pdf` |

The shared `recurrent_tools.py` handles complete recurrent states, including both hidden and cell states for LSTMs. The trained example is a tanh RNN, with GRU/LSTM Jacobian utility checks only.

## Saved arrays and checkpoint

`neural_ring_analysis.npz` stores training/test rates, generating test angles, PCA transformation, persistence diagrams, and null statistics. The cloud files store full ambient coordinates and H0/H1/H2 diagrams. `rnn_probe_states.npz` stores trials × time × hidden-unit arrays, readouts, cues, accepted fixed points, PCA parameters, and the loss history. The data can be inspected using `numpy.load`.

`delayed_memory_checkpoint.pt` stores model/readout state dictionaries, optimizer state, random-number states, seed, training horizon, and hidden size. It is a checkpoint generated by this project. PyTorch's serialization policy may require explicit `weights_only=False` to load the complete optimizer/RNG payload. Use `weights_only=False` only for checkpoints whose origin you trust. Read-only analysis can use the provided NumPy arrays without loading the checkpoint.

## Evidence and scope

The main recurring distinctions are: geometry versus topology; observations versus complete states; invariant sets versus attraction; stable points versus stable sets; learning dynamics versus inference dynamics; and a statistical rejection versus identification of a mechanism. The literature chapter states authors' actual claims and distinguishes limitations from later overinterpretations.

Source audit files list verification URLs and scope qualifications. The merged bibliography contains one entry per retained source, with explicit primary research, textbook, preprint, and software-documentation categories reflected in its metadata. Numerical outputs are educational results for the declared settings. Root searches are not certified exhaustive; a near-zero eigenvalue in a finite ring network is not proof of exact continuous symmetry; rejecting a neuron-independence null does not establish recurrent generation.
