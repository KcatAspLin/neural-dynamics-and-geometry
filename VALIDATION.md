# Validation record

**Mathematical Foundations of Neural Dynamics and Geometry**  
Foundations edition, 10 November 2026.

This record describes checks actually performed for the delivered edition. It is a reproducibility record, not independent peer review or a claim of exhaustive mathematical verification.

## Manuscript and compilation

- The compiled PDF contains 148 pages, including front matter, 13 chapters, five appendices, and the bibliography.
- The manuscript contains 97 exercises labeled introductory, intermediate, or advanced. Exercises are practice questions; a complete solutions manual is not included.
- The merged BibTeX database contains 56 entries. The `research/` audits identify source verification URLs and relevant qualifications. Citation aliases were resolved into the merged database.
- The book was built using pdfLaTeX, BibTeX, and two further pdfLaTeX passes. The final build has no unresolved citations or cross-references, multiply defined labels, overfull boxes, missing figures, or duplicate PDF page destinations.
- Every PDF page was included in the visual review. Representative mathematical passages, tables, figures, and code listings were inspected at larger size. Final cover, proposition, listing, and chapter-end corrections were inspected again.
- The mathematical review focused on theorem hypotheses, local versus global conclusions, autonomous versus driven dynamics, continuous versus discrete stability, manifold invariance, homology conventions, statistical null hypotheses, and the strength of claims about mechanisms.
- The source archive includes the generated figures, so compilation does not require Python execution. The packaged source was also compiled from a clean extracted directory.

## Computational execution

All nine numbered scripts completed through `python code/run_all.py --seed 17`. The runner used one numerical-library thread, and each script reset its own seed. The main environment was Python 3.12.14 on Linux with CPU PyTorch. Exact tested versions are in `code/requirements-tested.txt`; `results/environment.json` and each numbered JSON record additional provenance.

| Tutorial | Executed check or recorded result | Interpretation limit |
|---|---|---|
| 1. PCA and dimension | The clean circle has intrinsic dimension 1 and linear rank 2; participation ratio is approximately 1.998. Noise raises small-neighborhood dimension estimates. | Different dimension definitions answer different questions. Held-out full-coordinate PCA reconstruction error alone is not a rank-selection rule. |
| 2. Embeddings | On the declared Swiss roll, connected Isomap with six neighbors has intrinsic stress approximately 0.0322; using twelve neighbors raises it to approximately 0.4126. | These are one synthetic dataset and specified parameter settings, not a general algorithm ranking. |
| 3. Phase portraits | The Hopf oscillator radius agrees with the analytical value to within approximately 5.3e-10. The Euler example distinguishes numerical instability from ODE stability. | Numerical accuracy in these examples does not validate arbitrary integration settings. |
| 4. Fixed points | Three roots were located. The outer roots have Jacobian eigenvalues approximately -0.6057 and -0.3; the origin has 0.5 and -0.3. Analytical and finite-difference Jacobians agree within the declared tolerance. | Numerical root finding alone does not certify completeness. |
| 5. Ring network | The dimensionless equilibrium-map residual is approximately 1.68e-14. A radial perturbation relaxes, while a twelve-cell phase displacement persists. The leading eigenvalue is numerically near zero. | A finite discrete network does not establish exact continuous symmetry. Velocity residuals require division by the time constant. |
| 6. Point-cloud topology | Circle, sphere, product torus, and filled-disk data were analyzed in their full ambient coordinates with H0, H1, and H2 over the two-element field. Diagrams and finite/censored counts were saved. | Two plotted coordinates are not the space in which persistence was computed. A finite filtration cutoff requires censoring care. |
| 7. Neural persistence | The test maximum H1 lifetime is approximately 9.1655 versus a maximum surrogate lifetime of approximately 1.3168; the plus-one randomization p-value is 0.01 for 99 surrogates. | This rejects the specified neuron-column independence null for IID observations. It does not identify recurrent dynamics or establish attraction. |
| 8. Trained RNN | A 16-unit tanh delayed-memory RNN was trained for 650 updates. Test accuracy is 1.0 in this run. Two stable roots have spectral radii approximately 0.6211 and 0.5231; a third root has radius approximately 1.7448. | This is one trained tanh RNN. Root-search completeness is not certified. GRU and LSTM checks cover Jacobian utilities, not trained task performance. |
| 9. Competing dynamics | Unperturbed trajectories agree to approximately 1.72e-10. After a radial perturbation, the mean response contrast is approximately 0.3024 with a randomization p-value of 0.001 for 999 permutations. | The intervention distinguishes these two specified models; other driven models with feedback can reproduce similar responses. |

The saved checkpoint, arrays, JSON metrics, and figures are included. The scripts contain checks tied to known solutions, Jacobians, root residuals, shapes, or declared scientific comparisons. The broader dependency ranges in `requirements.txt` were not tested as a full compatibility matrix. A fixed seed does not guarantee identical results across software versions or platforms.

## Scope

All requested chapters are written as an introductory textbook-style course. Stochastic calculus, center-manifold theory, singular perturbation, advanced algebraic topology, Koopman theory, and related extensions are introduced with their scope and prerequisites stated; the book does not claim a full graduate treatment of each field. Persistent-cohomology circular coordinates are explained but are not implemented as a decoder in the tutorials.

All computational examples use synthetic data or models trained within this project. No biological dataset was reanalyzed. The literature chapter distinguishes reported results, the authors' claims, alternative explanations, and broader interpretations. The bibliography is curated to support the course rather than offered as an exhaustive review of every recent publication.
