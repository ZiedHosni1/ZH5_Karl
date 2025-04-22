This repository contains useful files as part of my NSCI0017 research project. Topic: "Fuel Molecule Generation Using Generative Adversarial Networks"

This repository contains:

- TenGAN (tg/): a PyTorch‑Lightning implementation of a GAN for molecular generation,
- Gaussian workflow (g16/): scripts to convert SMILES to Gaussian input (.gjf) and submit DFT g16 jobs.
- HPC helpers (hpc/): job‑submission wrappers (qsub), TensorBoard2CSV converter, plotting scripts, etc.
- Input datasets (data/): used SMILES datasets.
- Environment spec (env/tg_env.yml): reproducible Conda environment.
- Reports (reports/): summary tables of job outputs.

> "All models are wrong, but some are useful."
> — George E. P. Box

TODO: Add Getting Started Section (copying repo, creating conda env, preparing necessary outer file structure, etc)

TODO: Add TenGAN pipeline (single job submittion, outputs, explanations)

TODO: Add DFT pipeline (converting smiles to gjf, single job submittion)

TODO: Add post-processing section (summary tables, getting loss data and plotting it, plotting validity, uniqueness, etc.)

TODO: Add licenses? Verify.
