This repository contains useful files as part of my NSCI0017 research project. Topic: "Sustainable Aviation Fuel Molecule Generation Using Generative Adversarial Networks"

## Repository Structure
- TenGAN (tg/): a PyTorch‑Lightning implementation of modified TenGAN, a transformer encoder GAN for _de novo_ molecular generation,
- Gaussian workflow (g16/): scripts to convert SMILES to Gaussian input (.gjf) and submit DFT g16 jobs,
- HPC helpers (myriad/): various files useful for Myriad HPC, including the TenGAN job submission files for manually set parameters, and parameters queried from the included `TG_jobList.csv` file,
- Environment spec (env/tg_env.yml): reproducible Conda environment,
- Reports (reports/): summary tables of job outputs,
- Writing (writing/): contains separate folders for semi up to date latex files of my dissertation and manuscript, as well as a placeholder folder for figures.
- Poster (poster/): a poster summarizing the project, available in pptx and pdf formats.

## TenGAN model set-up
For detailed instructions on setting up the environment, training the TenGAN model, pre-training its components, running adversarial training, and evaluating results, please refer to the dedicated README in the `tg` subdirectory:
**[tg/README.md](tg/README.md)**

> "All models are wrong, but some are useful."
> — George E. P. Box

