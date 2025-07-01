This repository contains useful files as part of my NSCI0017 research project. Topic: "Fuel Molecule Generation Using Generative Adversarial Networks"

## Repository Structure
- TenGAN (tg/): a PyTorch‑Lightning implementation of modified TenGAN, a transformer encoder GAN for _de novo_ molecular generation,
- Gaussian workflow (g16/): scripts to convert SMILES to Gaussian input (.gjf) and submit DFT g16 jobs,
- HPC helpers (myriad/): various files useful for Myriad HPC, including the TenGAN job submission files for manually set parameters, and parameters queried from the included `TG_jobList.csv` file,
- Environment spec (env/tg_env.yml): reproducible Conda environment,
- Reports (reports/): summary tables of job outputs,
- Writin (writing/): contains separate folders for semi up to date latex files of my dissertation and manuscript, as well as a placeholder folder for figures.

## TenGAN model set-up
For detailed instructions on setting up the environment, training the TenGAN model, pre-training its components, running adversarial training, and evaluating results, please refer to the dedicated README in the `tg` subdirectory:
**[tg/README.md](tg/README.md)**

> "All models are wrong, but some are useful."
> — George E. P. Box

TODO: Add top model(s) to `/tg/res`, change main.py parameter defaults to them.

TODO: Add TenGAN pipeline (single job submittion, outputs, explanations)

TODO: Add DFT pipeline (converting smiles to gjf, single job submittion)

TODO: Add post-processing section (summary tables, getting loss data and plotting it, plotting validity, uniqueness, etc.)

TODO: Add licenses? Verify.
