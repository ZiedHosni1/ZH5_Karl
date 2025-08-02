# TenGAN


Model has been adapted from the original implementation of TenGAN to support the following features:
- New properties for fuel-relevant molecular generation, such as net heat of combustion (NHOC) and volume-normalised NHOC (vol_NHOC).
- Added early stopping mechanisms to pretraining processes.
- Overhauled the logging system to use Weight and Biases (wandb) for better tracking of experiments, including hyperparameters, losses, and generated molecules.

## Installation
Execute the following commands:
```bash
conda env create -n tg_env -f ../env/tg_env.yml
source activate tg_env
```

## File Description
  - **dataset:** contains the training datasets. Each dataset contains only one column of SMILES strings.
	  - QM9.csv
	  - ZINC.csv
	  - combinedSet_10k.csv - self-composed dataset of fuel-like molecules, composed of 10,000 molecules. 
      - combinedSet_100k.csv - self-composed dataset of fuel-like molecules, composed of 100,000 molecules. 

  - **res:** all generated datasets, saved models, and experimental results are automatically saved in this folder. Currently it is empty as I search for the best-performing models. 
	- save_models: all training results, pre-trained and trained filler and discriminator models are saved in this folder.

	- main.py: defines all hyper-parameters, pretraining of the generator, pretraining of the discriminator, adversarial training of the TenGAN and Ten(W)GAN.
	
	- mol_metrics.py: defines the vocabulary, tokenization of SMILES strings, and all the objective functions of the chemical properties.

	- data_iter.py: load data for the generator and discriminator.

	- generator.py: defines the generator.

	- discriminator.py: defines the discriminator.

	- rollout.py: defines the Monte Carlo method.

	- utils.py: defines the performance evaluation methods of the generated molecules, such as the validity, uniqueness, novelty, and diversity. 

## Available Chemical Properties at Present:
	- solubility
	- druglikeness
	- synthesizability
	- nhoc
	- vol_nhoc
 
## Experimental Reproduction

  - TenGAN on the combinedSet_10k dataset with vol_nhoc as the optimised property, full pre-train and train:
  ```
  $ python main.py --dataset_name combinedSet_10k --properties vol_nhoc --gen_pretrain --dis_pretrian --adv_training
  ```
  
## Model heavily adapted from
  ```
  C. Li and Y. Yamanishi (2024). TenGAN: Pure transformer encoders make an efficient discrete GAN for de novo molecular generation. AISTATS 2024.
  ```
  
  BibTeX format:
  ```
  @inproceedings{li2024tengan,
  title={TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation},
  author={Li, Chen and Yamanishi, Yoshihiro},
  booktitle={27th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  volume={２３８},
  year={2024}
  }
  ```
