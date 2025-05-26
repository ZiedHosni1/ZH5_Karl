# TenGAN

A PyTorch implementation of "TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation."
The paper has been accepted by [AISTATS 2024](https://). ![Overview of TenGAN](https://github.com/naruto7283/TenGAN/blob/main/tengan_overview.png)

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
	  - comb_1.csv - self-composed dataset of (potential) fuel molecules
   
  - **res:** all generated datasets, saved models, and experimental results are automatically saved in this folder. Currently it is empty as I search for the best-performing models. 
	- save_models: all training results, pre-trained and trained filler and discriminator models are saved in this folder.

	- main.py: definite all hyper-parameters, pretraining of the generator, pretraining of the discriminator, adversarial training of the TenGAN and Ten(W)GAN.
	
	- mol_metrics.py: definite the vocabulary, tokenization of SMILES strings, and all the objective functions of the chemical properties.	

	- data_iter.py: load data for the generator and discriminator.

	- generator.py: defines the generator.

	- discriminator.py: defines the discriminator.

	- rollout.py: defines the Monte Carlo method.

	- utils.py: defines the performance evaluation methods of the generated molecules, such as the validity, uniqueness, novelty, and diversity. 

## Available Chemical Properties at Present:
	- solubility
	- druglikeness
	- synthesizability
	- molar_nhoc
	- safscore
 
## Experimental Reproduction

  - TenGAN on the comb-1 dataset with safscore as the optimized property, full pre-train and train:
  ```
  $ python main.py --dataset_name comb_1 --properties safscore --gen_pretrain --dis_pretrian --adv_training
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
