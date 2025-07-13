import json
import logging

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from mol_metrics import *
from rdkit import Chem
from rdkit.Chem import QED, Draw


# ============================================================================
# Show Top-12 Molecules
def top_mols_show(filename, properties, log_to_wandb=True):
    """
    filename: NEGATIVE FILES (generated dataset of SMILES)
    properties: 'druglikeness' or 'solubility' or 'synthesizability'
    """
    mols, scores = [], []
    # Read the generated SMILES data
    smiles = open(filename, "r").read()
    smiles = list(smiles.split("\n"))

    if properties == "druglikeness":
        scores = batch_druglikeness(smiles)
    elif properties == "synthesizability":
        scores = batch_SA(smiles)
    elif properties == "solubility":
        scores = batch_solubility(smiles)
    elif properties == "nhoc":
        scores = batch_nhoc_gc(smiles)
    elif properties == "vol_nhoc":
        scores = batch_vol_nhoc_gc(smiles)
    else:
        raise ValueError(f"Unknown property: {properties}")

    # Sort the scores
    dic = dict(zip(smiles, scores))
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)

    flag = 0
    top_mols = []
    top_scores = []
    for smi, score in dic:
        if flag >= 12:
            break
        mol_obj = Chem.MolFromSmiles(smi)
        if mol_obj is None:
            continue

        if properties == "synthesizability":
            qed_ok = QED.default(mol_obj) >= 0.5
            if smi in top_mols or score <= 0.80 or not qed_ok:
                continue

        if smi not in top_mols:
            flag += 1
            top_mols.append(mol_obj)
            top_scores.append(f"{score:.3f}")
            print(smi, "\t", f"{score:.3f}")

    if log_to_wandb and top_mols and wandb.run is not None:
        table = wandb.Table(columns=["ID", "Molecule", "SMILES", "Score"])
        for i, (mol, score) in enumerate(zip(top_mols, top_scores)):
            table.add_data(
                f"Top-{i + 1}",
                wandb.Image(Draw.MolToImage(mol)),
                Chem.MolToSmiles(mol),
                float(score),
            )
        wandb.log({f"Top_12_{properties}": table})

    return top_mols, top_scores


# Figure out the distributions
def distribution(real_file, gan_file, wgan_file):
    """
    real_file: original training dataset
    gan_file: the file of generated STGAN data
    wgan_file: the file of generated ST(W)GAN data
    """

    # Read trian Dataset
    real_lines = open(real_file, "r").read()
    real_lines = list(real_lines.split("\n"))
    # Read STGAN results
    gan_lines = open(gan_file, "r").read()
    gan_lines = list(gan_lines.split("\n"))
    # Read ST(W)GAN results
    wgan_lines = open(wgan_file, "r").read()
    wgan_lines = list(wgan_lines.split("\n"))

    # Read the novel SMILES of STGAN
    gan_valid, gan_novelty = [], []
    for s in gan_lines:
        mol = Chem.MolFromSmiles(s)
        if mol and s != "":
            gan_valid.append(s)
    for s in list(set(gan_valid)):
        if s not in real_lines:
            gan_novelty.append(s)
    gan_lines = gan_novelty

    # Read the novel SMILES of ST(W)GAN
    wgan_valid, wgan_novelty = [], []
    for s in wgan_lines:
        mol = Chem.MolFromSmiles(s)
        if mol and s != "":
            wgan_valid.append(s)
    for s in list(set(wgan_valid)):
        if s not in real_lines:
            wgan_novelty.append(s)
    wgan_lines = wgan_novelty

    # Compute property scores for real dataset, STGAN and ST(W)GAN
    for name in ["QED Score", "SA Score", "logP Score"]:
        real_scores, gan_scores, wgan_scores = [], [], []
        if name == "QED Score":
            real_scores = batch_druglikeness(real_lines)
            gan_scores = batch_druglikeness(gan_lines)
            wgan_scores = batch_druglikeness(wgan_lines)
        elif name == "SA Score":
            real_scores = batch_SA(real_lines)
            gan_scores = batch_SA(gan_lines)
            wgan_scores = batch_SA(wgan_lines)
        elif name == "logP Score":
            real_scores = batch_solubility(real_lines)
            gan_scores = batch_solubility(gan_lines)
            wgan_scores = batch_solubility(wgan_lines)
        # Compute the mean socres
        avg = [np.mean(real_scores), np.mean(gan_scores), np.mean(wgan_scores)]
        # Print Mean socres
        print("Mean Real {}: {:.3f}".format(name, avg[0]))
        print("Mean GAN {}: {:.3f}".format(name, avg[1]))
        print("Mean WGAN {}: {:.3f}\n".format(name, avg[2]))

        # Plot distribution figures for real dataset, STGAN and ST(W)GAN
        plt.subplots(figsize=(12, 7))
        # Font size
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(name, size=15)
        plt.ylabel("Density", size=15)
        # Set the min and max value of X axis
        plt.xlim(-0.1, 1.1)

        sns.distplot(
            real_scores,
            hist=False,
            kde_kws={"shade": True, "linewidth": 1},
            label="ORIGINAL",
        )
        sns.distplot(
            gan_scores,
            hist=False,
            kde_kws={"shade": True, "linewidth": 1},
            label="STGAN",
        )
        sns.distplot(
            wgan_scores,
            hist=False,
            kde_kws={"shade": True, "linewidth": 1},
            label="ST(W)GAN",
        )
        plt.legend(loc="upper right", prop={"size": 15})
        plt.savefig("res/" + name + ".pdf")


def evaluation(
    generated_smiles,
    gen_data_loader,
    property_name,  # e.g. "synthesizability"
    logger=None,  # WandbLogger or None
    step=0,  # global_step for dashboard alignment
    time=None,
    epoch=None,
):
    generated_mols = np.array(
        [Chem.MolFromSmiles(s) for s in generated_smiles if len(s.strip())]
    )
    if len(generated_mols) == 0:
        print("No SMILES data is generated, please pre-train the generator again!")
        return 0, 0, 0, 0

    valid_smiles = [
        Chem.MolToSmiles(m)
        for m in generated_mols
        if m and m.GetNumAtoms() > 1 and Chem.MolToSmiles(m) != " "
    ]
    unique_smiles = list(set(valid_smiles))
    train_smiles = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in gen_data_loader.train_data
    ]
    novel_smiles = [s for s in unique_smiles if s not in train_smiles]

    validity = len(valid_smiles) / len(generated_mols)
    uniqueness = len(unique_smiles) / max(len(valid_smiles), 1)
    novelty = len(novel_smiles) / max(len(unique_smiles), 1)
    diversity = batch_diversity(novel_smiles)

    # console output, unchanged
    print("\nResults Report:")
    print("*" * 80)
    print(f"Total Mols:   {len(generated_mols)}")
    print(f"Validity:     {len(valid_smiles)}    ({validity * 100:.2f}%)")
    print(f"Uniqueness:   {len(unique_smiles)}    ({uniqueness * 100:.2f}%)")
    print(f"Novelty:      {len(novel_smiles)}    ({novelty * 100:.2f}%)")
    print(f"Diversity:    {diversity:.2f}\n")
    print("Samples of Novel SMILES:")
    for sm in novel_smiles[:5]:
        print(sm)
    print()

    # property scores
    if novel_smiles:
        vals = reward_fn(property_name, novel_smiles)
        mean_s, std_s = float(np.mean(vals)), float(np.std(vals))
        min_s, max_s = float(np.min(vals)), float(np.max(vals))
        print(
            f"[{property_name}]: [Mean: {mean_s:.3f}   STD: {std_s:.3f}   "
            f"MIN: {min_s:.3f}   MAX: {max_s:.3f}]"
        )
    else:
        mean_s = std_s = min_s = max_s = 0.0
        print("No novel SMILES generated!")

    print("*" * 80 + "\n")

    # structured metrics for WandB / CSV
    p = property_name
    metrics = {
        "prop/validity": validity,
        "prop/uniqueness": uniqueness,
        "prop/novelty": novelty,
        "prop/diversity": diversity,
        f"prop/{p}_mean": mean_s,
        f"prop/{p}_std": std_s,
        f"prop/{p}_min": min_s,
        f"prop/{p}_max": max_s,
    }
    metrics["prop/objective"] = (
        metrics[f"prop/{p}_mean"] * metrics["prop/validity"] * metrics["prop/novelty"]
    )
    metrics["prop/objective"] = float(
        (
            metrics[f"prop/{p}_mean"]
            * metrics["prop/validity"]
            * metrics["prop/uniqueness"]
            * metrics["prop/novelty"]
        )
        ** (1.0 / 4.0)
    )
    # added for wandb sweep, the main metric I will look at
    if logger is not None:
        logger.log_metrics(metrics, step=step)

    logging.getLogger("tengan").info("[eval] %s", json.dumps({"step": step, **metrics}))
    return validity, uniqueness, novelty, diversity, metrics["prop/objective"]
