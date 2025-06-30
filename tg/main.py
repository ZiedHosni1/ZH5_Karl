import argparse
import copy
import glob
import os
import time
import warnings

import numpy as np
import pytorch_lightning
import torch
import wandb
from data_iter import DisDataLoader, GenDataLoader
from discriminator import DiscriminatorModel
from generator import GeneratorModel, GenSampler
from mol_metrics import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from rdkit import rdBase
from rdkit.Chem import Draw
from rollout import OwnModel, Rollout
from utils import *
from utils import evaluation

rdBase.DisableLog("rdApp.error")
warnings.filterwarnings("ignore")

# ===========================
# TODO: after completing experiments with TenGAN for my project, change the defaults to the best one from HPO

# Default settings
parser = argparse.ArgumentParser()
# General hyperparameters
parser.add_argument(
    "--dataset_name", type=str, default="ZINC", help="use QM9 or ZINC dataset"
)
parser.add_argument(
    "--max_len",
    type=int,
    default=70,
    help="the max length of SMILES data: 60 for QM9, 70 for ZINC",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="the batch size for both generator and discriminator",
)

# ===========================
# Generator
parser.add_argument(
    "--gen_pretrain", action="store_true", help="whether pretrain the dataset"
)
parser.add_argument(
    "--generated_num",
    type=int,
    default=10000,
    help="generate size of the negative file: 5000 for QM9, 10000 for ZINC",
)
parser.add_argument(
    "--gen_train_size",
    type=int,
    default=9600,
    help="the size of training data: 4800 for QM9, 9600 for ZINC",
)
parser.add_argument(
    "--gen_num_encoder_layers",
    type=int,
    default=4,
    help="the number of transformer encoder layers",
)
parser.add_argument(
    "--gen_d_model", type=int, default=128, help="the dimension of the embedding"
)
parser.add_argument(
    "--gen_dim_feedforward",
    type=int,
    default=1024,
    help="the dimension of the feedforward layer",
)
parser.add_argument("--gen_num_heads", type=int, default=4, help="the number of heads")
parser.add_argument(
    "--gen_max_lr", type=float, default=8e-4, help="the maximum learning rate"
)
parser.add_argument(
    "--gen_dropout", type=float, default=0.1, help="the dropout probability"
)
parser.add_argument(
    "--gen_epochs", type=int, default=150, help="the pretraining epochs"
)

# ===========================
# Discriminator
parser.add_argument(
    "--dis_pretrain", action="store_true", help="whether pretraining the discriminator"
)
parser.add_argument("--dis_wgan", action="store_true", help="whether applying the WGAN")
parser.add_argument(
    "--dis_minibatch",
    action="store_true",
    help="whether applying the mini-batch discrimination",
)
parser.add_argument(
    "--dis_num_encoder_layers",
    type=int,
    default=4,
    help="the number of transformer encoder layers of the discriminator",
)
parser.add_argument(
    "--dis_d_model",
    type=int,
    default=100,
    help="the dimension of the embedding of the discriminator",
)
parser.add_argument(
    "--dis_num_heads",
    type=int,
    default=5,
    help="the number of heads of the discriminator",
)
parser.add_argument(
    "--dis_epochs",
    type=int,
    default=10,
    help="the pretraining epochs for the discriminator",
)
parser.add_argument(
    "--dis_feed_forward",
    type=int,
    default=200,
    help="the dimension of the feedforward layer",
)
parser.add_argument(
    "--dis_dropout", type=float, default=0.25, help="the dropout probability"
)

# ===========================
# Adversarial training
parser.add_argument(
    "--adversarial_train",
    action="store_true",
    help="whether adversarial trian TenGAN or Ten(W)GAN",
)
parser.add_argument("--update_rate", type=float, default=0.8, help="the update rate")
parser.add_argument(
    "--properties",
    type=str,
    default="druglikeness",
    help="the chemical property for molecular generation (druglikeness, solubility, or synthesizability)",
)
parser.add_argument(
    "--dis_lambda",
    type=float,
    default=0.5,
    help="the tradeoff between RL and GAN. If 0: NAIVE elif 1: SeqGAN else: TenGAN",
)
parser.add_argument(
    "--adv_lr", type=float, default=8e-5, help="the learning rate for the GAN or WGAN"
)
parser.add_argument(
    "--save_name", type=int, default=66, help="the name of the loaded model"
)
parser.add_argument(
    "--roll_num",
    type=int,
    default=8,
    help="the rollout times for Monte Carlo tree search: 16 for QM9, 8 for ZINC",
)
parser.add_argument(
    "--adv_epochs",
    type=int,
    default=100,
    help="the adverarial training epochs for TenGAN or Ten(W)GAN",
)
args = parser.parse_known_args()[0]

# ===========================
# Other model paths
POSITIVE_FILE = (
    "dataset/" + args.dataset_name + ".csv"
)  # Save the real / original SMILES data
NEGATIVE_FILE = (
    "res/generated_smiles_" + args.dataset_name + ".csv"
)  # Save the generated SMILES data

if args.dis_lambda == 0:
    MODEL_NAME = "Naive"
elif args.dis_lambda == 1:
    MODEL_NAME = "SeqGAN"
else:
    MODEL_NAME = "TenGAN_" + str(args.dis_lambda)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0,1,2,3
GPUS = 0
# TODO: Add an argument for GPUs instead, not good to have to modify the scripts like that to move between CPU-only and GPU runs.

if args.dis_wgan:
    DIS_MAX_LR = 8e-4
else:
    DIS_MAX_LR = 8e-7

G_STEP = 1
if args.dis_wgan:
    D_STEP = 3
else:
    D_STEP = 1

# File paths
PATHS = (
    "res/save_models/"
    + args.dataset_name
    + "/"
    + MODEL_NAME
    + "/rollout_"
    + str(args.roll_num)
    + "/"
    + "/batch_"
    + str(args.batch_size)
    + "/"
    + args.properties
)
if not os.path.exists(PATHS):
    os.makedirs(PATHS)

# Save the pre-trained generator and discriminator
G_PRETRAINED_MODEL = PATHS + "/g_pretrained.pkl"
D_PRETRAINED_MODEL = PATHS + "/d_pretrained.pkl"
PROPERTY_FILE = PATHS + "/trained_results.csv"

# Save adversarial training information
if args.adversarial_train:
    with open(PROPERTY_FILE, "a+") as wf:
        wf.truncate(0)
        wf.write(
            "{},{},{},{},{},{},{},{},{},{}\n".format(
                "Epoch",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Validity",
                "Uniqueness",
                "Novelty",
                "Diversity",
                "Time",
            )
        )
TenGAN_G_MODEL = PATHS + "/Epoch_" + str(args.save_name) + "_gen.pkl"
TenGAN_D_MODEL = PATHS + "/Epoch_" + str(args.save_name) + "_dis.pkl"

args = parser.parse_args()

# ===========================
print("\n\n\nVocabulary Information:")
print("==================================================================")
tokenizer = Tokenizer()
tokenizer.build_vocab()
print(tokenizer.char_to_int)

# Save all hyperparameters
with open(PATHS + "/hyperparameters.csv", "a+") as hp:
    # Clean the hyperparameters file
    hp.truncate(0)

    params = {}
    print("\n\nParameter Information:")
    print("==================================================================")
    params["POSITIVE_FILE"] = POSITIVE_FILE
    params["NEGATIVE_FILE"] = NEGATIVE_FILE
    params["G_PRETRAINED_MODEL"] = G_PRETRAINED_MODEL
    params["D_PRETRAINED_MODEL"] = D_PRETRAINED_MODEL
    params["PROPERTY_FILE"] = PROPERTY_FILE
    params["BATCH_SIZE"] = args.batch_size
    params["MAX_LEN"] = args.max_len
    params["VOCAB_SIZE"] = len(tokenizer.char_to_int)
    params["DEVICE"] = DEVICE
    params["GPUS"] = GPUS
    for param in params:
        string = param + " " * (25 - len(param))
        print("{}:   {}".format(string, params[param]))
        pass
    print("\n")

    params = {}
    params["GEN_PRETRAIN"] = args.gen_pretrain
    params["GENERATED_NUM"] = args.generated_num
    params["GEN_TRAIN_SIZE"] = args.gen_train_size
    params["GEN_NUM_ENCODER_LAYERS"] = args.gen_num_encoder_layers
    params["GEN_DIM_FEEDFORWARD"] = args.gen_dim_feedforward
    params["GEN_D_MODEL"] = args.gen_d_model
    params["GEN_NUM_HEADS"] = args.gen_num_heads
    params["GEN_MAX_LR"] = args.gen_max_lr
    params["GEN_DROPOUT"] = args.gen_dropout
    params["GEN_EPOCHS"] = args.gen_epochs
    for param in params:
        string = param + " " * (25 - len(param))
        print("{}:   {}".format(string, params[param]))
        pass
    print("\n")

    params = {}
    params["DIS_PRETRAIN"] = args.dis_pretrain
    params["DIS_WGAN"] = args.dis_wgan
    params["DIS_MINIBATCH"] = args.dis_minibatch
    params["DIS_NUM_ENCODER_LAYERS"] = args.dis_num_encoder_layers
    params["DIS_D_MODEL"] = args.dis_d_model
    params["DIS_NUM_HEADS"] = args.dis_num_heads
    params["DIS_MAX_LR"] = DIS_MAX_LR
    params["DIS_EPOCHS"] = args.dis_epochs
    params["DIS_FEED_FORWARD"] = args.dis_feed_forward
    params["DIS_DROPOUT"] = args.dis_dropout
    for param in params:
        string = param + " " * (25 - len(param))
        print("{}:   {}".format(string, params[param]))
        pass
    print("\n")

    params = {}
    params["ADVERSARIAL_TRAIN"] = args.adversarial_train
    params["PROPERTIES"] = args.properties
    params["DIS_LAMBDA"] = args.dis_lambda
    params["MODEL_NAME"] = MODEL_NAME
    params["UPDATE_RATE"] = args.update_rate
    params["ADV_LR"] = args.adv_lr
    params["G_STEP"] = G_STEP
    params["D_STEP"] = D_STEP
    params["ADV_EPOCHS"] = args.adv_epochs
    params["ROLL_NUM"] = args.roll_num
    for param in params:
        string = param + " " * (25 - len(param))
        print("{}:   {}".format(string, params[param]))
        pass
    print("==================================================================")


def pg_loss(probs, targets, rewards):
    """
    probs: [batch_size * max_len, vocab_size], logP of the gen output
    targets: [batch_size * max_len], integers
    rewards: [batch_size, seq_len]
    """
    one_hot = torch.zeros(probs.size(), dtype=torch.bool).to(
        DEVICE
    )  # [batch_size * max_len, vocab_size] with all 'False'
    # Set 1 for the token (vocab_size) of each row of one_hot
    one_hot.scatter_(
        1, targets.data.view(-1, 1), 1
    )  # [batch_size * max_len, vocab_size]
    # Select the values in probs according to one_hot
    loss = torch.masked_select(probs, one_hot)  # [batch_size * max_len]
    loss = loss * rewards.contiguous().view(-1)  # [batch_size * max_len]
    loss = -torch.sum(loss)
    return loss


# ============================================================================


def main():
    # ===========================
    # For reproducing experiments
    start_time = time.time()
    print(
        "\n\n\nStart time is {}\n\n\n".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
    )
    # Apply the seed to reproduct the results
    np.random.seed(27)
    torch.manual_seed(27)
    torch.cuda.manual_seed(27)

    wandb.init(
        project="TenGAN-SAF",  # A project to group your runs
        config=args,  # Save all argparse hyperparameters
        name=f"{MODEL_NAME}_{args.properties}_roll{args.roll_num}",  # Optional: give the run a name
    )
    # Use the config object from wandb to access hyperparameters
    # This ensures consistency if you use WandB sweeps later.
    config = wandb.config
    hyper_csv = os.path.join(PATHS, "hyperparameters.csv")
    if os.path.isfile(hyper_csv):  # safety check
        param_art = wandb.Artifact("hyperparameters", type="metadata")
        param_art.add_file(hyper_csv)
        wandb.log_artifact(param_art, aliases=["latest"])

    wandb.define_metric(
        "epoch"
    )  # defining both as for some its more clear to log per epoch, for others per step
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="epoch")
    global_step = 0

    run_logger = WandbLogger(experiment=wandb.run)

    early_stop_callback = EarlyStopping(
        monitor="gen_pre_val_loss",  # The metric to monitor
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode="min",
    )

    # ===========================
    # Generator objects definition
    gen_data_loader = GenDataLoader(POSITIVE_FILE, args.gen_train_size, args.batch_size)
    gen_data_loader.setup()
    gen = GeneratorModel(
        n_tokens=gen_data_loader.tokenizer.n_tokens,
        num_encoder_layers=args.gen_num_encoder_layers,
        dim_feedforward=args.gen_dim_feedforward,
        d_model=args.gen_d_model,
        nhead=args.gen_num_heads,
        dropout=args.gen_dropout,
        epochs=args.gen_epochs,
        max_lr=args.gen_max_lr,
    )

    gen_trainer = pytorch_lightning.Trainer(
        max_epochs=args.gen_epochs,
        gpus=GPUS,
        logger=run_logger,
        callbacks=[early_stop_callback],
        weights_summary=None,
        progress_bar_refresh_rate=5,
        gradient_clip_val=5.0,
        gradient_clip_algorithm="norm",
    )

    # Pre-train the generator
    if args.gen_pretrain:
        print("\n\nPre-train Generator...")
        gen_trainer.fit(gen, gen_data_loader)
        # Pre-train time cost
        print(
            "Generator Pre-train Time:\033[1;35m {:.2f}\033[0m hours".format(
                (time.time() - start_time) / 3600.0
            )
        )
        # Save the pre-trained generator model into file
        torch.save(gen.state_dict(), G_PRETRAINED_MODEL)
        artifact = wandb.Artifact(f"generator-{wandb.run.id}", type="model")
        artifact.add_file(G_PRETRAINED_MODEL)
        wandb.log_artifact(artifact)
    else:
        # Load the pre-trained generator
        print("\n\nLoad Pre-trained Generator.")
        gen.load_state_dict(torch.load(G_PRETRAINED_MODEL))
    gen.to(DEVICE)
    # Sample the generated data
    print("Generating {} samples...".format(args.generated_num))
    sampler = GenSampler(gen, gen_data_loader.tokenizer, args.batch_size, args.max_len)
    generated_smiles = sampler.sample_multi(args.generated_num, NEGATIVE_FILE)
    validity, uniqueness, novelty, diversity = evaluation(
        generated_smiles,
        gen_data_loader,
        args.properties,  # property_name
        logger=run_logger,  # WandbLogger created earlier
        step=global_step,  # keep curves on one x-axis
    )

    # ===========================
    # Discriminator objects definition
    dis_data_loader = DisDataLoader(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
    dis_data_loader.setup()
    dis = DiscriminatorModel(
        n_tokens=dis_data_loader.tokenizer.n_tokens,
        d_model=args.dis_d_model,
        nhead=args.dis_num_heads,
        num_encoder_layers=args.dis_num_encoder_layers,
        dim_feedforward=args.dis_feed_forward,
        dropout=args.dis_dropout,
        max_lr=DIS_MAX_LR,
        epochs=args.dis_epochs,
        pad_token=tokenizer.char_to_int[tokenizer.pad],
        dis_wgan=args.dis_wgan,
        minibatch=args.dis_minibatch,
    )

    dis_early_stop_callback = EarlyStopping(
        monitor="dis_pre_val_loss", patience=5, verbose=True, mode="min"
    )

    dis_trainer = pytorch_lightning.Trainer(
        max_epochs=args.dis_epochs,
        gpus=GPUS,
        logger=run_logger,
        callbacks=[dis_early_stop_callback],
        weights_summary=None,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
    )

    # Pre-train the discriminator
    if args.dis_pretrain:
        print("\n\nPre-train Discriminator...")
        dis_trainer.fit(dis, dis_data_loader)
        # Pre-train time cost
        print(
            "Discriminator Pre-train Time:\033[1;35m {:.2f}\033[0m hours".format(
                (time.time() - start_time) / 3600.0
            )
        )
        # Save the pre-trained discriminator model into file
        torch.save(dis.state_dict(), D_PRETRAINED_MODEL)
        artifact = wandb.Artifact(f"discriminator-{wandb.run.id}", type="model")
        artifact.add_file(D_PRETRAINED_MODEL)
        wandb.log_artifact(artifact)
    else:
        # Load the pre-trained discirminator
        print("\n\nLoad Pre-trained Discriminator.")
        dis.load_state_dict(torch.load(D_PRETRAINED_MODEL))
    dis.to(DEVICE)

    # define wandb watcher
    wandb.watch(models=[gen, dis], log="all", log_freq=100)

    # ===========================
    # Adversarial objects definition
    roll_own_model = OwnModel(  # Create deepcopy gen model for rollout sampling
        n_tokens=gen_data_loader.tokenizer.n_tokens,
        d_model=args.gen_d_model,
        nhead=args.gen_num_heads,
        num_encoder_layers=args.gen_num_encoder_layers,
        dim_feedforward=args.gen_dim_feedforward,
        dropout=args.gen_dropout,
    )
    dic = {}
    # Deep copy gen to roll_own_model
    gen.requires_grad = False
    for name, param in gen.named_parameters():
        # Deepcopy: if model's parameter changes, parameters in own_model do not change
        dic[name] = copy.deepcopy(param.data)
    for name, param in roll_own_model.named_parameters():
        param.data = dic[name]
    roll_own_model.to(DEVICE)
    gen.requires_grad = True
    adv_trainer = pytorch_lightning.Trainer(
        max_epochs=D_STEP,
        gpus=GPUS,
        logger=run_logger,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )

    pg_optimizer = torch.optim.Adam(params=gen.parameters(), lr=args.adv_lr)
    rollout = Rollout(gen, roll_own_model, tokenizer, args.update_rate, DEVICE)

    # Adversarial training
    if args.adversarial_train:
        print("\n\nAdversarial Training...")

        # --- Define Optimizer and LR Scheduler ---
        pg_optimizer = torch.optim.Adam(params=gen.parameters(), lr=args.adv_lr)
        adv_scheduler = torch.optim.lr_scheduler.StepLR(
            pg_optimizer, step_size=20, gamma=0.9
        )

        rollout = Rollout(gen, roll_own_model, tokenizer, args.update_rate, DEVICE)

        # --- Loop through adversarial epochs ---
        for epoch in range(args.adv_epochs):
            for g_step in range(G_STEP):
                # Sampling a batch of samples
                samples = sampler.sample()

                # Encode the generated samples
                encoded = [torch.tensor(tokenizer.encode(s)) for s in samples]
                encoded = torch.nn.utils.rnn.pad_sequence(encoded).squeeze().to(DEVICE)

                # Forward pass through generator
                gen_pred = gen.forward(encoded[:-1])
                gen_pred = gen_pred.transpose(0, 1)
                gen_pred = gen_pred.contiguous().view(-1, gen_pred.size(-1))
                gen_pred = torch.nn.functional.log_softmax(gen_pred, dim=1)

                targets = (
                    encoded[1:]
                    .transpose(0, 1)
                    .contiguous()
                    .view(
                        -1,
                    )
                )

                # Calculate rewards using rollout
                rewards = rollout.get_reward(
                    samples,
                    sampler,
                    args.roll_num,
                    dis,
                    args.dis_lambda,
                    args.properties,
                )
                rewards = torch.tensor(rewards).to(DEVICE)

                # Compute and backpropagate policy gradient loss
                loss = pg_loss(gen_pred, targets, rewards)

                print(
                    "\n\n\n\033[1;35mEpoch {}\033[0m / {}, G_STEP {} / {}, PG_Loss: {:.3f}".format(
                        epoch + 1, args.adv_epochs, g_step + 1, G_STEP, loss.item()
                    )
                )

                pg_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 5, norm_type=2)
                pg_optimizer.step()

                global_step += 1
                wandb.log(
                    {
                        "adv_gen_pg_loss": loss.item(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    commit=True,
                )
                batch_size = encoded.size(1)
                disc_inputs = encoded  # [n_valid, seq_len]
                disc_labels = torch.zeros(
                    disc_inputs.size(0), dtype=torch.long, device=DEVICE
                )
                disc_loss, disc_acc = dis.step((disc_inputs, disc_labels))
                wandb.log(
                    {
                        "adv_dis_loss": disc_loss.item(),
                        "adv_dis_acc": disc_acc.item(),
                    },
                    step=global_step,
                    commit=False,  # groups with previous log
                )

            adv_scheduler.step()

            # Update models
            rollout.update_params()

            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.adv_epochs:
                g_model_path = os.path.join(PATHS, f"Epoch_{epoch + 1}_gen.pkl")
                torch.save(gen.state_dict(), g_model_path)

                adv_artifact = wandb.Artifact(
                    f"{wandb.run.name}-checkpoint", type="model"
                )
                adv_artifact.add_file(g_model_path, name=f"gen_epoch_{epoch + 1}.pkl")

                if args.dis_lambda > 0:
                    d_model_path = os.path.join(PATHS, f"Epoch_{epoch + 1}_dis.pkl")
                    torch.save(dis.state_dict(), d_model_path)
                    adv_artifact.add_file(
                        d_model_path, name=f"dis_epoch_{epoch + 1}.pkl"
                    )

                wandb.log_artifact(adv_artifact)

            # Generate and Evaluate Samples
            print("Generating {} samples for evaluation...".format(args.generated_num))
            generated_smiles = sampler.sample_multi(args.generated_num, NEGATIVE_FILE)

            smiles_artifact = wandb.Artifact(
                f"generated_smiles_epoch_{epoch + 1}", type="dataset"
            )
            smiles_artifact.add_file(NEGATIVE_FILE)
            wandb.log_artifact(smiles_artifact, aliases=[f"epoch_{epoch + 1}"])

            current_time = (time.time() - start_time) / 3600.0
            print(
                "\nTotal Computational Time: \033[1;35m {:.2f} \033[0m hours.".format(
                    current_time
                )
            )

            evaluation(
                generated_smiles,
                gen_data_loader,
                args.properties,  # property_name
                logger=run_logger,  # WandbLogger
                step=global_step,
                time=current_time,
                epoch=epoch,
            )

            # Train discriminator
            if args.dis_lambda > 0:
                print("Updating Discriminator...")
                dis_data_loader.setup()
                adv_trainer.fit(dis, dis_data_loader)

    # Show Top-12 molecules
    if not os.path.isfile(NEGATIVE_FILE):
        print("Generated dataset does NOT exist!\n")
    else:
        print("Top-12 Molecules of [{}]:".format(args.properties))
        top_mols, top_scores = top_mols_show(NEGATIVE_FILE, args.properties)
        img = Draw.MolsToGridImage(
            top_mols[:], molsPerRow=3, subImgSize=(1000, 1000), legends=top_scores[:]
        )
        if args.dis_wgan:
            img.save("res/top_12_w.pdf")
        else:
            img.save("res/top_12.pdf")
        print("*" * 80)

    # Figure out distributions
    all_files = glob.glob("res/*.csv")
    print("\n\nFile names for drawing distributions:", all_files)
    # NOTE: Add '_w.csv' to the name of WGAN file
    if len(all_files) == 2:
        distribution(POSITIVE_FILE, all_files[0], all_files[1])
    else:
        print("Distributions are not generated.")
    print("*" * 80)

    print("\n--- Generating Loss Plots ---")
    try:
        plot_script = os.path.join(os.path.dirname(__file__), "plot_losses.py")
        # Pass the directory containing the log files/subdirs
        log_parent_dir = PATHS
        plot_cmd = f'python {plot_script} --log_dir "{log_parent_dir}"'
        print(f"Executing: {plot_cmd}")
        os.system(plot_cmd)
        print("--- Plot generation complete ---")
    except Exception as e:
        print(f"ERROR: Failed to generate plots: {e}")

    wandb.finish()


# ============================================================================
if __name__ == "__main__":
    main()
