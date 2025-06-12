import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import argparse
import os
import glob
plt.style.use(['science','no-latex'])

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Plot training losses from log files.")
parser.add_argument('--log_dir', type=str, required=True, help='Directory containing the log files (gen_pretrain_logs, dis_pretrain_logs, adversarial_losses.csv)')
args = parser.parse_args()

LOG_DIR = args.log_dir
print(f"Reading logs from: {LOG_DIR}")

# --- Find Log Files ---
# CSVLogger creates versioned subdirectories (e.g., version_0)
gen_log_subdir = os.path.join(LOG_DIR, "gen_pretrain_logs", "gen")
dis_log_subdir = os.path.join(LOG_DIR, "dis_pretrain_logs", "dis")
adv_log_file = os.path.join(LOG_DIR, "adversarial_losses.csv")

gen_metrics_files = glob.glob(os.path.join(gen_log_subdir, "version_*/metrics.csv"))
dis_metrics_files = glob.glob(os.path.join(dis_log_subdir, "version_*/metrics.csv"))

gen_metrics_file = gen_metrics_files[0] if gen_metrics_files else None
dis_metrics_file = dis_metrics_files[0] if dis_metrics_files else None

# --- Plotting Function ---
def plot_losses(df, train_loss_col, val_loss_col, title, filename):
    if df is None or df.empty:
        print(f"Skipping plot '{title}': Dataframe is empty or None.")
        return

    # Filter out rows where epoch is NaN (might happen with lightning logging)
    df = df.dropna(subset=['epoch'])
    df['epoch'] = df['epoch'].astype(int) # Ensure epoch is integer for plotting

    # Filter data for the specific columns, dropping NaNs for each series individually
    train_data = df[['epoch', train_loss_col]].dropna()
    val_data = df[['epoch', val_loss_col]].dropna()

    if train_data.empty and val_data.empty:
        print(f"Skipping plot '{title}': No valid data found for specified columns.")
        return

    print(f"Plotting '{title}'...")
    try:
            fig, ax = plt.subplots()
            if not train_data.empty:
                ax.plot(train_data['epoch'], train_data[train_loss_col], label='Train Loss', marker='o', markersize=3, linestyle='-')
            if not val_data.empty:
                ax.plot(val_data['epoch'], val_data[val_loss_col], label='Validation Loss', marker='s', markersize=3, linestyle='--')

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()
            ax.autoscale(tight=True)
            fig.tight_layout()
            plot_path = os.path.join(LOG_DIR, filename)
            fig.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
            plt.close(fig) # Close the figure to free memory
    except KeyError as e:
        print(f"KeyError while plotting '{title}': {e}. Check if columns exist in the CSV.")
    except Exception as e:
        print(f"Error during plotting '{title}': {e}")


# --- Generate Plots ---

# Plot Generator Pre-training Loss
if gen_metrics_file and os.path.exists(gen_metrics_file):
    try:
        gen_df = pd.read_csv(gen_metrics_file)
        # Check if required columns exist
        if 'gen_pre_train_loss' in gen_df.columns and 'gen_pre_val_loss' in gen_df.columns:
             plot_losses(gen_df, 'gen_pre_train_loss', 'gen_pre_val_loss', 'Generator Pre-training Loss', 'gen_pretrain_loss.pdf')
        else:
             print("Generator pre-training loss columns not found in metrics.csv.")
    except pd.errors.EmptyDataError:
        print(f"Warning: Generator metrics file {gen_metrics_file} is empty.")
    except Exception as e:
        print(f"Error reading or processing Generator metrics file {gen_metrics_file}: {e}")
else:
    print("Generator pre-training metrics file not found.")

# Plot Discriminator Pre-training Loss
if dis_metrics_file and os.path.exists(dis_metrics_file):
    try:
        dis_df = pd.read_csv(dis_metrics_file)
        # Check if required columns exist
        if 'dis_pre_train_loss' in dis_df.columns and 'dis_pre_val_loss' in dis_df.columns:
            plot_losses(dis_df, 'dis_pre_train_loss', 'dis_pre_val_loss', 'Discriminator Pre-training Loss', 'dis_pretrain_loss.pdf')
        else:
            print("Discriminator pre-training loss columns not found in metrics.csv.")
    except pd.errors.EmptyDataError:
        print(f"Warning: Discriminator metrics file {dis_metrics_file} is empty.")
    except Exception as e:
        print(f"Error reading or processing Discriminator metrics file {dis_metrics_file}: {e}")
else:
    print("Discriminator pre-training metrics file not found.")

# Plot Adversarial Training Loss (Generator PG Loss only for now)
if os.path.exists(adv_log_file):
    try:
        adv_df = pd.read_csv(adv_log_file)
        if not adv_df.empty and 'gen_pg_loss' in adv_df.columns:
            print("Plotting Adversarial Training Generator Loss...")
            with plt.style.context(['science', 'grid']):
                fig, ax = plt.subplots()
                # Ensure 'epoch' column exists and is suitable for x-axis
                if 'epoch' in adv_df.columns:
                    adv_plot_data = adv_df[['epoch', 'gen_pg_loss']].dropna()
                    if not adv_plot_data.empty:
                        ax.plot(adv_plot_data['epoch'], adv_plot_data['gen_pg_loss'], label='Generator PG Loss', marker='o', markersize=3, linestyle='-')
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        ax.set_title("Adversarial Training Generator Loss")
                        ax.legend()
                        ax.autoscale(tight=True)
                        fig.tight_layout()
                        plot_path = os.path.join(LOG_DIR, 'adv_gen_loss.pdf')
                        fig.savefig(plot_path)
                        print(f"Saved plot to {plot_path}")
                        plt.close(fig) # Close the figure
                    else:
                        print("No valid data points for adversarial generator loss plot.")
                else:
                    print("Epoch column missing in adversarial_losses.csv.")
        else:
             print("Adversarial loss file is empty or missing 'gen_pg_loss' column.")
    except pd.errors.EmptyDataError:
        print(f"Warning: Adversarial loss file {adv_log_file} is empty.")
    except Exception as e:
        print(f"Error reading or processing adversarial loss file {adv_log_file}: {e}")
else:
    print("Adversarial loss file not found.")
