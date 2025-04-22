#!/usr/bin/env python3
"""
plot_results.py

Generate quick‐look PNG plots from trained_results.csv for a TenGAN run.

Usage:
    python plot_results.py --run_dir PATH/TO/tg_jobs/JOBID
"""
import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import SciencePlots
plt.style.use('science')


def main():
    p = argparse.ArgumentParser(description="Plot TenGAN training metrics")
    p.add_argument('--run_dir', required=True,
                   help="Path to tg_jobs/JOBID folder")
    args = p.parse_args()
    R = pathlib.Path(args.run_dir)

    # Load adversarial‐training stats
    df = pd.read_csv(R / 'trained_results.csv')

    # Plot Synthesizability mean ± SD
    plt.figure(figsize=(8, 4))
    plt.plot(df['Epoch'], df['Mean'], label='Mean SA')
    plt.fill_between(
        df['Epoch'],
        df['Mean'] - df['Std'],
        df['Mean'] + df['Std'],
        alpha=0.25,
        label='±1 SD'
    )
    plt.xlabel('Adversarial epoch')
    plt.ylabel('Synthesizability (0.1–1.0)')
    plt.ylim(0.1, 1.0)
    plt.title('SA reward vs. epoch')
    plt.legend()
    plt.tight_layout()
    (R / 'plots').mkdir(exist_ok=True, parents=True)
    plt.savefig(R / 'plots' / 'sa_progress.png', dpi=180)
    plt.close()

    # Plot other metrics
    for col in ['Validity', 'Uniqueness', 'Novelty', 'Diversity']:
        plt.figure(figsize=(6, 3))
        plt.plot(df['Epoch'], df[col])
        plt.xlabel('Epoch')
        plt.ylabel(col)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(R / 'plots' / f'{col.lower()}.png', dpi=160)
        plt.close()


if __name__ == '__main__':
    main()
