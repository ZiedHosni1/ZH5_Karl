#!/usr/bin/env python3
"""
tb2csv.py

Flatten all TensorBoard scalars (train/val losses, etc.) into one CSV.
Usage:
    python tb2csv.py --logdir PATH/TO/lightning_logs --outfile losses.csv
"""
import argparse
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def main():
    p = argparse.ArgumentParser(description="Convert TensorBoard logs to CSV")
    p.add_argument('--logdir',  required=True,
                   help='Path to lightning_logs folder')
    p.add_argument('--outfile', required=True, help='Path to output CSV file')
    args = p.parse_args()

    ea = event_accumulator.EventAccumulator(
        args.logdir, size_guidance={'scalars': 0})
    ea.Reload()

    rows = []
    for tag in ea.Tags()['scalars']:
        for event in ea.Scalars(tag):
            rows.append({
                'wall_time': event.wall_time,
                'step': event.step,
                'tag': tag,
                'value': event.value
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.outfile, index=False)
    print(f"Wrote {args.outfile}")


if __name__ == '__main__':
    main()
