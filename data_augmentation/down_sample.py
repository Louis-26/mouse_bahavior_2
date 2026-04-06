#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from sampling_common import NEG_LABEL, POS_LABEL, compute_csv_class_percentages, copy_unchanged_pair, load_segments, rebuild_pair, summarize_segments


TRAIN_BASENAMES = ("CQ_2", "CQ_3")
KEEP_UNCHANGED_BASENAMES = ("CQ_4",)
DEFAULT_TARGET_NEGATIVE_RATIO = 0.53


def choose_downsampled_segments(segments, target_negative_ratio):
    positives = [segment for segment in segments if segment["label"] == POS_LABEL]
    negatives = [segment for segment in segments if segment["label"] == NEG_LABEL]

    if not positives:
        raise RuntimeError("No scratching segments found for down-sampling.")
    if not negatives:
        raise RuntimeError("No no-behavior segments found for down-sampling.")

    summarize_segments("Original distribution", segments)

    positive_duration = sum(segment["duration_sec"] for segment in positives)
    negative_duration = sum(segment["duration_sec"] for segment in negatives)
    target_negative_duration = positive_duration * target_negative_ratio / (1.0 - target_negative_ratio)

    if negative_duration <= target_negative_duration:
        print("Negative duration is already within target range. Keeping original segments.")
        selected_negative_rows = {segment["row_idx"] for segment in negatives}
    else:
        shuffled_negatives = negatives.copy()
        random.shuffle(shuffled_negatives)

        selected_negative_rows = set()
        running_negative_duration = 0.0
        for segment in shuffled_negatives:
            selected_negative_rows.add(segment["row_idx"])
            running_negative_duration += segment["duration_sec"]
            if running_negative_duration >= target_negative_duration:
                break

    selected_segments = []
    for segment in segments:
        if segment["label"] == POS_LABEL or segment["row_idx"] in selected_negative_rows:
            selected_segments.append({**segment, "aug_mode": None})

    summarize_segments("Down-sampled distribution", selected_segments)
    return selected_segments


def parse_args():
    project_root = CURRENT_DIR.parent
    parser = argparse.ArgumentParser(description="Down-sample no-behavior time for CQ_2 and CQ_3.")
    parser.add_argument("--input_dir", type=Path, default=project_root / "preprocess_dataset")
    parser.add_argument("--output_dir", type=Path, default=project_root / "down_sample_dataset")
    parser.add_argument("--target_negative_ratio", type=float, default=DEFAULT_TARGET_NEGATIVE_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing down-sampled dataset to: {args.output_dir}")

    for basename in KEEP_UNCHANGED_BASENAMES:
        copy_unchanged_pair(basename, args.input_dir, args.output_dir)
        print(f"Copied unchanged pair: {basename}")

    for basename in TRAIN_BASENAMES:
        print(f"\n=== Building {basename} ===")
        segments = load_segments(args.input_dir / f"{basename}.csv")
        selected_segments = choose_downsampled_segments(segments, args.target_negative_ratio)
        rebuild_pair(basename, args.input_dir, args.output_dir, selected_segments)

    print("\nFinal CSV summaries:")
    for basename in TRAIN_BASENAMES + KEEP_UNCHANGED_BASENAMES:
        compute_csv_class_percentages(args.output_dir / f"{basename}.csv")


if __name__ == "__main__":
    main()
