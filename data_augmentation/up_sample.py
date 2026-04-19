#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
	sys.path.insert(0, str(CURRENT_DIR))

from sampling_common import AUGMENTATION_MODES, NEG_LABEL, POS_LABEL, compute_csv_class_percentages, copy_unchanged_pair, load_segments, rebuild_pair, summarize_segments


TRAIN_BASENAMES = ("CQ_2", "CQ_3")
KEEP_UNCHANGED_BASENAMES = ("CQ_4",)
DEFAULT_TARGET_POSITIVE_RATIO = 0.47
DEFAULT_MAX_AUG_PER_SEGMENT = 8


def choose_upsampled_segments(segments, target_positive_ratio, enable_augmentation, max_aug_per_segment):
	positives = [segment for segment in segments if segment["label"] == POS_LABEL]
	negatives = [segment for segment in segments if segment["label"] == NEG_LABEL]

	if not positives:
		raise RuntimeError("No scratching segments found for up-sampling.")
	if not negatives:
		raise RuntimeError("No no-behavior segments found for up-sampling.")

	summarize_segments("Original distribution", segments)

	positive_duration = sum(segment["duration_sec"] for segment in positives)
	negative_duration = sum(segment["duration_sec"] for segment in negatives)
	target_positive_duration = negative_duration * target_positive_ratio / (1.0 - target_positive_ratio)
	extra_positive_duration = max(0.0, target_positive_duration - positive_duration)

	duplicate_counts = {segment["row_idx"]: 0 for segment in positives}
	duplicated_segments = defaultdict(list)

	while extra_positive_duration > 0:
		progress_made = False
		for segment in positives:
			if duplicate_counts[segment["row_idx"]] >= max_aug_per_segment:
				continue

			duplicated_segments[segment["row_idx"]].append(
				{
					**segment,
					"aug_mode": random.choice(AUGMENTATION_MODES) if enable_augmentation else None,
				}
			)
			duplicate_counts[segment["row_idx"]] += 1
			extra_positive_duration -= segment["duration_sec"]
			progress_made = True

			if extra_positive_duration <= 0:
				break

		if not progress_made:
			break

	selected_segments = []
	for segment in segments:
		selected_segments.append({**segment, "aug_mode": None})
		selected_segments.extend(duplicated_segments.get(segment["row_idx"], []))

	summarize_segments("Up-sampled distribution", selected_segments)
	return selected_segments


def parse_args():
	project_root = CURRENT_DIR.parent
	parser = argparse.ArgumentParser(description="Up-sample scratching time for CQ_2 and CQ_3.")
	parser.add_argument("--input_dir", type=Path, default=project_root / "preprocess_dataset")
	parser.add_argument("--output_dir", type=Path, default=project_root / "up_sample_dataset")
	parser.add_argument("--target_positive_ratio", type=float, default=DEFAULT_TARGET_POSITIVE_RATIO)
	parser.add_argument("--max_aug_per_segment", type=int, default=DEFAULT_MAX_AUG_PER_SEGMENT)
	parser.add_argument("--disable_augmentation", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def main():
	args = parse_args()
	random.seed(args.seed)
	np.random.seed(args.seed)

	args.output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Writing up-sampled dataset to: {args.output_dir}")

	for basename in KEEP_UNCHANGED_BASENAMES:
		copy_unchanged_pair(basename, args.input_dir, args.output_dir)
		print(f"Copied unchanged pair: {basename}")

	for basename in TRAIN_BASENAMES:
		print(f"\n=== Building {basename} ===")
		segments = load_segments(args.input_dir / f"{basename}.csv")
		selected_segments = choose_upsampled_segments(
			segments,
			target_positive_ratio=args.target_positive_ratio,
			enable_augmentation=not args.disable_augmentation,
			max_aug_per_segment=args.max_aug_per_segment,
		)
		rebuild_pair(basename, args.input_dir, args.output_dir, selected_segments)

	print("\nFinal CSV summaries:")
	for basename in TRAIN_BASENAMES + KEEP_UNCHANGED_BASENAMES:
		compute_csv_class_percentages(args.output_dir / f"{basename}.csv")


if __name__ == "__main__":
	main()
