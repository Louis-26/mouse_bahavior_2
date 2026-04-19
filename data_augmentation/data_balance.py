import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
INPUT_DIR = "../preprocess_dataset"
OUTPUT_DIR = "../balanced_dataset"

# Rebuild these
TRAIN_BASENAMES = ["CQ_2", "CQ_3"]

# Copy these unchanged
KEEP_UNCHANGED_BASENAMES = ["CQ_4"]

# Preserve exact labels from your CSV
POS_LABEL = "scracthing"
NEG_LABEL = "no behavior"

# Target class balance by total labeled duration in each rebuilt file
TARGET_NEG_RATIO = 0.53   # 50-55% no behavior

# Skip very tiny segments
MIN_SEGMENT_SEC = 0.15

# Augmentation for duplicated positive segments
ENABLE_AUGMENTATION = True
MAX_POS_AUG_PER_SEGMENT = 8

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================================
# TIME HELPERS
# =========================================================
def parse_time_str(t):
    if pd.isna(t):
        return 0.0

    t = str(t).strip()

    if ":" in t:
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)

    try:
        return float(t)
    except Exception:
        print(f"Warning: cannot parse time '{t}', using 0.0")
        return 0.0


def to_time_str(sec):
    if sec < 0:
        sec = 0.0

    hours = int(sec // 3600)
    sec -= hours * 3600
    minutes = int(sec // 60)
    sec -= minutes * 60
    whole = int(sec)
    millis = int(round((sec - whole) * 1000))

    if millis == 1000:
        millis = 0
        whole += 1
    if whole == 60:
        whole = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1

    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"


# =========================================================
# VIDEO HELPERS
# =========================================================
def read_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return fps, total_frames, width, height


# =========================================================
# AUGMENTATION
# =========================================================
def build_segment_augmentor(mode):
    if mode == "brightness":
        alpha = 1.0 + random.uniform(-0.08, 0.08)
        beta = random.uniform(-12, 12)

        def fn(frame):
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return fn

    elif mode == "noise":
        def fn(frame):
            noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
            return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return fn

    elif mode == "blur":
        def fn(frame):
            return cv2.GaussianBlur(frame, (3, 3), 0)
        return fn

    elif mode == "flip":
        def fn(frame):
            return cv2.flip(frame, 1)
        return fn

    else:
        def fn(frame):
            return frame
        return fn


def write_segment_to_writer(video_path, writer, start_sec, end_sec, aug_mode=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, int(round(end_sec * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    augment_fn = build_segment_augmentor(aug_mode)
    written = 0

    for _ in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break

        frame = augment_fn(frame)
        writer.write(frame)
        written += 1

    cap.release()
    return written, fps


# =========================================================
# CSV HELPERS
# =========================================================
def load_segments(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Start", "End", "Notes"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing required column '{col}'")

    df = df.dropna(subset=["Start", "End", "Notes"])

    # Remove accidental repeated header rows inside the data
    df = df[df["Start"].astype(str).str.strip() != "Start"]
    df = df[df["End"].astype(str).str.strip() != "End"]
    df = df[df["Notes"].astype(str).str.strip() != "Notes"]

    # Keep only rows that look like time strings
    df = df[df["Start"].astype(str).str.contains(r":", na=False)]
    df = df[df["End"].astype(str).str.contains(r":", na=False)]

    segments = []
    for i, row in df.iterrows():
        start_sec = parse_time_str(row["Start"])
        end_sec = parse_time_str(row["End"])
        label = str(row["Notes"]).strip()

        if end_sec <= start_sec:
            continue

        duration_sec = end_sec - start_sec
        if duration_sec < MIN_SEGMENT_SEC:
            continue

        if label not in [POS_LABEL, NEG_LABEL]:
            continue

        segments.append({
            "row_idx": i,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "label": label,
        })

    return segments


def rebuild_csv_rows(selected_segments):
    rows = []
    cursor = 0.0

    for seg in selected_segments:
        start = cursor
        end = cursor + seg["duration_sec"]
        rows.append({
            "Start": to_time_str(start),
            "End": to_time_str(end),
            "Notes": seg["label"]
        })
        cursor = end

    return pd.DataFrame(rows)


def compute_csv_class_percentages(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    durations = {}
    total_duration = 0.0

    for _, row in df.iterrows():
        start_sec = parse_time_str(row["Start"])
        end_sec = parse_time_str(row["End"])
        label = str(row["Notes"]).strip()

        duration = max(0.0, end_sec - start_sec)
        durations[label] = durations.get(label, 0.0) + duration
        total_duration += duration

    print(f"\nClass percentages for {csv_path.name}:")
    for label, dur in sorted(durations.items()):
        pct = 100.0 * dur / total_duration if total_duration > 0 else 0.0
        print(f"  {label}: {dur:.3f}s ({pct:.2f}%)")

    return durations, total_duration


# =========================================================
# BALANCING
# =========================================================
def choose_balanced_segments(segments):
    positives = [s for s in segments if s["label"] == POS_LABEL]
    negatives = [s for s in segments if s["label"] == NEG_LABEL]

    if len(positives) == 0:
        raise RuntimeError("No positive segments found.")
    if len(negatives) == 0:
        raise RuntimeError("No negative segments found.")

    pos_total_dur = sum(s["duration_sec"] for s in positives)
    neg_total_dur = sum(s["duration_sec"] for s in negatives)

    print(f"Original duration totals:")
    print(f"  {NEG_LABEL}: {neg_total_dur:.3f}s")
    print(f"  {POS_LABEL}: {pos_total_dur:.3f}s")
    print(f"  original negative ratio: {neg_total_dur / (neg_total_dur + pos_total_dur):.4f}")

    # Start with all positives
    selected_pos = []
    pos_aug_count = {}

    for seg in positives:
        item = dict(seg)
        item["augmented"] = False
        item["aug_mode"] = None
        selected_pos.append(item)
        pos_aug_count[seg["row_idx"]] = 0

    selected_pos_dur = sum(s["duration_sec"] for s in selected_pos)

    # Target negative duration for the chosen positive duration
    # neg / (neg + pos) = TARGET_NEG_RATIO
    target_neg_dur = selected_pos_dur * TARGET_NEG_RATIO / (1.0 - TARGET_NEG_RATIO)

    # Case 1: enough negatives, downsample negatives by duration
    if neg_total_dur >= target_neg_dur:
        shuffled_neg = negatives.copy()
        random.shuffle(shuffled_neg)

        selected_neg = []
        running_neg_dur = 0.0

        for seg in shuffled_neg:
            selected_neg.append(dict(seg, augmented=False, aug_mode=None))
            running_neg_dur += seg["duration_sec"]
            if running_neg_dur >= target_neg_dur:
                break

    # Case 2: not enough negatives, keep all negatives and upsample positives
    else:
        selected_neg = [dict(seg, augmented=False, aug_mode=None) for seg in negatives]
        selected_neg_dur = sum(s["duration_sec"] for s in selected_neg)

        target_pos_dur = selected_neg_dur * (1.0 - TARGET_NEG_RATIO) / TARGET_NEG_RATIO

        aug_modes = ["brightness", "noise", "blur", "flip"]

        while selected_pos_dur < target_pos_dur:
            base = random.choice(positives)

            if pos_aug_count[base["row_idx"]] >= MAX_POS_AUG_PER_SEGMENT:
                if all(v >= MAX_POS_AUG_PER_SEGMENT for v in pos_aug_count.values()):
                    break
                continue

            new_item = dict(base)
            new_item["augmented"] = ENABLE_AUGMENTATION
            new_item["aug_mode"] = random.choice(aug_modes) if ENABLE_AUGMENTATION else None

            selected_pos.append(new_item)
            selected_pos_dur += base["duration_sec"]
            pos_aug_count[base["row_idx"]] += 1

    final_segments = selected_neg + selected_pos
    random.shuffle(final_segments)

    final_neg_dur = sum(s["duration_sec"] for s in final_segments if s["label"] == NEG_LABEL)
    final_pos_dur = sum(s["duration_sec"] for s in final_segments if s["label"] == POS_LABEL)

    print(f"Selected segments: {len(final_segments)}")
    print(f"  {NEG_LABEL}: {sum(1 for s in final_segments if s['label'] == NEG_LABEL)} segments")
    print(f"  {POS_LABEL}: {sum(1 for s in final_segments if s['label'] == POS_LABEL)} segments")
    print(f"Selected duration totals:")
    print(f"  {NEG_LABEL}: {final_neg_dur:.3f}s")
    print(f"  {POS_LABEL}: {final_pos_dur:.3f}s")
    print(f"  final negative ratio by duration: {final_neg_dur / (final_neg_dur + final_pos_dur):.4f}")

    return final_segments


# =========================================================
# REBUILD VIDEO + CSV
# =========================================================
def rebuild_one_pair(basename, input_dir, output_dir):
    video_path = Path(input_dir) / f"{basename}.mp4"
    csv_path = Path(input_dir) / f"{basename}.csv"

    out_video_path = Path(output_dir) / f"{basename}.mp4"
    out_csv_path = Path(output_dir) / f"{basename}.csv"

    print(f"\n=== Rebuilding {basename} ===")
    segments = load_segments(csv_path)
    kept = choose_balanced_segments(segments)

    fps_meta, total_frames, width, height = read_video_metadata(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_meta, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_video_path}")

    csv_rows_segments = []
    total_written_frames = 0

    try:
        for idx, seg in enumerate(kept):
            aug_mode = seg.get("aug_mode") if seg.get("augmented", False) else None

            written_frames, fps = write_segment_to_writer(
                video_path=video_path,
                writer=writer,
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                aug_mode=aug_mode
            )

            if written_frames == 0:
                continue

            duration_sec = written_frames / fps
            total_written_frames += written_frames

            csv_rows_segments.append({
                "duration_sec": duration_sec,
                "label": seg["label"]
            })

            if (idx + 1) % 20 == 0:
                print(f"  processed {idx + 1}/{len(kept)} segments")

    finally:
        writer.release()

    if total_written_frames == 0:
        raise RuntimeError(f"No frames written for {basename}")

    out_df = rebuild_csv_rows(csv_rows_segments)
    out_df.to_csv(out_csv_path, index=False)

    print(f"Saved: {out_video_path}")
    print(f"Saved: {out_csv_path}")
    print(f"Total frames written: {total_written_frames}")
    print(f"Total duration: {total_written_frames / fps_meta:.2f}s")

    compute_csv_class_percentages(out_csv_path)


# =========================================================
# MAIN
# =========================================================
def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {output_dir}")

    # Copy only files that must remain unchanged
    for basename in KEEP_UNCHANGED_BASENAMES:
        src_video = input_dir / f"{basename}.mp4"
        src_csv = input_dir / f"{basename}.csv"
        dst_video = output_dir / f"{basename}.mp4"
        dst_csv = output_dir / f"{basename}.csv"

        if not src_video.exists():
            raise FileNotFoundError(f"Missing file: {src_video}")
        if not src_csv.exists():
            raise FileNotFoundError(f"Missing file: {src_csv}")

        shutil.copy2(src_video, dst_video)
        shutil.copy2(src_csv, dst_csv)
        print(f"Copied unchanged: {basename}.mp4/.csv")

    # Rebuild train files
    for basename in TRAIN_BASENAMES:
        rebuild_one_pair(basename, input_dir, output_dir)

    print("\n" + "=" * 60)
    print("Final class percentages in balanced_dataset")
    print("=" * 60)

    for basename in TRAIN_BASENAMES + KEEP_UNCHANGED_BASENAMES:
        csv_path = output_dir / f"{basename}.csv"
        if csv_path.exists():
            compute_csv_class_percentages(csv_path)

    print("\nDone.")
    print(f"Balanced dataset written to: {output_dir}")


if __name__ == "__main__":
    main()