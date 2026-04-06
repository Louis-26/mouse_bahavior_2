from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


POS_LABEL = "scracthing"
NEG_LABEL = "no behavior"
AUGMENTATION_MODES = ("brightness", "noise", "blur", "flip")


def parse_time_str(value):
    if pd.isna(value):
        return 0.0

    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)

    return float(text)


def to_time_str(seconds):
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))

    if milliseconds == 1000:
        milliseconds = 0
        whole_seconds += 1
    if whole_seconds == 60:
        whole_seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def load_segments(csv_path, min_segment_sec=0.15, pos_label=POS_LABEL, neg_label=NEG_LABEL):
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    for required_column in ("Start", "End", "Notes"):
        if required_column not in df.columns:
            raise ValueError(f"{csv_path} is missing required column '{required_column}'")

    df = df.dropna(subset=["Start", "End", "Notes"])
    df = df[df["Start"].astype(str).str.strip() != "Start"]
    df = df[df["End"].astype(str).str.strip() != "End"]
    df = df[df["Notes"].astype(str).str.strip() != "Notes"]
    df = df[df["Start"].astype(str).str.contains(r":", na=False)]
    df = df[df["End"].astype(str).str.contains(r":", na=False)]

    segments = []
    for row_idx, row in df.iterrows():
        start_sec = parse_time_str(row["Start"])
        end_sec = parse_time_str(row["End"])
        label = str(row["Notes"]).strip()

        if label not in {pos_label, neg_label}:
            continue
        if end_sec <= start_sec:
            continue

        duration_sec = end_sec - start_sec
        if duration_sec < min_segment_sec:
            continue

        segments.append(
            {
                "row_idx": int(row_idx),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "label": label,
            }
        )

    if not segments:
        raise RuntimeError(f"No usable segments found in {csv_path}")

    return segments


def build_segment_augmentor(mode):
    if mode == "brightness":
        alpha = 1.0 + random.uniform(-0.08, 0.08)
        beta = random.uniform(-12, 12)

        def apply(frame):
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        return apply

    if mode == "noise":
        def apply(frame):
            noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
            return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return apply

    if mode == "blur":
        def apply(frame):
            return cv2.GaussianBlur(frame, (3, 3), 0)

        return apply

    if mode == "flip":
        def apply(frame):
            return cv2.flip(frame, 1)

        return apply

    def apply(frame):
        return frame

    return apply


def copy_unchanged_pair(basename, input_dir, output_dir):
    src_video = Path(input_dir) / f"{basename}.mp4"
    src_csv = Path(input_dir) / f"{basename}.csv"
    dst_video = Path(output_dir) / f"{basename}.mp4"
    dst_csv = Path(output_dir) / f"{basename}.csv"

    if not src_video.exists():
        raise FileNotFoundError(f"Missing file: {src_video}")
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing file: {src_csv}")

    shutil.copy2(src_video, dst_video)
    shutil.copy2(src_csv, dst_csv)


def rebuild_csv_rows(segment_rows):
    rows = []
    cursor = 0.0
    for segment in segment_rows:
        start = cursor
        end = cursor + segment["duration_sec"]
        rows.append(
            {
                "Start": to_time_str(start),
                "End": to_time_str(end),
                "Notes": segment["label"],
            }
        )
        cursor = end
    return pd.DataFrame(rows)


def summarize_segments(name, segments, pos_label=POS_LABEL, neg_label=NEG_LABEL):
    pos_duration = sum(segment["duration_sec"] for segment in segments if segment["label"] == pos_label)
    neg_duration = sum(segment["duration_sec"] for segment in segments if segment["label"] == neg_label)
    total_duration = pos_duration + neg_duration
    pos_ratio = pos_duration / total_duration if total_duration else 0.0
    neg_ratio = neg_duration / total_duration if total_duration else 0.0

    print(f"{name}:")
    print(f"  {neg_label}: {neg_duration:.3f}s ({neg_ratio * 100:.2f}%)")
    print(f"  {pos_label}: {pos_duration:.3f}s ({pos_ratio * 100:.2f}%)")
    print(f"  total length: {total_duration:.3f}s")


def compute_csv_class_percentages(csv_path, pos_label=POS_LABEL, neg_label=NEG_LABEL):
    segments = load_segments(csv_path, pos_label=pos_label, neg_label=neg_label)
    summarize_segments(csv_path.name, segments, pos_label=pos_label, neg_label=neg_label)


def rebuild_pair(basename, input_dir, output_dir, selected_segments):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = input_dir / f"{basename}.mp4"
    output_video_path = output_dir / f"{basename}.mp4"
    output_csv_path = output_dir / f"{basename}.csv"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_video_path}")

    csv_rows = []
    total_written_frames = 0

    try:
        for segment in selected_segments:
            start_frame = max(0, int(round(segment["start_sec"] * fps)))
            end_frame = min(total_frames, int(round(segment["end_sec"] * fps)))

            if end_frame <= start_frame:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            augment_fn = build_segment_augmentor(segment.get("aug_mode"))
            written_frames = 0

            for _ in range(start_frame, end_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(augment_fn(frame))
                written_frames += 1

            if written_frames == 0:
                continue

            csv_rows.append(
                {
                    "duration_sec": written_frames / fps,
                    "label": segment["label"],
                }
            )
            total_written_frames += written_frames
    finally:
        cap.release()
        writer.release()

    if total_written_frames == 0:
        raise RuntimeError(f"No frames were written for {basename}")

    rebuild_csv_rows(csv_rows).to_csv(output_csv_path, index=False)
    compute_csv_class_percentages(output_csv_path)
