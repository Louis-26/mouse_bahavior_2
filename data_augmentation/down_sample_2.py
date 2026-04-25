"""
Downsample 'no behavior' segments to achieve a 33.3/66.7 time ratio with 'scratching'.

- Automatically repairs corrupted start timestamps (uses previous row's end),
  but preserves a legitimate initial offset on the first row.
- Processes only CQ_2.csv and CQ_3.csv from the input directory.
- Cuts matching .mp4 videos with frame-accurate re-encoding, then concatenates.
- Recalculates CSV timestamps to match the new concatenated video.
- Preserves rows with labels other than 'scratch' / 'no behavior' (they are kept
  and their video portions are included in the output).
- Copies all other files directly to the output directory.

Requirements: ffmpeg (with libx264 and aac) on PATH.

Usage:
  python downsample.py --input_dir ./raw --output_dir ./processed
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import tempfile

random.seed(42)

TARGET_FILES = {"CQ_2.csv", "CQ_3.csv"}
SCRATCH_KEYS = ("scratch", "scract")   # tolerate typo
NOBEHAV_KEY = "no behavior"


def to_seconds(ts: str) -> float:
    h, m, s = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def to_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    sec %= 3600
    m = int(sec // 60)
    sec %= 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def is_scratch(label: str) -> bool:
    l = label.lower()
    return any(k in l for k in SCRATCH_KEYS)


def is_nobehav(label: str) -> bool:
    return NOBEHAV_KEY in label.lower()


def parse_csv(filepath):
    """Parse CSV (robust to quoted fields) and repair corrupted start timestamps."""
    raw_rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return "", []
        for parts in reader:
            if len(parts) < 3 or not parts[1].strip():
                continue
            try:
                start = to_seconds(parts[0])
                end = to_seconds(parts[1])
                label = parts[2].strip()
                raw_rows.append((start, end, label))
            except Exception:
                continue

    rows = []
    prev_end = None  # seeded by first row to preserve a legitimate initial offset
    for start, end, label in raw_rows:
        if prev_end is None:
            prev_end = start  # trust the first row's start
        if abs(start - prev_end) > 1.0:
            start = prev_end
        dur = end - start
        if dur < 0:
            prev_end = max(prev_end, end)
            continue
        rows.append({
            "start": start,
            "end": end,
            "duration": dur,
            "label": label,
            "start_str": to_timestamp(start),
            "end_str": to_timestamp(end),
        })
        prev_end = end

    header_str = ",".join(header) if header else ""
    return header_str, rows


def cut_and_concat_video(video_path, kept_rows, output_video_path, tmpdir):
    """Cut kept segments with frame-accurate re-encoding and concatenate them."""
    # Merge consecutive ranges so ffmpeg makes fewer cuts.
    segments = []
    for r in kept_rows:
        if segments and abs(r["start"] - segments[-1][1]) < 0.05:
            segments[-1] = (segments[-1][0], r["end"])
        else:
            segments.append((r["start"], r["end"]))

    print(f"  Video: cutting {len(segments)} segments from {os.path.basename(video_path)}")

    segment_files = []
    for i, (seg_start, seg_end) in enumerate(segments):
        seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
        segment_files.append(seg_path)
        duration = seg_end - seg_start
        # Accurate seek: -ss AFTER -i, and re-encode so cuts are frame-accurate.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-ss", f"{seg_start:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            seg_path,
        ]
        subprocess.run(cmd, check=True)

    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list, "w") as f:
        for sp in segment_files:
            # ffmpeg concat demuxer requires escaped single-quoted paths
            f.write(f"file '{sp}'\n")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        "-movflags", "+faststart",
        output_video_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"  Video saved: {output_video_path}")


def recalculate_timestamps(kept_rows):
    """Shift timestamps so kept rows are contiguous starting at 0."""
    new_rows = []
    cursor = 0.0
    for r in kept_rows:
        dur = r["duration"]
        nr = dict(r)
        nr["start"] = cursor
        nr["end"] = cursor + dur
        nr["start_str"] = to_timestamp(cursor)
        nr["end_str"] = to_timestamp(cursor + dur)
        new_rows.append(nr)
        cursor += dur
    return new_rows


def downsample(filepath, outpath, input_dir, output_dir):
    header, rows = parse_csv(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]

    scratch_rows = [r for r in rows if is_scratch(r["label"])]
    nobehav_rows = [r for r in rows if is_nobehav(r["label"])]
    other_rows = [r for r in rows if not is_scratch(r["label"]) and not is_nobehav(r["label"])]

    total_scratch = sum(r["duration"] for r in scratch_rows)
    total_nobehav = sum(r["duration"] for r in nobehav_rows)
    total_other = sum(r["duration"] for r in other_rows)
    original_total = total_scratch + total_nobehav + total_other
    target_nobehav = total_scratch / 2  # scratch : nobehav = 2 : 1 (66.7 / 33.3)

    print(f"\n--- {os.path.basename(filepath)} ---")
    print(f"  Original length: {original_total:.2f}s ({original_total/60:.1f} min)")
    print(f"  Scratching:  {len(scratch_rows):>5} rows | {total_scratch:8.2f}s")
    print(f"  No behavior: {len(nobehav_rows):>5} rows | {total_nobehav:8.2f}s")
    if other_rows:
        print(f"  Other:       {len(other_rows):>5} rows | {total_other:8.2f}s (kept as-is)")

    if total_nobehav <= target_nobehav:
        print("  No downsampling needed. Copying as-is.")
        shutil.copy2(filepath, outpath)
        video_src = os.path.join(input_dir, f"{basename}.mp4")
        if os.path.exists(video_src):
            shutil.copy2(video_src, os.path.join(output_dir, f"{basename}.mp4"))
        return

    # Randomly sample no-behavior rows to match target duration
    indices = list(range(len(nobehav_rows)))
    random.shuffle(indices)

    kept_indices = set()
    accumulated = 0.0
    truncate_info = None  # (nobehav_index, truncated_duration)

    for idx in indices:
        dur = nobehav_rows[idx]["duration"]
        if accumulated + dur <= target_nobehav:
            kept_indices.add(idx)
            accumulated += dur
        else:
            remaining = target_nobehav - accumulated
            if remaining > 0.001:
                kept_indices.add(idx)
                truncate_info = (idx, remaining)
                accumulated += remaining
            break

    kept_nobehav = []
    for i in sorted(kept_indices):
        r = dict(nobehav_rows[i])
        if truncate_info and truncate_info[0] == i:
            r["end"] = r["start"] + truncate_info[1]
            r["end_str"] = to_timestamp(r["end"])
            r["duration"] = truncate_info[1]
        kept_nobehav.append(r)

    # Keep scratching + sampled no-behavior + all other-label rows, in original order
    all_kept = scratch_rows + kept_nobehav + other_rows
    all_kept.sort(key=lambda r: r["start"])

    final_scratch = sum(r["duration"] for r in all_kept if is_scratch(r["label"]))
    final_nobehav = sum(r["duration"] for r in all_kept if is_nobehav(r["label"]))
    final_other = sum(r["duration"] for r in all_kept if not is_scratch(r["label"]) and not is_nobehav(r["label"]))
    final_total = final_scratch + final_nobehav + final_other

    print(f"  Kept no-behavior: {len(kept_nobehav)} / {len(nobehav_rows)} rows")
    print(f"  New length: {final_total:.2f}s ({final_total/60:.1f} min)")
    print(f"  Reduction:  {original_total/60:.1f} min -> {final_total/60:.1f} min "
          f"(saved {(original_total-final_total)/60:.1f} min)")
    if final_total > 0:
        print(f"  Final shares: scratch {final_scratch/final_total*100:.1f}% | "
              f"no-behavior {final_nobehav/final_total*100:.1f}% | "
              f"other {final_other/final_total*100:.1f}%")

    # --- Video cutting ---
    video_src = os.path.join(input_dir, f"{basename}.mp4")
    if os.path.exists(video_src):
        video_dst = os.path.join(output_dir, f"{basename}.mp4")
        with tempfile.TemporaryDirectory() as tmpdir:
            cut_and_concat_video(video_src, all_kept, video_dst, tmpdir)
    else:
        print(f"  Warning: video not found: {video_src} (skipping video cut)")

    # --- Recalculate CSV timestamps to match concatenated video ---
    all_kept_new = recalculate_timestamps(all_kept)

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header.split(","))
        for r in all_kept_new:
            writer.writerow([r["start_str"], r["end_str"], r["label"]])

    print(f"  CSV saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample 'no behavior' to a 33.3/66.7 ratio with scratching. "
                    "Repairs corrupted timestamps, cuts and concatenates .mp4 videos "
                    "with frame-accurate re-encoding. Processes only CQ_2 and CQ_3; "
                    "copies other files to output dir."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with raw CSVs and .mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Directory for downsampled outputs")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH. Install ffmpeg before running.")
        return
    if not os.path.isdir(args.input_dir):
        print(f"Error: input directory not found: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Process target CSVs
    for fname in sorted(os.listdir(args.input_dir)):
        src = os.path.join(args.input_dir, fname)
        if not os.path.isfile(src):
            continue
        if fname in TARGET_FILES:
            dst = os.path.join(args.output_dir, fname)
            downsample(src, dst, args.input_dir, args.output_dir)

    # Copy non-target files directly to output_dir
    handled_videos = {os.path.splitext(f)[0] + ".mp4" for f in TARGET_FILES}
    for fname in sorted(os.listdir(args.input_dir)):
        src = os.path.join(args.input_dir, fname)
        if not os.path.isfile(src):
            continue
        if fname in TARGET_FILES or fname in handled_videos:
            continue
        dst = os.path.join(args.output_dir, fname)
        shutil.copy2(src, dst)
        print(f"  Copied: {fname}")


if __name__ == "__main__":
    main()