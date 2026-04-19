"""
Downsample 'no behavior' segments to achieve a 50/50 time ratio with 'scratching'.

- Automatically repairs corrupted start timestamps (uses previous row's end).
- Processes only CQ_2.csv and CQ_3.csv from the input directory.
- Cuts matching .mp4 videos to remove dropped segments, concatenates kept ones.
- Recalculates CSV timestamps to match the new concatenated video.
- Copies all other files directly to the output directory.

Requirements: ffmpeg

Usage:
  python downsample.py --input_dir ./raw --output_dir ./processed
"""

import argparse
import random
import shutil
import subprocess
import tempfile
import os

random.seed(42)

TARGET_FILES = {"CQ_2.csv", "CQ_3.csv"}


def to_seconds(ts: str) -> float:
    h, m, s = ts.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def to_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    sec %= 3600
    m = int(sec // 60)
    sec %= 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def parse_csv(filepath):
    """Parse CSV and repair corrupted start timestamps."""
    raw_rows = []
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3 or not parts[1].strip():
                continue
            try:
                start = to_seconds(parts[0].strip())
                end = to_seconds(parts[1].strip())
                label = parts[2].strip()
                raw_rows.append((start, end, label))
            except Exception:
                continue

    # Repair: if start deviates from previous end by > 1s, use previous end
    rows = []
    prev_end = 0.0
    for start, end, label in raw_rows:
        if abs(start - prev_end) > 1.0:
            start = prev_end
        dur = end - start
        if dur < 0:
            prev_end = max(prev_end, end)
            continue
        rows.append({
            'start': start,
            'end': end,
            'duration': dur,
            'label': label,
            'start_str': to_timestamp(start),
            'end_str': to_timestamp(end),
        })
        prev_end = end

    return header, rows


def cut_and_concat_video(video_path, kept_rows, output_video_path, tmpdir):
    """Cut kept segments from the video and concatenate them."""
    # Merge consecutive time ranges for fewer ffmpeg cuts
    segments = []
    for r in kept_rows:
        if segments and abs(r['start'] - segments[-1][1]) < 0.01:
            segments[-1] = (segments[-1][0], r['end'])
        else:
            segments.append((r['start'], r['end']))

    print(f"  Video: cutting {len(segments)} segments from {os.path.basename(video_path)}")

    segment_files = []
    for i, (seg_start, seg_end) in enumerate(segments):
        seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
        segment_files.append(seg_path)
        duration = seg_end - seg_start
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{seg_start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            seg_path
        ]
        subprocess.run(cmd, check=True)

    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list, 'w') as f:
        for sp in segment_files:
            f.write(f"file '{sp}'\n")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        output_video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"  Video saved: {output_video_path}")


def recalculate_timestamps(kept_rows):
    """Shift timestamps so kept rows are contiguous starting at 0."""
    new_rows = []
    cursor = 0.0
    for r in kept_rows:
        dur = r['duration']
        new_row = dict(r)
        new_row['start'] = cursor
        new_row['end'] = cursor + dur
        new_row['start_str'] = to_timestamp(cursor)
        new_row['end_str'] = to_timestamp(cursor + dur)
        new_rows.append(new_row)
        cursor += dur
    return new_rows


def downsample(filepath, outpath, input_dir, output_dir):
    header, rows = parse_csv(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]

    is_scratch = lambda l: 'scract' in l.lower() or 'scratch' in l.lower()
    is_nobehav = lambda l: 'no behavior' in l.lower()

    scratch_rows = [r for r in rows if is_scratch(r['label'])]
    nobehav_rows = [r for r in rows if is_nobehav(r['label'])]

    total_scratch = sum(r['duration'] for r in scratch_rows)
    total_nobehav = sum(r['duration'] for r in nobehav_rows)
    original_total = total_scratch + total_nobehav
    target_nobehav = total_scratch

    print(f"\n--- {os.path.basename(filepath)} ---")
    print(f"  Original length: {original_total:.2f}s ({original_total/60:.1f} min)")
    print(f"  Scratching:    {len(scratch_rows):>5} rows  |  {total_scratch:.2f}s ({total_scratch/60:.1f} min) | {total_scratch/original_total*100:.1f}%")
    print(f"  No behavior:   {len(nobehav_rows):>5} rows  |  {total_nobehav:.2f}s ({total_nobehav/60:.1f} min) | {total_nobehav/original_total*100:.1f}%")

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
    truncate_info = None

    for idx in indices:
        dur = nobehav_rows[idx]['duration']
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

    # Build kept no-behavior rows in chronological order
    kept_nobehav = []
    for i in sorted(kept_indices):
        r = dict(nobehav_rows[i])
        if truncate_info and truncate_info[0] == i:
            r['end'] = r['start'] + truncate_info[1]
            r['end_str'] = to_timestamp(r['end'])
            r['duration'] = truncate_info[1]
        kept_nobehav.append(r)

    # Merge and sort by original start time
    all_kept = scratch_rows + kept_nobehav
    all_kept.sort(key=lambda r: r['start'])

    final_scratch = sum(r['duration'] for r in all_kept if is_scratch(r['label']))
    final_nobehav = sum(r['duration'] for r in all_kept if is_nobehav(r['label']))
    final_total = final_scratch + final_nobehav

    print(f"  Kept no-behavior: {len(kept_nobehav)} / {len(nobehav_rows)} rows")
    print(f"  New length:    {final_total:.2f}s ({final_total/60:.1f} min)")
    print(f"  Reduction:     {original_total/60:.1f} min -> {final_total/60:.1f} min (saved {(original_total-final_total)/60:.1f} min)")
    print(f"  Final ratio:   {final_scratch/final_total*100:.1f}% / {final_nobehav/final_total*100:.1f}%")

    # --- Video cutting ---
    video_src = os.path.join(input_dir, f"{basename}.mp4")
    if os.path.exists(video_src):
        video_dst = os.path.join(output_dir, f"{basename}.mp4")
        with tempfile.TemporaryDirectory() as tmpdir:
            cut_and_concat_video(video_src, all_kept, video_dst, tmpdir)
    else:
        print(f"  Warning: video not found: {video_src} (skipping video cut)")

    # --- Recalculate timestamps for concatenated video ---
    all_kept_new = recalculate_timestamps(all_kept)

    with open(outpath, 'w') as f:
        f.write(header + '\n')
        for r in all_kept_new:
            f.write(f"{r['start_str']},{r['end_str']},{r['label']}\n")

    print(f"  CSV saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample 'no behavior' to 50/50 with scratching. "
                    "Repairs corrupted timestamps, cuts and concatenates .mp4 videos. "
                    "Processes only CQ_2 and CQ_3; copies other files to output dir."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with raw CSVs and .mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Directory for downsampled outputs")
    args = parser.parse_args()

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


if __name__ == '__main__':
    main()