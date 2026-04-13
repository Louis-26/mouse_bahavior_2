import argparse
import csv
import glob
import os


def parse_timestamp(ts):
    """Parse HH:MM:SS.mmm to seconds."""
    parts = ts.strip().split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def compute_proportion(dataset_folder):
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "CQ_*.csv")))
    if not csv_files:
        print(f"No CQ_*.csv files found in {dataset_folder}")
        return

    overall_durations = {}
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        durations = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row["Start"].strip() or not row["End"].strip():
                    continue
                start = parse_timestamp(row["Start"])
                end = parse_timestamp(row["End"])
                label = row["Notes"]
                durations[label] = durations.get(label, 0.0) + (end - start)

        total = sum(durations.values())
        print(f"{filename}:")
        for label, dur in sorted(durations.items()):
            pct = dur / total * 100 if total > 0 else 0
            print(f"  {label}: {dur:.3f}s ({pct:.2f}%)")
        print(f"  total length: {total:.3f}s")
        print()

        for label, dur in durations.items():
            overall_durations[label] = overall_durations.get(label, 0.0) + dur

    overall_total = sum(overall_durations.values())
    print("overall time proportion:")
    for label, dur in sorted(overall_durations.items()):
        pct = dur / overall_total * 100 if overall_total > 0 else 0
        print(f"  {label}: {dur:.3f}s ({pct:.2f}%)")
    print(f"  total length: {overall_total:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute behavior proportions for CQ CSV files.")
    parser.add_argument("--dataset_folder", required=True, help="Path to folder containing CQ_2.csv, CQ_3.csv, CQ_4.csv")
    args = parser.parse_args()
    compute_proportion(args.dataset_folder)
