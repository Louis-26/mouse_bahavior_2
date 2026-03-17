"""
Convert prediction results from JSON format to CSV format.

This script takes prediction results in JSON format (e.g., statistics.json) and converts
the segments to CSV format with start time, end time, and behavior label.

Input example: /data/zhaozhenghao/Projects/Mouse/results/UMich_CQ/video_inference/CQ_4/statistics.json
Output example: The CSV file will be saved in the same folder as the input JSON file

CSV format:
- Start: HH:MM:SS.mmm
- End: HH:MM:SS.mmm
- Notes: behavior label
"""

import json
import csv
import argparse
from pathlib import Path
import pandas as pd


def seconds_to_timestamp(seconds):
    """
    Convert seconds (float) to timestamp format HH:MM:SS.mmm
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        String in format HH:MM:SS.mmm
    """
    # Round to milliseconds (3 decimal places) to avoid floating point precision issues
    seconds = round(seconds, 3)
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def json_to_csv(json_path, output_path=None):
    """
    Convert prediction results JSON to CSV and XLSX format.
    
    Args:
        json_path: Path to the input JSON file (e.g., statistics.json)
        output_path: Path to the output CSV file. If None, saves in the same folder
                     as the input with name 'predictions.csv'
    """
    json_path = Path(json_path)
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Determine output path
    if output_path is None:
        csv_output_path = json_path.parent / 'predictions.csv'
        xlsx_output_path = json_path.parent / 'predictions.xlsx'
    else:
        csv_output_path = Path(output_path)
        xlsx_output_path = csv_output_path.with_suffix('.xlsx')
    
    # Extract segments
    segments = data.get('segments', [])
    
    if not segments:
        print(f"Warning: No segments found in {json_path}")
        return
    
    # Prepare data for both CSV and Excel
    rows = []
    
    for segment in segments:
        start_time = segment['start_time']
        end_time = segment['end_time']
        label = segment['label']
        if label == "scracthing":
            label = "scratching"
        
        start_timestamp = seconds_to_timestamp(start_time)
        end_timestamp = seconds_to_timestamp(end_time)
        duration = end_time - start_time
        duration_timestamp = seconds_to_timestamp(duration)
        
        rows.append([start_timestamp, end_timestamp, duration_timestamp, label])
    
    # Write to CSV
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Start', 'End', 'Duration', 'Notes'])
        writer.writerows(rows)
    
    # Write to Excel
    df = pd.DataFrame(rows, columns=['Start', 'End', 'Duration', 'Notes'])
    df.to_excel(xlsx_output_path, index=False, engine='openpyxl')
    
    print(f"Successfully converted {len(segments)} segments to:")
    print(f"  - CSV: {csv_output_path}")
    print(f"  - Excel: {xlsx_output_path}")
    return csv_output_path, xlsx_output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert prediction results JSON to CSV format'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        help='Path to the input JSON file (e.g., statistics.json)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output CSV file (default: predictions.csv in the same folder)'
    )
    
    args = parser.parse_args()
    
    json_to_csv(args.json_path, args.output)


if __name__ == '__main__':
    main()