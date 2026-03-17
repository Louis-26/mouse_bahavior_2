#!/usr/bin/env python3
"""
Unified Label Processing Tool for Behavior Annotation Data

This script converts behavior annotation Excel files to standardized CSV format
with millisecond-precision timestamps (HH:MM:SS.mmm).

Supports multiple time formats:
- CQ2 format: "0.05.425" or range format with "--", "-", "to", "→", "~"
- CQ3/CQ4 format: "00:03.067" or "0.05.425"
- Standard format: "HH:MM:SS.mmm"

Usage:
    # Auto-detect format
    python label_process.py --input itch_video_CQ_2.xlsx --output CQ_2.csv
    
    # Specify format explicitly
    python label_process.py --input file.xlsx --output file.csv --format cq2
    
    # Process multiple files
    python label_process.py --input_dir datasets/ --output_dir processed/
"""

import pandas as pd
import re
import argparse
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List


def parse_time_cq2_style(s: str) -> timedelta:
    """
    Parse CQ2-style time formats.
    
    Supports:
    - "0.05.425" -> 00:00:05.425
    - "HH:MM:SS.mmm" -> standard format
    - Single number -> seconds
    """
    s = str(s).strip()
    
    # Check if already in HH:MM:SS format
    if re.match(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$", s):
        hh, mm, rest = s.split(":")
        if "." in rest:
            ss, ms = rest.split(".")
            return timedelta(
                hours=int(hh), 
                minutes=int(mm),
                seconds=int(ss), 
                milliseconds=int(ms.ljust(3, '0')[:3])
            )
        else:
            return timedelta(hours=int(hh), minutes=int(mm), seconds=int(rest))
    
    # Parse dot-separated format: "0.05.425"
    parts = s.split(".")
    if len(parts) == 3:
        m, sec, ms = parts
        return timedelta(
            minutes=int(m), 
            seconds=int(sec),
            milliseconds=int((ms + "000")[:3])
        )
    elif len(parts) == 2:
        sec, ms = parts
        return timedelta(
            seconds=int(sec),
            milliseconds=int((ms + "000")[:3])
        )
    else:
        return timedelta(seconds=float(s))


def parse_time_cq3_style(s: str) -> timedelta:
    """
    Parse CQ3/CQ4-style time formats.
    
    Converts formats like '00:03.067' or '0.05.425' to timedelta.
    More flexible than CQ2, normalizing colons to dots.
    """
    s = str(s).strip()
    s = s.replace(":", ".")
    parts = [p for p in s.split(".") if p]
    
    if len(parts) == 3:
        m, sec, ms = parts
        return timedelta(
            minutes=int(m), 
            seconds=int(sec), 
            milliseconds=int((ms + "000")[:3])
        )
    elif len(parts) == 2:
        sec, ms = parts
        return timedelta(
            seconds=int(sec), 
            milliseconds=int((ms + "000")[:3])
        )
    elif len(parts) == 1:
        return timedelta(seconds=float(parts[0]))
    
    return timedelta(0)


def parse_time_auto(s: str) -> timedelta:
    """
    Auto-detect and parse time format.
    Tries CQ3 style first (more flexible), then falls back to CQ2.
    """
    try:
        return parse_time_cq3_style(s)
    except:
        try:
            return parse_time_cq2_style(s)
        except:
            return timedelta(0)


def format_timedelta_to_timestamp(td: timedelta) -> str:
    """Convert timedelta to HH:MM:SS.mmm format."""
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // (3600 * 1000)
    rem = total_ms % (3600 * 1000)
    minutes = rem // (60 * 1000)
    rem %= 60 * 1000
    seconds = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def detect_time_behavior_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Auto-detect time and behavior columns from DataFrame.
    
    Returns:
        Tuple of (time_column_name, behavior_column_name)
    """
    df.columns = [str(c).strip() for c in df.columns]
    
    time_col = None
    beh_col = None
    
    # Look for time column
    for c in df.columns:
        lc = c.lower()
        if any(keyword in lc for keyword in ["time", "(.s.)", "(.s", "start", "end"]):
            time_col = c
            break
    
    # Look for behavior column
    for c in df.columns:
        lc = c.lower()
        if any(keyword in lc for keyword in ["behavior", "behaviour", "notes", "label", "action"]):
            beh_col = c
            break
    
    # Fallback: use first three columns if detection fails
    if time_col is None:
        time_col = df.columns[0]
        print(f"⚠️  Warning: Could not detect time column, using: {time_col}")
    
    if beh_col is None:
        # Try to find behavior column in remaining columns
        for i, c in enumerate(df.columns):
            if c != time_col:
                beh_col = c
                break
        if beh_col is None:
            beh_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]
        print(f"⚠️  Warning: Could not detect behavior column, using: {beh_col}")
    
    return time_col, beh_col


def process_cq2_format(df: pd.DataFrame, time_col: str, beh_col: str) -> pd.DataFrame:
    """
    Process CQ2-style format with range notation (e.g., "start -- end").
    """
    split_pattern = re.compile(r"\s*(?:--|-|to|TO|→|~)\s*")
    starts, ends = [], []
    
    for row_value in df[time_col].astype(str).str.strip():
        parts = split_pattern.split(row_value)
        if len(parts) == 2:
            # Range format: start -- end
            s_td = parse_time_cq2_style(parts[0])
            e_td = parse_time_cq2_style(parts[1])
            starts.append(format_timedelta_to_timestamp(s_td))
            ends.append(format_timedelta_to_timestamp(e_td))
        else:
            # Single timestamp
            td = parse_time_cq2_style(parts[0]) if parts else timedelta(0)
            starts.append(format_timedelta_to_timestamp(td))
            ends.append("")
    
    notes = df[beh_col].astype(str).str.strip()
    return pd.DataFrame({"Start": starts, "End": ends, "Notes": notes})


def process_cq3_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process CQ3/CQ4-style format with separate Start, End, Notes columns.
    """
    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Use first 3 columns
    start_col = df.columns[0]
    end_col = df.columns[1]
    note_col = df.columns[2]
    
    starts = [format_timedelta_to_timestamp(parse_time_cq3_style(v)) 
              for v in df[start_col]]
    ends = [format_timedelta_to_timestamp(parse_time_cq3_style(v)) 
            for v in df[end_col]]
    notes = df[note_col].astype(str).str.strip()
    
    return pd.DataFrame({"Start": starts, "End": ends, "Notes": notes})


def auto_detect_format(df: pd.DataFrame) -> str:
    """
    Auto-detect the format of the Excel file.
    
    Returns:
        'cq2', 'cq3', or 'standard'
    """
    # Check if we have exactly 3 columns with start/end pattern
    if len(df.columns) == 3:
        col_names = [str(c).lower() for c in df.columns]
        if any('start' in c or 'begin' in c for c in col_names) and \
           any('end' in c or 'stop' in c for c in col_names):
            return 'cq3'
    
    # Check for range notation in first column (CQ2 style)
    first_col = df.iloc[:, 0].astype(str)
    range_pattern = re.compile(r".*(?:--|to|→|~).*")
    if first_col.str.match(range_pattern).any():
        return 'cq2'
    
    # Default to cq3 (more flexible)
    return 'cq3'


def convert_file(input_path: str, output_path: str, format_type: str = 'auto'):
    """
    Convert behavior annotation Excel file to standardized CSV format.
    
    Args:
        input_path: Path to input Excel file
        output_path: Path to output CSV file
        format_type: 'auto', 'cq2', 'cq3', or 'cq4'
    """
    print(f"\n{'='*70}")
    print(f"Processing: {Path(input_path).name}")
    print(f"{'='*70}")
    
    # Read Excel file
    df = pd.read_excel(input_path, sheet_name=0)
    print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Auto-detect format if needed
    if format_type == 'auto':
        format_type = auto_detect_format(df)
        print(f"🔍 Auto-detected format: {format_type.upper()}")
    else:
        print(f"📝 Using specified format: {format_type.upper()}")
    
    # Process based on format
    if format_type == 'cq2':
        time_col, beh_col = detect_time_behavior_columns(df)
        print(f"   Time column: {time_col}")
        print(f"   Behavior column: {beh_col}")
        out_df = process_cq2_format(df, time_col, beh_col)
    else:  # cq3 or cq4 (same processing)
        out_df = process_cq3_format(df)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    
    print(f"✅ Converted {len(out_df)} behavior intervals")
    print(f"💾 Saved to: {output_path}")
    print()


def batch_convert_directory(input_dir: str, output_dir: str, 
                           format_type: str = 'auto',
                           pattern: str = "*.xlsx"):
    """
    Batch convert all Excel files in a directory.
    
    Args:
        input_dir: Directory containing Excel files
        output_dir: Directory to save CSV files
        format_type: Format to use for all files
        pattern: File pattern to match (default: *.xlsx)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    excel_files = list(input_dir.glob(pattern))
    
    if not excel_files:
        print(f"❌ No Excel files found in {input_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Batch Processing: {len(excel_files)} files")
    print(f"{'='*70}")
    
    success_count = 0
    for excel_file in excel_files:
        try:
            # Generate output filename
            output_file = output_dir / f"{excel_file.stem}.csv"
            convert_file(str(excel_file), str(output_file), format_type)
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {excel_file.name}: {e}\n")
    
    print(f"{'='*70}")
    print(f"✅ Batch conversion complete: {success_count}/{len(excel_files)} files")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert behavior annotation Excel files to standardized CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with auto-detection
  python label_process.py --input data.xlsx --output data.csv
  
  # Single file with specific format
  python label_process.py --input CQ_2.xlsx --output CQ_2.csv --format cq2
  
  # Batch process directory
  python label_process.py --input_dir datasets/ --output_dir processed/
  
  # Batch process with specific format
  python label_process.py --input_dir datasets/ --output_dir processed/ --format cq3
        """
    )
    
    # Single file mode
    parser.add_argument('--input', '--input_file', type=str,
                       help='Path to input Excel file')
    parser.add_argument('--output', '--output_file', type=str,
                       help='Path to output CSV file')
    
    # Batch mode
    parser.add_argument('--input_dir', type=str,
                       help='Directory containing Excel files for batch processing')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save CSV files for batch processing')
    parser.add_argument('--pattern', type=str, default='*.xlsx',
                       help='File pattern for batch processing (default: *.xlsx)')
    
    # Common options
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'cq2', 'cq3', 'cq4'],
                       help='Time format type (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_dir:
        # Batch mode
        if not args.output_dir:
            parser.error("--output_dir is required when using --input_dir")
        batch_convert_directory(args.input_dir, args.output_dir, 
                              args.format, args.pattern)
    elif args.input:
        # Single file mode
        if not args.output:
            parser.error("--output is required when using --input")
        convert_file(args.input, args.output, args.format)
    else:
        # No arguments provided, show help
        parser.print_help()
        print("\n❌ Error: Either --input/--output or --input_dir/--output_dir must be specified")


if __name__ == "__main__":
    main()
