#!/usr/bin/env python3
"""
Analyze trajectory prediction evaluation CSV result files.

Computes comprehensive statistics (count, mean, min, max, std) for evaluation
result CSV files and outputs formatted summary tables.

Usage:
    python analyze_results.py --input-path /path/to/csv/files --output-file results.txt
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from datetime import datetime


def extract_eval_type(fname):
    """Extract evaluation type from filename."""
    if 'best_of' in fname:
        return 'best_of'
    elif 'most_likely' in fname:
        return 'most_likely'
    elif 'z_mode' in fname:
        return 'z_mode'
    elif '_full' in fname:
        return 'full'
    else:
        # Fallback: extract from last part of filename
        return fname.split('_')[-1].replace('.csv', '')


def extract_metric_type(fname):
    """Extract metric type from filename."""
    fname_lower = fname.lower()
    if 'ade' in fname_lower:
        return 'ADE'
    elif 'fde' in fname_lower:
        return 'FDE'
    elif 'kde' in fname_lower:
        return 'KDE'
    else:
        return 'UNKNOWN'


def compute_stats_streaming(filepath, progress_interval=1_000_000):
    """Compute statistics for a CSV file using streaming with progress."""
    count = 0
    total = 0.0
    total_sq = 0.0
    min_val = float('inf')
    max_val = float('-inf')

    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = float(row['value'])
                count += 1
                total += val
                total_sq += val * val
                min_val = min(min_val, val)
                max_val = max(max_val, val)

                # Progress reporting every N rows
                if count % progress_interval == 0:
                    print(f"  Processed {count:,} rows...", flush=True)
    except KeyError:
        print(f"  Warning: 'value' column not found in {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error processing {filepath}: {e}", file=sys.stderr)
        return None

    if count == 0:
        return None

    mean = total / count
    variance = (total_sq / count) - (mean * mean)
    std = math.sqrt(max(0, variance))

    return {'count': count, 'mean': mean, 'min': min_val, 'max': max_val, 'std': std}


def discover_files(input_path, pattern='*.csv'):
    """Discover and group CSV files by metric type."""
    search_pattern = os.path.join(input_path, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        return {}, []

    metrics = defaultdict(list)
    for fpath in files:
        fname = os.path.basename(fpath)
        metric = extract_metric_type(fname)
        metrics[metric].append(fpath)

    return metrics, files


def format_summary_table(results):
    """Format results as a summary table string."""
    lines = []
    lines.append("=" * 95)
    lines.append("EVALUATION RESULTS SUMMARY")
    lines.append("=" * 95)
    lines.append(f"{'Metric':<8} {'Type':<15} {'Count':<18} {'Mean':<12} {'Min':<12} {'Max':<12} {'Std':<12}")
    lines.append("-" * 95)

    for r in results:
        lines.append(f"{r['metric']:<8} {r['type']:<15} {r['count']:<18,} {r['mean']:<12.6f} {r['min']:<12.6f} {r['max']:<12.6f} {r['std']:<12.6f}")

    lines.append("=" * 95)
    return '\n'.join(lines)


def format_key_metrics(results):
    """Format key benchmark metrics."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("KEY METRICS (Standard Benchmark Format)")
    lines.append("=" * 60)

    # Find key results
    best_of_ade = next((r for r in results if r['metric'] == 'ADE' and r['type'] == 'best_of'), None)
    best_of_fde = next((r for r in results if r['metric'] == 'FDE' and r['type'] == 'best_of'), None)
    full_ade = next((r for r in results if r['metric'] == 'ADE' and r['type'] == 'full'), None)
    full_fde = next((r for r in results if r['metric'] == 'FDE' and r['type'] == 'full'), None)

    lines.append(f"\nADE (Best-of-20): {best_of_ade['mean']:.4f}" if best_of_ade else "\nADE (Best-of-20): N/A")
    lines.append(f"FDE (Best-of-20): {best_of_fde['mean']:.4f}" if best_of_fde else "FDE (Best-of-20): N/A")
    lines.append(f"ADE (Mean):       {full_ade['mean']:.4f}" if full_ade else "ADE (Mean): N/A")
    lines.append(f"FDE (Mean):       {full_fde['mean']:.4f}" if full_fde else "FDE (Mean): N/A")

    # Group by eval type
    lines.append("")
    lines.append("-" * 60)
    lines.append("All Metrics by Type:")
    lines.append("-" * 60)

    for eval_type in ['most_likely', 'best_of', 'z_mode', 'full']:
        type_results = [r for r in results if r['type'] == eval_type]
        if type_results:
            lines.append(f"\n{eval_type}:")
            for r in type_results:
                lines.append(f"  {r['metric']:<8} Mean: {r['mean']:.4f}")

    # KDE summary
    lines.append("")
    lines.append("-" * 60)
    lines.append("KDE NLL (Negative Log-Likelihood):")
    lines.append("-" * 60)
    kde_results = [r for r in results if r['metric'] == 'KDE']
    for r in sorted(kde_results, key=lambda x: x['type']):
        lines.append(f"  {r['type']:<15} Mean: {r['mean']:.4f}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze trajectory prediction evaluation CSV result files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python analyze_results.py --input-path /path/to/results --output-file analysis.txt
        """
    )
    parser.add_argument('--input-path', required=True,
                        help='Path to directory containing CSV result files')
    parser.add_argument('--output-file', default='results_analysis.txt',
                        help='Output file path (default: results_analysis.txt)')
    parser.add_argument('--pattern', default='*.csv',
                        help='Glob pattern for CSV files (default: *.csv)')

    args = parser.parse_args()

    # Validate input path
    if not os.path.isdir(args.input_path):
        print(f"Error: Input path does not exist or is not a directory: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    # Discover files
    print(f"Scanning directory: {args.input_path}")
    metrics, all_files = discover_files(args.input_path, args.pattern)

    if not all_files:
        print(f"Error: No CSV files found matching pattern '{args.pattern}' in {args.input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(all_files)} result files:")
    for metric, files in metrics.items():
        print(f"  {metric}: {len(files)} files")
        for fpath in files:
            fname = os.path.basename(fpath)
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            eval_type = extract_eval_type(fname)
            print(f"    - {fname} ({fsize_mb:.1f} MB) -> {eval_type}")

    # Process files
    results = []

    for metric_name in ['ADE', 'FDE', 'KDE']:
        if metric_name not in metrics:
            continue
        print(f"\n{'='*60}")
        print(f"Processing {metric_name} metrics...")
        print('='*60)

        for fpath in sorted(metrics[metric_name]):
            fname = os.path.basename(fpath)
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            eval_type = extract_eval_type(fname)

            print(f"\n{eval_type} ({fsize_mb:.1f} MB):")
            stats = compute_stats_streaming(fpath)

            if stats:
                print(f"  Done! Count: {stats['count']:,}")
                results.append({'metric': metric_name, 'type': eval_type, **stats})
            else:
                print("  No data found!")

    if not results:
        print("\nError: No valid data found in any CSV files.", file=sys.stderr)
        sys.exit(1)

    # Format output
    output_lines = []
    output_lines.append(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Input path: {args.input_path}")
    output_lines.append(f"Files processed: {len(all_files)}")
    output_lines.append("")
    output_lines.append(format_summary_table(results))
    output_lines.append(format_key_metrics(results))

    output_text = '\n'.join(output_lines)

    # Print to console
    print("\n" + output_text)

    # Write to file
    with open(args.output_file, 'w') as f:
        f.write(output_text + '\n')

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_file}")
    print('='*60)


if __name__ == '__main__':
    main()
