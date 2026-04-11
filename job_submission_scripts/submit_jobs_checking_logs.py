#!/usr/bin/env python3
"""
Recursively scan TAMBO simulation log directories and summarize results.

Usage:
    python summarize_logs.py [LOG_DIR]
    python summarize_logs.py                          # uses default path
    python summarize_logs.py /path/to/logs
    python summarize_logs.py /path/to/logs --verbose  # print per-file breakdown
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

DEFAULT_LOG_DIR = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/logs"

# Patterns from the SBATCH script output
RE_SUCCESS  = re.compile(r"Runtime for simulation #(\d+):\s+(\d+):(\d+)\s+\(mm:ss\)")
RE_TIMEOUT  = re.compile(r"timed out after (\d+)m")
RE_FAILED   = re.compile(r"Simulation failed with exit code (\d+)")
RE_TOTAL    = re.compile(r"Total simulations run:\s+(\d+)")
RE_WALLTIME = re.compile(r"Total wall time:\s+(\d+):(\d+)")


def scan_log_file(path: Path):
    """Parse a single .out log file and return a stats dict."""
    stats = {
        "successes": 0,
        "timeouts": 0,
        "failures": 0,
        "total_runtime_s": 0,   # cumulative successful simulation time
        "runtimes": [],         # individual (mm, ss) tuples
    }

    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return stats

    for m in RE_SUCCESS.finditer(text):
        stats["successes"] += 1
        mm, ss = int(m.group(2)), int(m.group(3))
        stats["total_runtime_s"] += mm * 60 + ss
        stats["runtimes"].append((mm, ss))

    stats["timeouts"] = len(RE_TIMEOUT.findall(text))
    stats["failures"] = len(RE_FAILED.findall(text))

    return stats


def fmt_duration(total_seconds):
    """Format seconds into human-readable H:MM:SS."""
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(description="Summarize TAMBO simulation log files.")
    parser.add_argument("log_dir", nargs="?", default=DEFAULT_LOG_DIR,
                        help=f"Root log directory to scan. Default: {DEFAULT_LOG_DIR}")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-file breakdown.")
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    if not log_root.exists():
        print(f"Error: directory not found: {log_root}", file=sys.stderr)
        sys.exit(1)

    # Collect all .out files grouped by their parent subdirectory
    out_files = sorted(log_root.rglob("*.out"))
    err_files = sorted(log_root.rglob("*.err"))

    if not out_files:
        print(f"No .out log files found under {log_root}")
        sys.exit(0)

    # Group files by subdirectory (e.g. pdg_11, pdg_111, or root)
    groups = defaultdict(list)
    for f in out_files:
        rel = f.parent.relative_to(log_root)
        key = str(rel) if str(rel) != "." else "(root)"
        groups[key].append(f)

    # ---- Per-group summary ----
    grand = {"successes": 0, "timeouts": 0, "failures": 0, "total_runtime_s": 0, "files": 0}

    for group_name in sorted(groups.keys()):
        files = groups[group_name]
        g_success = 0
        g_timeout = 0
        g_failure = 0
        g_runtime = 0

        for f in files:
            stats = scan_log_file(f)
            g_success += stats["successes"]
            g_timeout += stats["timeouts"]
            g_failure += stats["failures"]
            g_runtime += stats["total_runtime_s"]

            if args.verbose and (stats["successes"] or stats["timeouts"] or stats["failures"]):
                print(f"  {f.name}: {stats['successes']} ok, {stats['timeouts']} timeout, {stats['failures']} fail")

        total_sims = g_success + g_timeout + g_failure
        pct = (g_success / total_sims * 100) if total_sims else 0

        print(f"\n{'=' * 60}")
        print(f"  Directory: {group_name}")
        print(f"  Log files scanned:    {len(files)}")
        print(f"  Simulations total:    {total_sims}")
        print(f"  Successful:           {g_success}  ({pct:.1f}%)")
        print(f"  Timed out:            {g_timeout}")
        print(f"  Failed:               {g_failure}")
        print(f"  Cumulative sim time:  {fmt_duration(g_runtime)}")
        if g_success > 0:
            avg_s = g_runtime / g_success
            print(f"  Avg time/success:     {fmt_duration(int(avg_s))}")
        print(f"{'=' * 60}")

        grand["successes"] += g_success
        grand["timeouts"] += g_timeout
        grand["failures"] += g_failure
        grand["total_runtime_s"] += g_runtime
        grand["files"] += len(files)

    # ---- Grand total ----
    gt = grand["successes"] + grand["timeouts"] + grand["failures"]
    pct = (grand["successes"] / gt * 100) if gt else 0

    print(f"\n{'#' * 60}")
    print(f"  GRAND TOTAL")
    print(f"  Directories scanned:  {len(groups)}")
    print(f"  Log files scanned:    {grand['files']}")
    print(f"  Simulations total:    {gt}")
    print(f"  Successful:           {grand['successes']}  ({pct:.1f}%)")
    print(f"  Timed out:            {grand['timeouts']}")
    print(f"  Failed:               {grand['failures']}")
    print(f"  Cumulative sim time:  {fmt_duration(grand['total_runtime_s'])}")
    if grand["successes"] > 0:
        avg_s = grand["total_runtime_s"] / grand["successes"]
        print(f"  Avg time/success:     {fmt_duration(int(avg_s))}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()