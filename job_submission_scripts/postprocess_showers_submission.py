#!/usr/bin/env python3
"""
Submit per-chunk SLURM jobs for postprocess_showers.py.

Reads each registry_pdg_<N>.txt (listing combined hit-parquet paths), splits
it into chunks of N parquets each (default 200), and sbatchs a NON-array
SLURM job per chunk. Each job produces 3 HDF5 files:

    <output>/pdg_<N>/chunk_<XXXX>_electrons.h5
    <output>/pdg_<N>/chunk_<XXXX>_muons.h5
    <output>/pdg_<N>/chunk_<XXXX>_photons.h5

Example:

#electron
python /n/home04/hhanif/TAMBO-opt/job_submission_scripts/postprocess_showers_submission.py     --registry-dir /n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_samples_for_training/     --output-dir   /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files     --script       /n/home04/hhanif/TAMBO-opt/job_submission_scripts/postprocess_showers.py     --chunk-size   500     --partition    serial_requeue --time 4:00:00 --mem 16G --cpus 1 --particles electrons --electrons-nmax 4096 --electrons-dx 10

#photon
python /n/home04/hhanif/TAMBO-opt/job_submission_scripts/postprocess_showers_submission.py     --registry-dir /n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_samples_for_training/     --output-dir   /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files     --script       /n/home04/hhanif/TAMBO-opt/job_submission_scripts/postprocess_showers.py     --chunk-size   25     --partition    serial_requeue --time 4:00:00 --mem 16G --cpus 1 --particles photons --photons-nmax 4096 --photons-dx 8


python /n/home04/hhanif/TAMBO-opt/job_submission_scripts/postprocess_showers_submission.py \
    --output-dir   /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files \
    --script       /n/home04/hhanif/TAMBO-opt/job_submission_scripts/postprocess_showers.py \
    --partition    serial_requeue \
    --time 4:00:00 \
    --mem 16G \
    --cpus 1 \
    --particles photons \
    --photons-nmax 4096 \
    --photons-dx 8 \
    --resubmit /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files/_submit/run_20260426_183055/


    python submit_postprocess.py \\
        --registry-dir /path/to/registries \\
        --output-dir   /path/to/h5_out \\
        --script       /abs/path/to/postprocess_showers.py \\
        --chunk-size   200 \\
        --partition    shared --time 02:00:00 --mem 16G --cpus 4

    # limit to specific PDGs
    python submit_postprocess.py ... --pdgs 11 -11 111

    # dry-run (write sbatch scripts + chunk lists, don't submit)
    python submit_postprocess.py ... --dry-run

    # smoke test: submit exactly ONE job per PDG (5 jobs total)
    python submit_postprocess.py ... --test-run
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime


DEFAULT_PDGS = [11, -11, 111, 211, -211]

# Default cluster env setup — overridable via --env-setup / --env-setup-file /
# --no-env-setup. Change these defaults if your lab uses a different module +
# environment combination.
DEFAULT_ENV_SETUP = [
    "module load python/3.12.11-fasrc01",
    'eval "$(mamba shell hook --shell bash)"',
    "mamba config set changeps1 False",
    "mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/",
]


def _read_registry(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _chunked(items: list, n: int):
    for i in range(0, len(items), n):
        yield i // n, items[i:i + n]


def _write_chunk_list(path: str, paths: list[str]) -> None:
    with open(path, "w") as f:
        for p in paths:
            f.write(p + "\n")


def _write_sbatch_script(
    sbatch_path: str,
    job_name: str,
    log_dir: str,
    partition: str,
    time_limit: str,
    mem: str,
    cpus: int,
    account: str | None,
    extra_sbatch: list[str],
    env_setup: list[str],
    # command
    python_exec: str,
    script_path: str,
    chunk_list_path: str,
    incident_pdg: int,
    chunk_id: int,
    output_dir: str,
    post_args: list[str],
) -> None:
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --time={time_limit}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --output={log_dir}/{job_name}_%j.out")
    lines.append(f"#SBATCH --error={log_dir}/{job_name}_%j.err")
    if account:
        lines.append(f"#SBATCH --account={account}")
    for line in extra_sbatch:
        lines.append(f"#SBATCH {line}")

    lines.append("")
    lines.append("echo \"Host: $(hostname)\"")
    lines.append("echo \"Date: $(date)\"")
    lines.append(f"echo \"Job : {job_name}\"")
    lines.append("")

    # Environment setup (module load, mamba/conda activate, etc.)
    # These run BEFORE `set -e` because `mamba shell hook` and similar can
    # legitimately touch unset variables.
    if env_setup:
        lines.append("# ── Environment setup ──")
        for cmd_line in env_setup:
            lines.append(cmd_line)
        lines.append("")

    lines.append("set -e")
    lines.append("")

    cmd = [
        python_exec,
        script_path,
        "--chunk-list",   chunk_list_path,
        "--incident-pdg", str(incident_pdg),
        "--chunk-id",     str(chunk_id),
        "--output-dir",   output_dir,
        *post_args,
    ]
    # single-line command for clarity in the sbatch script
    lines.append("srun " + " ".join(_shell_quote(c) for c in cmd))
    lines.append("")

    with open(sbatch_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(sbatch_path, 0o755)


def _shell_quote(s: str) -> str:
    # lightweight quoter — wraps in single quotes if needed
    import shlex
    return shlex.quote(str(s))


OOM_PATTERNS = [
    "out of memory",
    "oom-kill",
    "killed process",
    "memoryerror",
    "cannot allocate memory",
    "bus error",
    "slurmstepd: error.*memory",
]


def _is_oom_log(log_path: str) -> bool:
    """Return True if the log file contains OOM-related patterns."""
    try:
        with open(log_path, errors="replace") as f:
            text = f.read().lower()
        import re
        return any(re.search(pat, text) for pat in OOM_PATTERNS)
    except OSError:
        return False


def _find_oom_jobs(run_dir: str) -> list[dict]:
    """
    Scan <run_dir>/logs/*.err for OOM failures.
    Returns list of dicts with: job_name, sbatch_path, chunk_list_path, pdg, chunk_id.
    """
    import re
    log_dir    = os.path.join(run_dir, "logs")
    sbatch_dir = os.path.join(run_dir, "sbatch")
    chunks_dir = os.path.join(run_dir, "chunks")

    if not os.path.isdir(log_dir):
        print(f"ERROR: no logs dir found in run_dir: {log_dir}", file=sys.stderr)
        sys.exit(1)

    failed = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".err"):
            continue
        log_path = os.path.join(log_dir, fname)
        if not _is_oom_log(log_path):
            continue

        # fname pattern: pp_pdg<PDG>_c<CHUNK>_<JOBID>.err
        m = re.match(r"(pp_pdg(-?\d+)_c(\d{4}))_\d+\.err$", fname)
        if not m:
            print(f"  WARNING: cannot parse job name from {fname}, skipping.")
            continue

        job_name        = m.group(1)
        pdg             = int(m.group(2))
        chunk_id        = int(m.group(3))
        sbatch_path     = os.path.join(sbatch_dir, f"{job_name}.sbatch")
        chunk_list_path = os.path.join(chunks_dir, f"pdg_{pdg}_chunk_{chunk_id:04d}.txt")

        if not os.path.isfile(sbatch_path):
            print(f"  WARNING: sbatch script missing for {job_name}: {sbatch_path}")
            sbatch_path = None
        if not os.path.isfile(chunk_list_path):
            print(f"  WARNING: chunk list missing for {job_name}: {chunk_list_path}")
            chunk_list_path = None

        failed.append({
            "job_name":        job_name,
            "pdg":             pdg,
            "chunk_id":        chunk_id,
            "sbatch_path":     sbatch_path,
            "chunk_list_path": chunk_list_path,
            "log_path":        log_path,
        })

    return failed


def _bump_mem(mem_str: str, factor: float = 2.0) -> str:
    """Double (or scale by factor) a SLURM memory string like 8G or 16384M."""
    import re
    m = re.match(r"^(\d+(?:\.\d+)?)([MmGgTt]?)$", mem_str.strip())
    if not m:
        return mem_str
    val  = float(m.group(1)) * factor
    unit = m.group(2).upper() or "M"
    return f"{int(val)}{unit}"


def _build_post_args(args) -> list[str]:
    """Build passthrough args list for postprocess_showers.py."""
    post_args: list[str] = []
    if args.electrons_dx   is not None: post_args += ["--electrons-dx",   str(args.electrons_dx)]
    if args.muons_dx       is not None: post_args += ["--muons-dx",       str(args.muons_dx)]
    if args.photons_dx     is not None: post_args += ["--photons-dx",     str(args.photons_dx)]
    if args.electrons_nmax is not None: post_args += ["--electrons-nmax", str(args.electrons_nmax)]
    if args.muons_nmax     is not None: post_args += ["--muons-nmax",     str(args.muons_nmax)]
    if args.photons_nmax   is not None: post_args += ["--photons-nmax",   str(args.photons_nmax)]
    if args.no_time:  post_args.append("--no-time")
    if args.no_dedup: post_args.append("--no-dedup")
    if args.dedup_time_tol       is not None:
        post_args += ["--dedup-time-tol",       str(args.dedup_time_tol)]
    if args.dedup_energy_rel_tol is not None:
        post_args += ["--dedup-energy-rel-tol", str(args.dedup_energy_rel_tol)]
    if args.dedup_xy_tol         is not None:
        post_args += ["--dedup-xy-tol",         str(args.dedup_xy_tol)]
    if args.particles:
        post_args += ["--particles", *args.particles]
    return post_args


def _resolve_env_setup(args) -> list[str]:
    """Resolve env setup lines from args (same precedence as main submission)."""
    if args.no_env_setup:
        return []
    if args.env_setup_file:
        with open(args.env_setup_file) as f:
            return [ln.rstrip("\n") for ln in f
                    if ln.strip() and not ln.lstrip().startswith("#")]
    if args.env_setup is not None:
        return list(args.env_setup)
    return list(DEFAULT_ENV_SETUP)


def _resubmit_oom(run_dir: str, args) -> None:
    """Find OOM-failed jobs in run_dir and resubmit them with bumped memory."""
    print(f"\nScanning run dir for OOM failures: {run_dir}")
    failed = _find_oom_jobs(run_dir)

    if not failed:
        print("No OOM failures found in logs. Nothing to resubmit.")
        return

    new_mem         = _bump_mem(args.mem, factor=args.oom_mem_factor)
    env_setup_lines = _resolve_env_setup(args)
    post_args       = _build_post_args(args)

    print(f"Found {len(failed)} OOM job(s).  Original mem: {args.mem}  ->  New mem: {new_mem}\n")

    # create a resubmit sub-dir inside the original run_dir
    stamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
    resub_dir    = os.path.join(run_dir, f"resubmit_oom_{stamp}")
    resub_sbatch = os.path.join(resub_dir, "sbatch")
    resub_logs   = os.path.join(resub_dir, "logs")
    for d in (resub_sbatch, resub_logs):
        os.makedirs(d, exist_ok=True)

    resubmitted = 0
    manifest    = []

    for job in failed:
        if job["chunk_list_path"] is None:
            print(f"  SKIP {job['job_name']} -- chunk list missing")
            continue

        job_name    = job["job_name"] + "_oom"
        sbatch_path = os.path.join(resub_sbatch, f"{job_name}.sbatch")
        output_dir  = os.path.abspath(args.output_dir)
        pdg         = job["pdg"]
        chunk_id    = job["chunk_id"]

        # remove broken/partial h5 outputs before resubmitting
        for particle in (args.particles or ["electrons", "muons", "photons"]):
            broken = os.path.join(output_dir, f"pdg_{pdg}",
                                  f"chunk_{chunk_id:04d}_{particle}.h5")
            if os.path.exists(broken):
                os.remove(broken)
                print(f"  Removed broken h5: {broken}")

        _write_sbatch_script(
            sbatch_path     = sbatch_path,
            job_name        = job_name,
            log_dir         = resub_logs,
            partition       = args.partition,
            time_limit      = args.time,
            mem             = new_mem,
            cpus            = args.cpus,
            account         = args.account,
            extra_sbatch    = args.extra_sbatch,
            env_setup       = env_setup_lines,
            python_exec     = args.python,
            script_path     = os.path.abspath(args.script),
            chunk_list_path = job["chunk_list_path"],
            incident_pdg    = pdg,
            chunk_id        = chunk_id,
            output_dir      = output_dir,
            post_args       = post_args,
        )

        if args.dry_run:
            print(f"  [dry-run] would resubmit {job['job_name']} -> {job_name}  mem={new_mem}")
            manifest.append({"job_name": job_name, "jobid": None})
        else:
            try:
                out = subprocess.check_output(
                    ["sbatch", sbatch_path], stderr=subprocess.STDOUT,
                ).decode().strip()
                jobid = out.rsplit()[-1] if out else ""
                print(f"  Resubmitted {job['job_name']} -> {job_name}  (jobid={jobid}, mem={new_mem})")
                manifest.append({"job_name": job_name, "jobid": jobid})
                resubmitted += 1
            except subprocess.CalledProcessError as e:
                print(f"  ERROR resubmitting {job['job_name']}: {e.output.decode().strip()}",
                      file=sys.stderr)
                manifest.append({"job_name": job_name, "jobid": None, "error": True})

    manifest_path = os.path.join(resub_dir, "resubmitted.tsv")
    with open(manifest_path, "w") as f:
        f.write("job_name\tjobid\n")
        for r in manifest:
            f.write(f"{r['job_name']}\t{r.get('jobid') or ''}\n")

    print(f"\nResubmitted : {resubmitted}/{len(failed)} OOM jobs")
    print(f"New mem     : {new_mem}")
    print(f"Resub dir   : {resub_dir}")
    print(f"Manifest    : {manifest_path}")


def main():
    p = argparse.ArgumentParser(
        description="Submit non-array SLURM jobs for postprocess_showers.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--registry-dir", required=False, default=None,
                   help="Directory containing registry_pdg_<N>.txt files. "
                        "Required for normal submission; not needed with --resubmit.")
    p.add_argument("--output-dir", required=True,
                   help="Root dir for H5 output (per-PDG subdirs created here).")
    p.add_argument("--script", required=True,
                   help="Absolute path to postprocess_showers.py.")
    p.add_argument("--python", default="python",
                   help="Python executable name/path used inside the sbatch job. "
                        "Defaults to bare 'python' so it picks up whatever env the "
                        "--env-setup lines activate.")
    p.add_argument("--pdgs", nargs="+", type=int, default=DEFAULT_PDGS,
                   help="Incident PDG values to process (looks for registry_pdg_<N>.txt each).")
    p.add_argument("--chunk-size", type=int, default=200,
                   help="Number of parquet files per SLURM job.")

    # SLURM resources
    p.add_argument("--partition", default="shared")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--mem", default="16G")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--account", default=None)
    p.add_argument("--extra-sbatch", nargs="*", default=[],
                   help="Extra raw #SBATCH directives, e.g. --extra-sbatch '--qos=normal'")

    # environment setup inside each sbatch script
    p.add_argument("--env-setup", nargs="*", default=None,
                   help="One or more shell commands to run BEFORE srun "
                        "(e.g. module load …, mamba activate …). "
                        "If omitted, a sensible cluster default is used.")
    p.add_argument("--env-setup-file", default=None,
                   help="Path to a file containing one shell command per line to "
                        "paste verbatim into each sbatch script before srun. "
                        "Overrides --env-setup and the built-in default.")
    p.add_argument("--no-env-setup", action="store_true",
                   help="Skip all environment setup lines in the sbatch script.")

    # work-dir for logs + generated sbatch/chunk-list files
    p.add_argument("--work-dir", default=None,
                   help="Dir to write chunk lists + sbatch scripts + logs. "
                        "Default: <output-dir>/_submit")

    # passthrough flags for postprocess_showers.py
    p.add_argument("--electrons-dx",   type=float)
    p.add_argument("--muons-dx",       type=float)
    p.add_argument("--photons-dx",     type=float)
    p.add_argument("--electrons-nmax", type=int)
    p.add_argument("--muons-nmax",     type=int)
    p.add_argument("--photons-nmax",   type=int)
    p.add_argument("--no-time",  action="store_true")
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--dedup-time-tol",       type=float)
    p.add_argument("--dedup-energy-rel-tol", type=float)
    p.add_argument("--dedup-xy-tol",         type=float)
    p.add_argument("--particles", nargs="+", default=None,
                   choices=["electrons", "muons", "photons"],
                   help="Only produce H5 files for these secondary types. "
                        "Pass any subset of electrons/muons/photons. "
                        "Omit to produce all three per job.")

    # control
    p.add_argument("--dry-run", action="store_true",
                   help="Create chunk lists and sbatch scripts but DO NOT submit.")
    p.add_argument("--max-jobs", type=int, default=None,
                   help="Global cap on total sbatch submissions.")
    p.add_argument("--test-run", action="store_true",
                   help="Submit exactly ONE job per PDG (5 jobs total when all 5 "
                        "PDGs are selected). Uses only the first --chunk-size "
                        "parquets from each registry. Useful for end-to-end smoke "
                        "tests before launching the full campaign.")

    # OOM resubmission
    p.add_argument("--resubmit", metavar="RUN_DIR", default=None,
                   help="Path to a previous run_<timestamp> directory. Scans its "
                        "logs/*.err files for OOM failures and resubmits those "
                        "chunks with bumped memory (see --oom-mem-factor). "
                        "All other flags (--partition, --time, --mem, passthrough "
                        "args, etc.) are reused unless overridden on the command line.")
    p.add_argument("--oom-mem-factor", type=float, default=2.0,
                   help="Multiply --mem by this factor when resubmitting OOM jobs. "
                        "Default: 2.0 (doubles the memory).")

    args = p.parse_args()

    # ── OOM resubmit mode ────────────────────────────────────────────────────
    if args.resubmit is not None:
        run_dir = os.path.abspath(args.resubmit)
        if not os.path.isdir(run_dir):
            print(f"ERROR: --resubmit path does not exist: {run_dir}", file=sys.stderr)
            sys.exit(1)
        _resubmit_oom(run_dir, args)
        sys.exit(0)
    # ─────────────────────────────────────────────────────────────────────────

    if not args.registry_dir:
        p.error("--registry-dir is required for normal submission (omit only when using --resubmit)")
    registry_dir = os.path.abspath(args.registry_dir)
    output_dir   = os.path.abspath(args.output_dir)
    script_path  = os.path.abspath(args.script)

    if not os.path.isdir(registry_dir):
        print(f"ERROR: registry dir missing: {registry_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(script_path):
        print(f"ERROR: postprocess script missing: {script_path}", file=sys.stderr)
        sys.exit(1)

    work_dir = os.path.abspath(args.work_dir) if args.work_dir \
               else os.path.join(output_dir, "_submit")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Resolve env-setup lines (precedence: --no-env-setup > file > --env-setup > default) ---
    if args.env_setup_file and not os.path.isfile(os.path.abspath(args.env_setup_file)):
        print(f"ERROR: --env-setup-file not found: {args.env_setup_file}", file=sys.stderr)
        sys.exit(1)
    env_setup_lines: list[str] = _resolve_env_setup(args)

    # one timestamped run dir keeps things tidy when re-running
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = os.path.join(work_dir, f"run_{stamp}")
    chunks_d = os.path.join(run_dir, "chunks")
    sbatch_d = os.path.join(run_dir, "sbatch")
    log_d    = os.path.join(run_dir, "logs")
    for d in (chunks_d, sbatch_d, log_d):
        os.makedirs(d, exist_ok=True)

    # build passthrough arg list for postprocess_showers.py
    post_args: list[str] = _build_post_args(args)

    submitted       = 0
    jobs_per_pdg    = {}
    submission_log  = []

    print(f"Registry dir  : {registry_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Script        : {script_path}")
    print(f"Python        : {args.python}")
    print(f"Chunk size    : {args.chunk_size}")
    print(f"Run dir       : {run_dir}")
    print(f"PDGs          : {args.pdgs}")
    print(f"Particles     : {args.particles if args.particles else 'electrons muons photons (all)'}")
    if env_setup_lines:
        print(f"Env setup     : {len(env_setup_lines)} line(s)")
        for ln in env_setup_lines:
            print(f"                  {ln}")
    else:
        print("Env setup     : (none)")
    if args.test_run:
        print(f"TEST RUN      : submitting only 1 job per PDG "
              f"(first {args.chunk_size} parquets each)")
    print()

    for pdg in args.pdgs:
        registry_path = os.path.join(registry_dir, f"registry_pdg_{pdg}.txt")
        if not os.path.exists(registry_path):
            print(f"  [pdg={pdg}] no registry found at {registry_path} — skipping")
            continue

        paths = _read_registry(registry_path)
        if not paths:
            print(f"  [pdg={pdg}] registry is empty — skipping")
            continue

        # test-run: only the first chunk from each registry
        if args.test_run:
            paths = paths[: args.chunk_size]

        n_chunks = math.ceil(len(paths) / args.chunk_size)
        print(f"  [pdg={pdg}] {len(paths)} parquets → {n_chunks} chunks"
              + ("  [TEST]" if args.test_run else ""))

        jobs_per_pdg[pdg] = 0

        for chunk_id, chunk_paths in _chunked(paths, args.chunk_size):
            if args.max_jobs is not None and submitted >= args.max_jobs:
                print(f"  reached --max-jobs={args.max_jobs}, stopping.")
                break

            chunk_list_path = os.path.join(
                chunks_d, f"pdg_{pdg}_chunk_{chunk_id:04d}.txt")
            _write_chunk_list(chunk_list_path, chunk_paths)

            job_name = f"pp_pdg{pdg}_c{chunk_id:04d}"
            sbatch_path = os.path.join(
                sbatch_d, f"{job_name}.sbatch")

            _write_sbatch_script(
                sbatch_path     = sbatch_path,
                job_name        = job_name,
                log_dir         = log_d,
                partition       = args.partition,
                time_limit      = args.time,
                mem             = args.mem,
                cpus            = args.cpus,
                account         = args.account,
                extra_sbatch    = args.extra_sbatch,
                env_setup       = env_setup_lines,
                python_exec     = args.python,
                script_path     = script_path,
                chunk_list_path = chunk_list_path,
                incident_pdg    = pdg,
                chunk_id        = chunk_id,
                output_dir      = output_dir,
                post_args       = post_args,
            )

            if args.dry_run:
                print(f"    [dry-run] would submit: {sbatch_path}")
                submission_log.append({
                    "pdg": pdg, "chunk_id": chunk_id,
                    "sbatch": sbatch_path, "jobid": None,
                })
            else:
                try:
                    out = subprocess.check_output(
                        ["sbatch", sbatch_path],
                        stderr=subprocess.STDOUT,
                    ).decode().strip()
                    # parse "Submitted batch job <id>"
                    jobid = out.rsplit()[-1] if out else ""
                    print(f"    submitted {job_name}  (jobid={jobid})")
                    submission_log.append({
                        "pdg": pdg, "chunk_id": chunk_id,
                        "sbatch": sbatch_path, "jobid": jobid,
                    })
                except subprocess.CalledProcessError as e:
                    print(f"    ERROR submitting {job_name}: {e.output.decode().strip()}",
                          file=sys.stderr)
                    submission_log.append({
                        "pdg": pdg, "chunk_id": chunk_id,
                        "sbatch": sbatch_path, "jobid": None, "error": True,
                    })

            submitted += 1
            jobs_per_pdg[pdg] += 1

        if args.max_jobs is not None and submitted >= args.max_jobs:
            break

    # write a single manifest
    manifest_path = os.path.join(run_dir, "submitted.tsv")
    with open(manifest_path, "w") as f:
        f.write("pdg\tchunk_id\tjobid\tsbatch\n")
        for r in submission_log:
            f.write(f"{r['pdg']}\t{r['chunk_id']}\t"
                    f"{r.get('jobid') or ''}\t{r['sbatch']}\n")

    print()
    print("=" * 60)
    print(f"Total jobs    : {submitted} ({'dry-run' if args.dry_run else 'submitted'})")
    for pdg, n in jobs_per_pdg.items():
        print(f"  pdg {pdg:>5d}  : {n}")
    print(f"Manifest      : {manifest_path}")
    print(f"Run dir       : {run_dir}")


if __name__ == "__main__":
    main()