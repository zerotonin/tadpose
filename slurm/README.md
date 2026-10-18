# SLURM submit scripts

These scripts run the GPU-heavy stages of TadPose on a SLURM cluster
(developed on the University of Otago Aoraki cluster).  They are
orchestration only — every script invokes the installed `tadpose` package
rather than a separate copy of the analysis code.

## One-time setup

1. Install the package on the cluster (`pip install -e .` in the TadPose
   checkout, ideally inside the RAPIDS/cuML conda environment).
2. Create `local_paths.json` from the template and fill in the `hpc`
   profile (`data_root`, `code_root`, `python_interpreter`, `partition`,
   `account`, …).  See the repository README.

Every script sources `load_paths.sh`, which exports that profile as
`TADPOSE_*` environment variables, so no machine-specific path or username
appears in these files:

```bash
source "$(dirname "$0")/load_paths.sh" hpc
${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering --help
```

Because `#SBATCH --account=` directives are parsed before the shell runs
(and so cannot read a runtime variable), the account is left as a
placeholder; override it on submission with
`sbatch --account=$TADPOSE_ACCOUNT …`.

## What each script does

| Script | Purpose | Invokes |
|--------|---------|---------|
| `main_clustering.sh` | Sweep k and leave-out for one feature matrix | `python -m tadpose.clustering` |
| `array_clustering.sh` | Same sweep as a SLURM job array | `python -m tadpose.clustering` |
| `flexible_clustering.sh` | Interactive sweep with partition cycling | `python -m tadpose.clustering` |
| `run_slurm_superprotos.sh` | Per-trial + aggregate superprototype counts | `python -m tadpose.analysis.SlurmSuperPrototypesAnalysis` / `AggregateSuperPrototypesAnalysis` |
| `submit_sleap.sh` | SLEAP pipeline: array of per-video tasks + chained merge (run on the login node) | `sbatch` |
| `sleap_features.sh` | One SLEAP video, stages 0 -> 2 (array task) | `tadpose sleap-features --slurm-array` |
| `sleap_matrix.sh` | Stage 3 clustering bundle, after all tasks succeed | `tadpose sleap-matrix --expect N` |
| `submit_sweep.sh` | k-means stability sweep: GPU workers, or the k-selection afterwards (login node) | `sbatch` |
| `sweep_worker.sh` | One GPU worker: detect the card, pack fits, drain the shared pool | `tadpose sweep work` |
| `kselect.sh` | k-selection tables + agreement figure for every feature set | `tadpose kselect` |
| `sweep_watch.sh` | One-screen sweep monitor: progress per delSize, claims, parked fits, cards, trailing-hour ETA (login node, under `watch`) | `find`, `squeue`, `sacct` |
| `kselect_watch.sh` | One-screen k-selection monitor: stage per feature set, k done, ETA, last log lines (login node, under `watch`) | `wc`, `squeue`, `sacct` |
| `load_paths.sh` | Export the `hpc` profile as `TADPOSE_*` vars | `python -m tadpose.config --export` |

The clustering scripts check whether a `(k, del_size, del_pos)` combination
has already been computed by asking the package for the would-be output
path (without creating it):

```bash
meta_file=$(${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering \
              --print-meta-path -sd "$result_dir" -t "$tag" \
              -nc "$k" -ds "$del_size" -dp "$del_pos")
[ -f "$meta_file" ] || sbatch ... --wrap="$base_cmd -t $tag -nc $k ..."
```

## Feature-set variants

`flexible_clustering.sh` still offers Vanilla / Velocity-only / Posture-only
/ Weighted options for historical continuity, but all of them now run the
same `tadpose.clustering` module.  The feature subset is chosen when the
z-scored matrix is **assembled** (the clustering step simply clusters
whatever matrix it is given), so select the option matching the matrix you
built and pass it as the data file.

## SLEAP pipeline

Turns Caroline Beck's SLEAP `_wells.csv` exports into the clustering bundle
(data flow: `docs/sleap_pipeline.rst`).  Needs these `hpc` profile keys in
`local_paths.json`: `sleap_predictions_root`, `sleap_geometry_csv`,
`sleap_output_root`, `cpu_partition`, `python_interpreter` (and optionally
`account`).  The package does not need to be installed: the scripts put the
checkout's `src/` on `PYTHONPATH`.

```bash
# from the workstation: copy predictions + the ring-geometry table
rsync -a <local SLEAP_2026_predictions>/ <cluster>:<sleap_2026 folder>/

# on the login node, in the checkout (sleap branch)
bash slurm/submit_sleap.sh            # one CPU task per video, then the merge
bash slurm/submit_sleap.sh --force    # redo videos that already have outputs
```

Per video ~3 min and ~4.5 GB (16 GB requested); the merge needs ~38 GB for
the 12 Sept-2024 plates (64 GB requested).  A task skips a video whose
stage-1 and stage-2 CSVs exist, outputs appear only once fully written, and
the merge refuses to run unless all N feature CSVs are present.  Logs go to
`<sleap_output_root>/logs/`.

## k-means stability sweep

Design and grid: `docs/sweep.rst`.  Extra `hpc` profile keys: `gpu_partitions`
(comma-separated), `conda_sh` (path to `conda.sh`), `rapids_env` (the conda env
with cuML).

```bash
# once, on the login node
tadpose sweep init --sweep-dir <sweep> --tag-prefix sleap2026 \
    --feature-set pdv=<bundle>/features_zscored.npy \
    --feature-set vel=<bundle>/features_vel_zscored.npy

bash slurm/submit_sweep.sh workers <sweep> 12                     # spread over gpu_partitions
bash slurm/submit_sweep.sh workers <sweep> 4 --partition=<partition> --qos=<granted qos>
tadpose sweep status --sweep-dir <sweep>                           # done / claimed / parked
bash slurm/submit_sweep.sh kselect <sweep>                         # when remaining=0
watch -n 60 slurm/kselect_watch.sh <sweep>                         # its progress
```

Without `--partition`, the workers are spread over `gpu_partitions`: each
partition gets at most its QOS per-user job cap, and `--mem` is clipped to its
per-job memory cap (both read from `sacctmgr` at submit time).  A single array
over all partitions stalls with `QOSMaxJobsPerUserLimit` as soon as the first
partition is full, and small-card partitions reject the 90 GB default.

Workers are requeue-safe and can be added at any time; a killed worker's claims
go stale after five minutes and are taken over.  Logs: `<sweep>/logs/`.
