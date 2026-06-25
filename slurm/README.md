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
