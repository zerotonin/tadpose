# Why `workflow.py` and `trajectory_io.py` are parked here

These two modules were carried over from the pre-refactor
`tadpole_wells` / `24well_pipe` layout but cannot be imported in the
current `tadpose` package as-is.

## `workflow.py` — `ExperimentSetupManager`

The top-level interactive orchestrator depends on three collaborator
manager classes that were **never migrated** into the `tadpose`
package:

- `InputFolderManager` (was `manager_classes.InputFolderManager`)
- `CameraManager`       (was `manager_classes.CameraManager`)
- `SeriesInfoManager`   (was `manager_classes.SeriesInfoManager`)

The other collaborators it uses *do* exist in the package
(`experiment_manager`, `plate_manager`, `file_manager`,
`preset_manager`, `slurm_jobs`, `database`), so reviving the
orchestrator requires only migrating the three missing managers from
the old repository, then rewiring imports to `tadpose.*` and routing
its hard-coded debug paths through `tadpose.config`.

### Template for the revival

The `yolo_tools` insect-recording pipeline (lab-internal sibling
project) is a **finished** implementation of this exact architecture —
TadPose's `database` module was forked from its `FlyChoiceDatabase`.
Use it as the reference when writing the missing managers:

| TadPose (to write / revive) | `yolo_tools` reference          |
|-----------------------------|---------------------------------|
| `workflow.ExperimentSetupManager` | `workflow/ExperimentSetupManager.py` |
| `slurm_jobs.SlurmJobManager`      | `workflow/SlurmJobManager.py`        |
| `InputFolderManager` / `CameraManager` / `SeriesInfoManager` | `analysis_file_manager/{AnalysisFileManager,PresetManager}.py`, `database/StimulusManager.py` patterns |

## `trajectory_io.py`

A thin argparse CLI wrapper around the old
`Velocity_and_Posture_Extractor`, whose logic now lives as functions
in `tadpose.feature_extraction`. It is superseded by the package CLI
(`tadpose.cli`, planned) and kept here only for reference.
