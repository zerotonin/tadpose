# Why `trajectory_io.py` is parked here

`trajectory_io.py` is a thin argparse CLI wrapper around the old
`Velocity_and_Posture_Extractor`, whose logic now lives as functions in
`tadpose.feature_extraction`.  It is superseded by the package CLI
(`tadpose <stage>`) and kept here only for reference.

## `workflow.py` has been revived

The top-level orchestrator (`ExperimentSetupManager`) is no longer parked.
It was restored to `tadpose.workflow` together with the three managers that
had never been migrated — `InputFolderManager`, `CameraManager` and
`SeriesInfoManager` — using the `yolo_tools` insect-recording pipeline as
the structural reference (TadPose's `database` module was forked from its
`FlyChoiceDatabase`).  All machine-specific paths now resolve through
`tadpose.config`.
