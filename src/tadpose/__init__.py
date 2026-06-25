# ╔══════════════════════════════════════════════════════════════╗
# ║  TadPose                                                    ║
# ║  « automated behavioural phenotyping from 24-well plates »  ║
# ╚══════════════════════════════════════════════════════════════╝
"""
TadPose — well detection, video segmentation, posture-velocity
feature extraction, and unsupervised behavioural clustering for
*Xenopus laevis* tadpole seizure models.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("tadpose")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
