"""Human input devices. Hardware and inference dependencies load on demand."""

from .types import DeviceFrame

__version__ = "0.1.0"
__all__ = ["DeviceFrame", "XRDevice", "XRRTCBodyPoseDevice", "MediaPipeTeleopDevice"]


def __getattr__(name):
    if name in ("XRDevice", "XRRTCBodyPoseDevice"):
        from .xr_teleop_device import XRDevice
        return XRDevice
    if name == "MediaPipeTeleopDevice":
        from .mediapipe_device import MediaPipeTeleopDevice
        return MediaPipeTeleopDevice
    raise AttributeError(name)
