"""Camera input using MediaPipe Tasks and the established human-frame conversion."""
from copy import deepcopy
import time

from ._mediapipe_processing import MediaPipeTeleopDevice as _Processing
from .types import DeviceFrame
from .models import resolve_model


class MediaPipeTeleopDevice(_Processing):
    """Omitted models download once into the user cache; local paths override them.

    Constructing opens the camera and starts capture. Use close() or a context manager.
    """

    def __init__(self, *, pose_model=None, hand_model=None, camera_id=0, display=False,
                 debug=False, mirror_actions=False, stale_after=0.5):
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self.stale_after = stale_after
        self._closed = False
        pose_model = resolve_model("pose", pose_model)
        hand_model = resolve_model("hand", hand_model)
        super().__init__(camera_id=camera_id, debug=debug, mirror_actions=mirror_actions,
                         pose_model=pose_model, hand_model=hand_model, display=display)

    def get_frame(self):
        with self.controller_state_lock:
            action = self.get_controller_state()
            if not action or not action.get("engaged") or time.monotonic() - self._last_received > self.stale_after:
                return None
            return DeviceFrame("mediapipe", self._sequence, self._last_received, deepcopy(action))

    def get_controller_state(self):
        with self.controller_state_lock:
            if self._closed or time.monotonic() - self._last_received > self.stale_after:
                return None
            return deepcopy(super().get_controller_state())

    def close(self):
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        thread = getattr(self, "pose_thread", None)
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                self._closed = False
                raise RuntimeError("MediaPipe camera worker did not stop")
        for name in ("cap", "pose", "hands"):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.release() if name == "cap" else resource.close()
        if getattr(self, "display", False):
            import cv2
            cv2.destroyAllWindows()

    stop = close
    cleanup = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
