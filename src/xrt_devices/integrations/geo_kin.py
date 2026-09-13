"""Public XRT device -> geo_kin_core frame adapter."""
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import time
from geo_kin_core.types import RetargetFrame, SEWPose

# Keys copied verbatim into RetargetFrame.extras (device/base-alignment data
# with no typed field yet).
_EXTRA_KEYS = (
    "body_center",
    "ankle_to_body",
    "tags",
    "left_finger_tip_centroid",
    "right_finger_tip_centroid",
    "left_finger_mcp_centroid",
    "right_finger_mcp_centroid",
    "head",
)


def _opt_array(value, shape=None):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _opt_sew(flat18) -> Optional[SEWPose]:
    if flat18 is None:
        return None
    return SEWPose.from_flat18(np.asarray(flat18, dtype=float).reshape(18))


def action_to_retarget_frame(action: dict) -> Optional[RetargetFrame]:
    """Convert a legacy XR action dict into a geo_kin_core RetargetFrame.

    Includes ``R_lower_upper`` so the 3-DOF waist (torso) solve works — the
    old upper-body hw demo stubbed it to identity; the new session actually
    consumes it. Returns None for an empty/None action (device not streaming).
    """
    if not action or ("engaged" in action and not bool(action["engaged"])):
        return None

    extras = {k: action[k] for k in _EXTRA_KEYS if action.get(k) is not None}

    return RetargetFrame(
        left_sew=_opt_sew(action.get("left_sew")),
        right_sew=_opt_sew(action.get("right_sew")),
        R_world_upper_body=_opt_array(action.get("R_world_upper_body"), (3, 3)),
        p_world_upper_body=_opt_array(action.get("p_world_upper_body"), (3,)),
        head_rotation=_opt_array(action.get("head_rotation"), (3, 3)),
        R_lower_upper=_opt_array(action.get("R_lower_upper"), (3, 3)),
        left_fingers=action.get("left_fingers"),
        right_fingers=action.get("right_fingers"),
        skeleton=action.get("skeleton"),
        # Leg keypoints arrive as per-leg dicts (H/K/A + ankle_rot + world
        # positions), not arrays — pass them through untouched for future leg
        # retargeting rather than coercing them to float arrays.
        left_hka=action.get("left_hka"),
        right_hka=action.get("right_hka"),
        left_gripper_val=action.get("left_gripper_val"),
        right_gripper_val=action.get("right_gripper_val"),
        extras=extras,
    )


class XRDeviceAdapter:
    """Start the public XR device; retain recorder ownership for reliable cleanup."""

    def __init__(self, *, record_data=False, output_dir="recordings", process_isolated=True,
                 diagnostics=False, **device_kwargs):
        from xrt_devices import XRDevice
        self.recorder = None
        self._diagnostics = diagnostics
        self._report_time = time.monotonic()
        self._reads = self._age_max = 0
        path = None
        if record_data:
            from xrt_devices.recording.writer import CSVRecorder
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            path = output / f"body_pose_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
            if not process_isolated:
                self.recorder = CSVRecorder(path)
            print(f"Recording XR bones to {path}")
        try:
            if process_isolated:
                from xrt_devices.process_device import XRDeviceProcess
                self.device = XRDeviceProcess(recording_path=path, **device_kwargs)
            else:
                self.device = XRDevice(recorder=self.recorder, **device_kwargs)
            self.device.start()
        except BaseException:
            if self.recorder is not None:
                self.recorder.close()
            raise

    @property
    def is_connected(self):
        return self.device.is_connected

    def get_raw_action(self):
        frame = self.device.get_frame()
        if self._diagnostics:
            now = time.monotonic()
            self._reads += 1
            if frame is not None:
                age = 1000 * (now - frame.received_at)
                self._age_max = max(self._age_max, age)
            if now - self._report_time >= 5:
                stats = self.device.diagnostics() if hasattr(self.device, "diagnostics") else {}
                print(f"[XRT input] reads={self._reads / (now-self._report_time):.1f}Hz "
                      f"max_receive_age={self._age_max:.1f}ms stats={stats}", flush=True)
                self._report_time = now
                self._reads = self._age_max = 0
        return None if frame is None else frame.action

    def get_frame(self):
        return action_to_retarget_frame(self.get_raw_action())

    def cleanup(self):
        try:
            self.device.close()
        finally:
            if self.recorder is not None:
                self.recorder.close()


class MediaPipeDeviceAdapter:
    """Camera tracking -> arm/hand frames (no inferred waist/base motion)."""

    def __init__(self, **kwargs):
        from xrt_devices import MediaPipeTeleopDevice
        self.device = MediaPipeTeleopDevice(**kwargs)

    @property
    def is_connected(self):
        return self.device.get_frame() is not None

    def get_frame(self):
        snapshot = self.device.get_frame()
        return None if snapshot is None else action_to_retarget_frame(snapshot.action)

    def cleanup(self):
        self.device.close()

"""Public CSV/NPZ playback adapters."""
from typing import Optional, Tuple
from geo_kin_core.frames import load_frames
from geo_kin_core.types import RetargetFrame

from ..schemas.skeleton import bones_to_skeleton


class OfflineCSVAdapter:
    """Recorded-CSV playback source with the same output type as the XR device.

    Mirrors :class:`g1_teleop.input.XRDeviceAdapter` so demos can swap a live
    headset for a recording without touching the solve/control path.
    """

    def __init__(self, csv_file, playback_speed: float = 1.0, loop: bool = True,
                 **reader_kwargs):
        """
        Args:
            csv_file: Recorded OpenXR body-pose CSV.
            **reader_kwargs: Forwarded to the public CSVDataReader.
        """
        from xrt_devices.recording.csv_reader import CSVDataReader
        from xrt_devices.processing import bones_to_action

        self._bones_to_action = bones_to_action
        self.reader = CSVDataReader(str(csv_file), playback_speed, loop=loop, **reader_kwargs)
        self.loop = loop

    @property
    def duration(self) -> float:
        """Recording length in seconds (before playback-speed scaling)."""
        return float(self.reader.get_duration())

    def get_bones_at_time(self, elapsed_time: float):
        """Raw bone list at `elapsed_time` (None past the end when loop=False)."""
        return self.reader.get_bones_at_time(elapsed_time)

    def get_frame_at_time(
        self, elapsed_time: float
    ) -> Tuple[Optional[RetargetFrame], Optional[object]]:
        """Return ``(frame, bones)`` at `elapsed_time`.

        ``bones`` is passed through so demos can drive the human-skeleton
        overlay; both are None once a non-looping recording is exhausted.
        """
        bones = self.get_bones_at_time(elapsed_time)
        if bones is None:
            return None, None
        frame = action_to_retarget_frame(self._bones_to_action(bones, self.reader.get_tags_at_time(elapsed_time)))
        if frame is not None:
            # Raw capture skeleton for the overlay (solvers ignore it).
            frame.skeleton = bones_to_skeleton(bones)
        return frame, bones

    def frame_at_time(self, elapsed_time: float) -> Optional[RetargetFrame]:
        """Frame at `elapsed_time` (common motion-source interface)."""
        return self.get_frame_at_time(elapsed_time)[0]

    def describe(self) -> str:
        return f"CSV recording {self.reader.csv_file_path if hasattr(self.reader, 'csv_file_path') else ''}".strip()


class FrameStreamSource:
    """Playback of a vendored geo_kin_core frame stream (.npz).

    Same interface as :class:`OfflineCSVAdapter` but with no device
    dependencies at all — this is what the demo uses by default so it runs on
    a clean checkout.
    """

    def __init__(self, path, playback_speed: float = 1.0, loop: bool = True):
        self.stream = load_frames(path)
        self.playback_speed = float(playback_speed)
        self.loop = loop

    @property
    def duration(self) -> float:
        return self.stream.duration

    def frame_at_time(self, elapsed_time: float) -> Optional[RetargetFrame]:
        return self.stream.frame_at_time(
            elapsed_time, loop=self.loop, playback_speed=self.playback_speed)

    def get_frame_at_time(self, elapsed_time: float) -> Tuple[Optional[RetargetFrame], None]:
        return self.frame_at_time(elapsed_time), None

    def describe(self) -> str:
        return (f"frame stream {self.stream.path.name} "
                f"({len(self.stream)} frames @ {self.stream.fps:g}Hz, source: {self.stream.source})")


def open_motion_source(frames=None, csv_file=None, playback_speed: float = 1.0,
                       loop: bool = True):
    """Open a motion source: a frame stream (preferred) or a recorded CSV.

    Exactly one of `frames` / `csv_file` must be given. Frame streams need
    nothing but numpy; CSVs use xrt-devices[recording].
    """
    if (frames is None) == (csv_file is None):
        raise ValueError("open_motion_source: pass exactly one of frames=/csv_file=")
    if frames is not None:
        return FrameStreamSource(frames, playback_speed=playback_speed, loop=loop)
    return OfflineCSVAdapter(csv_file, playback_speed=playback_speed, loop=loop)
