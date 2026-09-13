"""Optional typed-frame integration tested without inference, networking or hardware."""
import numpy as np
import struct
import pytest
pytest.importorskip("geo_kin_core")

from xrt_devices.integrations.geo_kin import (
    action_to_retarget_frame, OfflineCSVAdapter, XRDeviceAdapter, MediaPipeDeviceAdapter,
)
from xrt_devices.processing import bones_to_action
from xrt_devices.schemas.openxr_skeletons import FullBodyBoneId
from xrt_devices.schemas.body_pose import Bone
from xrt_devices.recording.writer import CSVRecorder
from xrt_devices.types import DeviceFrame


@pytest.fixture
def bones():
    rng = np.random.default_rng(17)
    return [Bone(int(b), rng.normal(size=3), (0, 0, 0, 1))
            for b in FullBodyBoneId if 0 <= int(b) < 84]


def test_capture_to_csv_to_typed_frame(tmp_path, bones):
    path = tmp_path / "capture.csv"
    with CSVRecorder(path) as recorder:
        recorder.write(0, bones)
        recorder.write(1, bones)
    source = OfflineCSVAdapter(path, loop=False)
    frame = source.frame_at_time(.5)
    expected = action_to_retarget_frame(bones_to_action(bones))
    np.testing.assert_allclose(frame.left_sew.S, expected.left_sew.S)
    np.testing.assert_allclose(frame.right_sew.W, expected.right_sew.W)
    assert frame.skeleton is not None
    assert len(frame.skeleton["names"]) > 60
    assert "left_finger_mcp_centroid" in frame.extras
    assert source.frame_at_time(2) is None


def test_live_and_replay_have_identical_skeleton(tmp_path, bones):
    from xrt_devices.xr_teleop_device import XRDevice
    from xrt_devices.schemas.body_pose import deserialize_pose_data
    packet = struct.pack("<i", len(bones)) + b"".join(
        struct.pack("<i7f", b.id, *b.position, *b.rotation) for b in bones)
    snapshot = XRDevice().process_packet(packet)
    live = action_to_retarget_frame(snapshot.action)
    path = tmp_path / "same-packet.csv"
    with CSVRecorder(path) as recorder:
        recorder.write(0, deserialize_pose_data(packet))
    replay = OfflineCSVAdapter(path).frame_at_time(0)
    assert live.skeleton["names"] == replay.skeleton["names"]
    np.testing.assert_array_equal(live.skeleton["parents"], replay.skeleton["parents"])
    np.testing.assert_allclose(live.skeleton["positions"], replay.skeleton["positions"])
    np.testing.assert_allclose(live.left_sew.W, replay.left_sew.W)


def test_live_adapter_lifecycle_and_recording(monkeypatch, tmp_path, bones):
    import xrt_devices
    class Device:
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder
            self.is_connected = True
            self.closed = False
        def start(self):
            self.recorder.write(0, bones)
        def get_frame(self):
            return DeviceFrame("xrt", 1, 0, bones_to_action(bones))
        def close(self):
            self.closed = True
    monkeypatch.setattr(xrt_devices, "XRDevice", Device)
    adapter = XRDeviceAdapter(record_data=True, output_dir=tmp_path, process_isolated=False)
    assert adapter.get_frame().right_sew is not None
    adapter.cleanup()
    assert adapter.device.closed
    assert OfflineCSVAdapter(next(tmp_path.glob("*.csv"))).frame_at_time(0) is not None


def test_camera_adapter_and_head_extras(monkeypatch, bones):
    import xrt_devices
    action = bones_to_action(bones)
    action["head"] = np.array([.2, .3])
    class Camera:
        def __init__(self, **kwargs):
            self.closed = False
        def get_frame(self):
            return DeviceFrame("mediapipe", 1, 0, action)
        def close(self):
            self.closed = True
    monkeypatch.setattr(xrt_devices, "MediaPipeTeleopDevice", Camera)
    adapter = MediaPipeDeviceAdapter(pose_model="pose.task", hand_model="hand.task")
    assert adapter.is_connected
    np.testing.assert_allclose(adapter.get_frame().extras["head"], [.2, .3])
    adapter.cleanup()
    assert adapter.device.closed
    action["engaged"] = np.bool_(False)
    assert action_to_retarget_frame(action) is None
