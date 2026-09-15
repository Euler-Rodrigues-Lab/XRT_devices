"""Analytic BVH kinematics and public motion-source playback contracts."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from xrt_devices.recording.bvh_reader import BVHDataReader, BVHParseError


def write_bvh(tmp_path, hierarchy, rows, frame_time=.5):
    path = tmp_path / "motion.bvh"
    path.write_text(f"HIERARCHY\n{hierarchy}\nMOTION\nFrames: {len(rows)}\n"
                    f"Frame Time: {frame_time}\n" + "\n".join(rows) + "\n")
    return path


SIMPLE = """ROOT Hips {
 OFFSET 10 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftArm {
  OFFSET 100 0 0
  CHANNELS 3 Yrotation Zrotation Xrotation
  JOINT LeftHand {
   OFFSET 0 100 0
   CHANNELS 0
   End Site { OFFSET 0 10 0 }
  }
 }
}"""


def test_fk_units_axes_and_noncommuting_channels(tmp_path):
    path = write_bvh(tmp_path, SIMPLE, ["5 20 30 90 0 0 90 0 0"])
    reader = BVHDataReader(path)
    poses = reader.transforms_at_time(0)
    # Root at (15,20,30) cm -> (0.15,-0.30,0.20) m.
    np.testing.assert_allclose(poses[0, :3, 3], [.15, -.3, .2])
    np.testing.assert_allclose(poses[1, :3, 3], [.15, -.3, 1.2], atol=1e-14)
    np.testing.assert_allclose(poses[2, :3, 3], [-.85, -.3, 1.2], atol=1e-14)
    expected = reader.world_rotation @ Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    expected = expected @ Rotation.from_euler("Y", 90, degrees=True).as_matrix()
    np.testing.assert_allclose(poses[2, :3, :3], expected, atol=1e-14)
    bones = reader.get_bones_at_time(0)
    assert [b.id for b in bones] == [1, 10, 19]
    np.testing.assert_allclose(Rotation.from_quat(bones[-1].rotation).as_matrix(), expected, atol=1e-14)


def test_interleaved_channels_and_custom_convention(tmp_path):
    path = write_bvh(tmp_path, "ROOT Pelvis { OFFSET 0 0 0 CHANNELS 2 Zrotation Xposition }", ["90 2"])
    reader = BVHDataReader(path, unit_scale=1, world_rotation=np.eye(3), joint_mapping={"Pelvis": 1})
    np.testing.assert_allclose(reader.get_bones_at_time(0)[0].position, [0, 2, 0], atol=1e-14)


def test_sampling_speed_loop_and_end(tmp_path):
    path = write_bvh(tmp_path, SIMPLE, ["0 0 0 0 0 0 0 0 0", "100 0 0 0 0 0 0 0 0"])
    reader = BVHDataReader(path, playback_speed=2, loop=False)
    assert reader.get_duration() == 1
    assert reader.get_bones_at_time(.249)[0].position[0] == pytest.approx(.1)
    assert reader.get_bones_at_time(.25)[0].position[0] == pytest.approx(1.1)
    assert reader.get_bones_at_time(.499) is not None
    assert reader.get_bones_at_time(.5) is None
    assert reader.get_action_and_bones_at_time(.5) == (None, None)
    reader.loop = True
    assert reader.get_bones_at_time(.5)[0].position[0] == pytest.approx(.1)
    for invalid in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="elapsed_time"):
            reader.get_bones_at_time(invalid)


@pytest.fixture
def humanoid(tmp_path):
    # Straight arms in a nondegenerate standing T pose, centimetres, Y up.
    def joint(name, xyz, children="", channels="0"):
        return f"JOINT {name} {{ OFFSET {xyz} CHANNELS {channels} {children} }}"
    left = joint("LeftShoulder", "10 10 0", joint("LeftArm", "10 0 0",
           joint("LeftForeArm", "20 0 0", joint("LeftHand", "20 0 0", channels="3 Zrotation Xrotation Yrotation"))))
    right = joint("RightShoulder", "-10 10 0", joint("RightArm", "-10 0 0",
            joint("RightForeArm", "-20 0 0", joint("RightHand", "-20 0 0"))))
    spine = joint("Spine", "0 10 0", joint("Spine1", "0 10 0", left + right + joint("Head", "0 20 0")))
    legs = "".join(joint(f"{side}UpLeg", f"{x} 0 0", joint(f"{side}Leg", "0 -40 0",
                   joint(f"{side}Foot", "0 -40 0"))) for side, x in (("Left", 10), ("Right", -10)))
    hierarchy = f"ROOT Hips {{ OFFSET 0 100 0 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation {spine}{legs} }}"
    return write_bvh(tmp_path, hierarchy, ["0 0 0 0 0 0 0 0 0", "100 0 0 0 0 0 90 0 0"])


def test_actions_body_wrist_legs_and_skeleton(humanoid):
    reader = BVHDataReader(humanoid, loop=False)
    action = reader.get_action_at_time(0)
    np.testing.assert_allclose(action["left_sew"][:9], [0, .2, 0, 0, .4, 0, 0, .6, 0], atol=1e-14)
    np.testing.assert_allclose(action["right_sew"][:9], [0, -.2, 0, 0, -.4, 0, 0, -.6, 0], atol=1e-14)
    np.testing.assert_allclose(action["R_lower_upper"], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(action["left_hka"]["A"], [0, .1, -.8], atol=1e-14)
    assert "left_fingers" not in action and "left_gripper_val" not in action
    assert action["skeleton"]["parents"][0] == -1
    moving = reader.get_action_at_time(.5)
    np.testing.assert_allclose(moving["left_sew"][:9], action["left_sew"][:9], atol=1e-14)
    r0 = action["left_sew"][9:].reshape(3, 3)
    r1 = moving["left_sew"][9:].reshape(3, 3)
    np.testing.assert_allclose(r1, r0 @ Rotation.from_euler("Z", 90, degrees=True).as_matrix(), atol=1e-14)
    offset = Rotation.from_euler("X", 30, degrees=True).as_matrix()
    adjusted = BVHDataReader(humanoid, wrist_orientation_offsets={"left": offset}).get_action_at_time(0)
    np.testing.assert_allclose(adjusted["left_sew"][9:].reshape(3, 3), r0 @ offset, atol=1e-14)
    with pytest.raises(ValueError, match="upper_arms"):
        BVHDataReader(humanoid, body_frame="hips")


def test_typed_adapter_and_factory(humanoid):
    pytest.importorskip("geo_kin_core")
    from xrt_devices.integrations.geo_kin import OfflineBVHAdapter, open_motion_source
    source = open_motion_source(bvh_file=humanoid, loop=False, unit_scale=.01)
    assert isinstance(source, OfflineBVHAdapter)
    frame, bones = source.get_frame_at_time(0)
    np.testing.assert_allclose(frame.left_sew.S, [0, .2, 0], atol=1e-14)
    assert frame.skeleton["names"][0] == "Hips"
    assert frame.left_fingers is None
    assert frame.left_hka is not None
    assert bones and source.duration == 1
    assert source.get_frame_at_time(1) == (None, None)
    assert "motion.bvh" in source.describe()
    for kwargs in ({}, {"frames": humanoid, "bvh_file": humanoid}, {"csv_file": humanoid, "bvh_file": humanoid}):
        with pytest.raises(ValueError, match="exactly one"):
            open_motion_source(**kwargs)


@pytest.mark.parametrize("replacement", [
    ("CHANNELS 6", "CHANNELS 7"),
    ("Xposition Yposition", "Xposition Xposition"),
    ("OFFSET 100", "OFFSET nan"),
    ("Frames: 1", "Frames: 2"),
    ("Frame Time: 0.5", "Frame Time: 0"),
    ("Frame Time: 0.5", "Frame Time: nan"),
    ("0 0 0 0 0 0 0 0 0", "0 0 0"),
    ("0 0 0 0 0 0 0 0 0", "nan 0 0 0 0 0 0 0 0"),
    ("JOINT LeftHand", "JOINT LeftArm"),
    ("MOTION", ""),
])
def test_reject_malformed(tmp_path, replacement):
    path = write_bvh(tmp_path, SIMPLE, ["0 0 0 0 0 0 0 0 0"])
    path.write_text(path.read_text().replace(*replacement))
    with pytest.raises(BVHParseError):
        BVHDataReader(path)


@pytest.mark.parametrize("kwargs", [
    {"unit_scale": 0}, {"playback_speed": -1}, {"playback_speed": np.inf},
    {"world_rotation": np.diag([-1, 1, 1])}, {"body_frame": "invalid"},
    {"joint_mapping": {"Hips": 84}}, {"joint_mapping": {"Hips": 1, "LeftArm": 1}},
    {"wrist_orientation_offsets": {"left": np.zeros((3, 3))}},
])
def test_reject_invalid_options(tmp_path, kwargs):
    path = write_bvh(tmp_path, SIMPLE, ["0 0 0 0 0 0 0 0 0"])
    with pytest.raises(ValueError):
        BVHDataReader(path, **kwargs)


def test_missing_required_and_degenerate_body(tmp_path, humanoid):
    original = humanoid.read_text()
    path = write_bvh(tmp_path, SIMPLE, ["0 0 0 0 0 0 0 0 0"])
    with pytest.raises(ValueError, match="missing required"):
        BVHDataReader(path).get_action_at_time(0)
    # Flatten shoulder and arm lateral offsets to collapse both shoulders.
    text = original.replace("OFFSET 10 10 0", "OFFSET 0 10 0")
    text = text.replace("OFFSET -10 10 0", "OFFSET 0 10 0")
    text = text.replace("OFFSET 10 0 0", "OFFSET 0 0 0").replace("OFFSET -10 0 0", "OFFSET 0 0 0")
    humanoid.write_text(text)
    assert BVHDataReader(humanoid).get_action_at_time(0) is None


def test_rotation_channel_order(tmp_path):
    path = write_bvh(tmp_path, SIMPLE, ["0 0 0 30 40 50 10 20 60"])
    reader = BVHDataReader(path, world_rotation=np.eye(3))
    transforms = reader.transforms_at_time(0)
    def rotate(axis, angle):
        return Rotation.from_euler(axis, angle, degrees=True).as_matrix()
    root = rotate("Z", 30) @ rotate("X", 40) @ rotate("Y", 50)
    child = root @ rotate("Y", 10) @ rotate("Z", 20) @ rotate("X", 60)
    np.testing.assert_allclose(transforms[0, :3, :3], root, atol=1e-14)
    np.testing.assert_allclose(transforms[1, :3, :3], child, atol=1e-14)
    # Parent rotations act on offsets; a joint's own rotation does not move its origin.
    np.testing.assert_allclose(transforms[1, :3, 3], [.1, 0, 0] + root @ [1., 0, 0], atol=1e-14)


def test_partial_mapping_preserves_visualization_without_fabricated_outputs(humanoid):
    mapping = {"Spine1": 3, "LeftArm": 10, "LeftForeArm": 11, "LeftHand": 19,
               "RightArm": 15, "RightForeArm": 16, "RightHand": 45}
    reader = BVHDataReader(humanoid, joint_mapping=mapping)
    action, bones = reader.get_action_and_bones_at_time(0)
    assert len(bones) == 7
    assert "Hips" in action["skeleton"]["names"]
    assert len(action["skeleton"]["names"]) > len(bones)
    assert "R_lower_upper" not in action
    assert "left_hka" not in action
    assert "head_rotation" not in action
