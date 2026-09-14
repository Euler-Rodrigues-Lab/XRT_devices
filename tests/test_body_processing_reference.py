"""Ground-truth action fixtures and active upper/lower landmark dependencies."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from xrt_devices.processing import (
    _get_body_frame, _get_lower_body_frame, _get_body_centric_coordinates,
    _calc_gripper_state, bones_to_action,
)
from xrt_devices.schemas.body_pose import Bone


FIXTURE = Path(__file__).parent / 'fixtures' / 'body_processing_reference.npz'


def input_bones(kind='iobt'):
    with np.load(FIXTURE) as data:
        return [Bone(int(i), p.copy(), q.copy()) for i, p, q in zip(
            data[kind+'/input_ids'], data[kind+'/input_positions'], data[kind+'/input_quaternions'])]


def flatten(value, prefix=''):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flatten(item, prefix+key+'/').items()}
    return {} if value is None else {prefix.rstrip('/'): np.asarray(value)}


@pytest.mark.parametrize('kind', ['iobt', 'seed'])
def test_complete_action_matches_original_client(kind):
    actual = flatten(bones_to_action(input_bones(kind)))
    with np.load(FIXTURE) as data:
        expected = {k[len(kind)+1:]: data[k] for k in data.files
                    if k.startswith(kind+'/') and not k.startswith(kind+'/input_')}
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0, err_msg=key)


def anchors():
    return {10: np.array([0., .25, 1.5]), 15: np.array([0., -.25, 1.5]),
            3: np.array([0., 0., 1.]), 70: np.array([0., .1, .9]),
            77: np.array([0., -.1, .9])}


def test_upper_frame_only_needs_arm_upper_and_spine_middle():
    points = anchors()
    origin, rotation = _get_body_frame({k: points[k] for k in (10, 15, 3)})
    np.testing.assert_allclose(origin, [0., 0., 1.5])
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    # These were read by the original function but never used in its returned frame.
    for bid in (1, 4, 8, 13, 70, 77):
        points[bid] = np.array([20., -30., 40.])
    changed = _get_body_frame(points)
    np.testing.assert_array_equal(changed[0], origin)
    np.testing.assert_array_equal(changed[1], rotation)
    # SpineMiddle actually contributes: displacing it tilts the upper frame.
    points[3] = np.array([.2, 0., 1.])
    assert not np.allclose(_get_body_frame(points)[1], rotation)


def test_upper_torso_yaw_is_independent_of_hips():
    points = anchors()
    yaw = Rotation.from_euler('z', .6).as_matrix()
    for bid in (10, 15):
        points[bid] = yaw @ points[bid]
    upper = _get_body_frame(points)[1]
    lower = _get_lower_body_frame(points)[1]
    np.testing.assert_allclose(upper, yaw, atol=1e-12)
    np.testing.assert_allclose(lower, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(lower.T @ upper, yaw, atol=1e-12)
    # Rotate hips independently: upper frame is unchanged, lower follows hips.
    for bid in (70, 77):
        points[bid] = yaw.T @ points[bid]
    np.testing.assert_allclose(_get_body_frame(points)[1], upper, atol=1e-12)
    np.testing.assert_allclose(_get_lower_body_frame(points)[1], yaw.T, atol=1e-12)


def test_arm_coordinates_do_not_depend_on_clavicles_or_hips():
    bones = input_bones()
    expected = _get_body_centric_coordinates(bones)
    reduced = [b for b in bones if b.id not in (1, 4, 8, 13, 70, 77)]
    actual = _get_body_centric_coordinates(reduced)
    for key, value in flatten(expected).items():
        np.testing.assert_allclose(flatten(actual)[key], value, atol=1e-12, rtol=0)


@pytest.mark.parametrize('missing', [10, 15, 3])
def test_missing_active_upper_anchor(missing):
    points = anchors()
    points.pop(missing)
    assert _get_body_frame(points) == (None, None)


@pytest.mark.parametrize('missing', [70, 77, 3])
def test_missing_active_lower_anchor(missing):
    points = anchors()
    points.pop(missing)
    assert _get_lower_body_frame(points) == (None, None)


def test_old_hip_upper_frame_is_rejected():
    with pytest.raises(ValueError, match='upper_arms'):
        _get_body_frame(anchors(), body_frame='hips')
    with pytest.raises(ValueError, match='upper_arms'):
        bones_to_action(input_bones(), body_frame='hips')


def test_gripper_sign_matches_original_threshold():
    for gap, expected in ((.1, -1), (.05, 1), (.01, 1)):
        left, right, left_gap, right_gap = _calc_gripper_state({
            23: np.zeros(3), 28: np.array([gap, 0., 0.]),
            49: np.zeros(3), 54: np.array([gap, 0., 0.])})
        assert left[0] == right[0] == expected
        assert left_gap == right_gap == gap


def test_live_packet_and_csv_default_match_ground_truth(tmp_path):
    pytest.importorskip('geo_kin_core')
    import struct
    from xrt_devices.xr_teleop_device import XRDevice
    from xrt_devices.schemas.body_pose import deserialize_pose_data
    from xrt_devices.recording.writer import CSVRecorder
    from xrt_devices.integrations.geo_kin import OfflineCSVAdapter
    bones = input_bones()
    packet = struct.pack('<i', len(bones)) + b''.join(
        struct.pack('<i7f', b.id, *b.position, *b.rotation) for b in bones)
    decoded = deserialize_pose_data(packet)
    positions = {b.id: np.asarray(b.position) for b in decoded}
    expected_origin = (positions[10] + positions[15]) / 2
    live = XRDevice().process_packet(packet).action
    np.testing.assert_allclose(live['p_world_upper_body'], expected_origin, atol=1e-12)
    path = tmp_path/'capture.csv'
    with CSVRecorder(path) as recorder:
        recorder.write(0., decoded)
    frame = OfflineCSVAdapter(path).frame_at_time(0.)
    np.testing.assert_allclose(frame.p_world_upper_body, expected_origin, atol=1e-12)
    np.testing.assert_allclose(frame.R_world_upper_body, live['R_world_upper_body'], atol=1e-12)
