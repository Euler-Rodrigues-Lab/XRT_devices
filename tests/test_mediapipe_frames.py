from types import SimpleNamespace as NS
import time

import numpy as np

from xrt_devices._mediapipe_processing import MediaPipeTeleopDevice
from xrt_devices._mediapipe_tasks import PoseLandmark, HandLandmark


def landmark(x=0., y=0., z=0., visibility=1.):
    return NS(x=x, y=y, z=z, visibility=visibility)


def setup_pose():
    device = MediaPipeTeleopDevice.__new__(MediaPipeTeleopDevice)
    device.mp_pose = NS(PoseLandmark=PoseLandmark)
    device.mp_hands = NS(HandLandmark=HandLandmark)
    device.R_std_cam = np.eye(3)
    world = [landmark() for _ in range(33)]
    pixels = [landmark(.5, .5) for _ in range(33)]
    for i, x, y in [(11,.2,0), (12,-.2,0), (13,.3,.2), (14,-.3,.2),
                    (15,.4,.3), (16,-.4,.3), (23,.15,.5), (24,-.15,.5)]:
        world[i] = landmark(x,y)
        pixels[i] = landmark(.5+x,y+.2)
    return device, NS(pose_world_landmarks=NS(landmark=world),
                      pose_landmarks=NS(landmark=pixels))


def test_hips_offscreen_do_not_tilt_frame():
    device, pose = setup_pose()
    expected = device._get_body_centric_coordinates(pose)
    for i in (23,24):
        pose.pose_landmarks.landmark[i].y = 1.5
        pose.pose_world_landmarks.landmark[i] = landmark(10,2,9)
    actual = device._get_body_centric_coordinates(pose)
    assert actual['body_frame']['hips_estimated']
    for side in ('left','right'):
        np.testing.assert_allclose(actual[side]['W'], expected[side]['W'])


def test_body_frame_is_rotation_with_skewed_hips():
    device, pose = setup_pose()
    pose.pose_world_landmarks.landmark[23].x += .3
    body = device._get_body_centric_coordinates(pose)['body_frame']
    rotation = body['R_world_body']
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1)


def test_hand_assignment_uses_pose_not_classifier_and_preserves_metric_offsets():
    device, pose = setup_pose()
    body = device._get_body_centric_coordinates(pose)
    # Deliberately wrong/duplicate labels, reverse detection order.
    hands = NS(multi_hand_landmarks=[], multi_hand_world_landmarks=[], multi_handedness=[])
    for side in ('right','left'):
        x,y = body[side]['wrist_image']
        hands.multi_hand_landmarks.append(NS(landmark=[landmark(x,y)] * 21))
        hands.multi_hand_world_landmarks.append(NS(landmark=[landmark(9+i*.01,8,7) for i in range(21)]))
        hands.multi_handedness.append(NS(classification=[NS(label='Left',score=.99)]))
    result = device._get_hand_centric_coordinates(hands, body)
    for side in ('left','right'):
        np.testing.assert_allclose(result[side]['landmarks']['wrist'], body[side]['W'])
        assert np.isclose(np.linalg.norm(result[side]['landmarks']['thumb_cmc']-body[side]['W']), .01)
    hands.multi_hand_landmarks.pop()
    hands.multi_hand_world_landmarks.pop()
    hands.multi_handedness.pop()
    assert set(device._get_hand_centric_coordinates(hands, body)) == {'right'}


def test_low_visibility_hips_and_degenerate_shoulders():
    device, pose = setup_pose()
    pose.pose_landmarks.landmark[23].visibility = .1
    assert device._get_body_centric_coordinates(pose)['body_frame']['hips_estimated']
    pose.pose_world_landmarks.landmark[11] = pose.pose_world_landmarks.landmark[12]
    assert device._get_body_centric_coordinates(pose) is None


def test_wrist_dropout_hold_changes_basis_and_expires():
    device, _ = setup_pose()
    device.human_sew_poses = {'R_world_body': np.eye(3)}
    device._last_received = time.monotonic()
    measured = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    device._compute_wrist_rotation_from_hand = lambda *args: measured
    missing = {'landmarks': None, 'confidence': 0}
    assert device._tracked_wrist_rotation('left', missing) is None
    np.testing.assert_allclose(device._tracked_wrist_rotation('left',
                               {'landmarks': {}, 'confidence': 1}), measured)
    device.human_sew_poses['R_world_body'] = measured
    np.testing.assert_allclose(device._tracked_wrist_rotation('left', missing), np.eye(3))
    assert device._tracked_wrist_rotation('right', missing) is None
    device._wrist_rotation_history['left'] = (measured, time.monotonic()-1)
    assert device._tracked_wrist_rotation('left', missing) is None


def test_finger_payload_includes_solver_thumb_alias():
    device, _ = setup_pose()
    device.debug = False
    device._compute_wrist_rotation_from_hand = lambda *args: np.eye(3)
    names = [name.lower() for name in HandLandmark.__members__]
    points = {name: np.array([i*.01, .02, .03]) for i, name in enumerate(names)}
    result = device._compute_finger_positions_wrist_frame(
        'right', {'landmarks': points, 'confidence': 1}, {'W': points['wrist']})
    np.testing.assert_allclose(result['thumb']['thumb_pip'], result['thumb']['thumb_ip'])
    assert result['thumb']['thumb_pip'] is not result['thumb']['thumb_ip']
