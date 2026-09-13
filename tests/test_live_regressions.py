import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from xrt_devices.xr_teleop_device import XRDevice


def test_binary_apriltags_and_empty_clear():
    device = XRDevice()
    device._apriltag(struct.pack("<ii7f", 1, 42, 1, 2, 3, 0, 0, 0, 1))
    np.testing.assert_allclose(device._tags[42]["position"], [3, -1, 2])
    np.testing.assert_allclose(device._tags[42]["rotation_matrix"], np.eye(3))
    device._apriltag(b"\x00\x00\x00\x00")
    assert device._tags == {}
    assert device.invalid_packets == 0


@pytest.mark.parametrize("packet", [b"", b"bad", struct.pack("<i", -1),
    struct.pack("<i", 1), struct.pack("<ii7f", 1, 0, 0, 0, 0, 0, 0, 0, 0)])
def test_invalid_apriltags_do_not_escape_callback(packet):
    device = XRDevice()
    device._apriltag(packet)
    assert device.invalid_packets == 1
    assert device.last_error


def test_overlay_draws_pose_and_hand():
    pytest.importorskip("cv2")
    from xrt_devices._drawing import draw_landmarks, POSE_CONNECTIONS, HAND_CONNECTIONS
    for connections, count in [(POSE_CONNECTIONS, 33), (HAND_CONNECTIONS, 21)]:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = NS(landmark=[NS(x=0.1 + i * 0.02, y=0.5,
                                   visibility=None, presence=None) for i in range(count)])
        draw_landmarks(image, landmarks, connections)
        assert image.any()
