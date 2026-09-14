import asyncio
import csv
import importlib
import json
import struct
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from xrt_devices.schemas.body_pose import deserialize_pose_data
from xrt_devices.xr_teleop_device import XRDevice


def packet(x=1.0):
    return struct.pack('<ii7f', 1, 0, x, 2, 3, 0, 0, 0, 1)


def test_wire_conversion():
    bone = deserialize_pose_data(packet())[0]
    assert bone.position == (3, -1, 2)
    assert bone.rotation == (0, 0, 0, 1)
    assert deserialize_pose_data(packet(), z_up=False)[0].position == (1, 2, 3)


@pytest.mark.parametrize('data', [b'', b'123', struct.pack('<i', -1), packet()[:-1],
                                      packet() + b'x', packet(float('nan'))])
def test_bad_packets(data):
    with pytest.raises(ValueError):
        deserialize_pose_data(data)


def test_offline_snapshot_stale_and_disconnect():
    device = XRDevice(process_bones_to_action_fn=lambda bones: {'position': np.array(bones[0].position)})
    device.process_packet(packet())
    frame = device.get_frame()
    frame.action['position'][0] = 999
    assert device.get_frame().action['position'][0] == 3
    device.process_packet(packet(), received_at=time.monotonic() - 1)
    assert device.get_frame() is None
    generation = device._generation
    device._connection_changed('disconnected')
    assert device.process_packet(packet(), generation=generation) is None
    device.close()
    device.close()


def test_latest_queue_and_ordered_events():
    device = XRDevice()
    device.submit_pose(packet(1))
    device.submit_pose(packet(2))
    assert device.dropped_frames == 1
    assert device._pending.get_nowait()[0] == packet(2)
    device._event('{"reset": true}')
    device._event('{"engage": false}')
    assert device.poll_unity_state() == [{'reset': True}, {'engage': False}]
    assert not device.send_haptics('{}')


def test_base_import_has_no_optional_stack():
    subprocess.run([sys.executable, '-c', '''
import sys
for name in ('torch', 'cv2', 'mediapipe', 'aiortc', 'robosuite', 'geo_kin', 'xr_robot_teleop_server'):
    sys.modules[name] = None
import xrt_devices
from xrt_devices.processing import bones_to_action
from xrt_devices import MediaPipeTeleopDevice, XRDevice
assert XRDevice().get_frame() is None
'''], check=True)


@pytest.mark.parametrize('column,start', [('time_elapsed', 0), ('timestamp', 1700000000)])
def test_csv_absolute_relative_and_end(tmp_path, column, start):
    from xrt_devices.recording.csv_reader import CSVDataReader
    path = tmp_path / 'poses.csv'
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=[column, 'data_type', 'id', 'pos_x', 'pos_y', 'pos_z',
                                                    'rot_x', 'rot_y', 'rot_z', 'rot_w'])
        writer.writeheader()
        for t in (1, 0):  # intentionally unsorted
            writer.writerow(dict(zip(writer.fieldnames, [start+t, 'bone', 0, t, 0, 0, 0, 0, 0, 1])))
    reader = CSVDataReader(path, loop=False)
    assert reader.get_bones_at_time(0)[0].position[0] == 0
    assert reader.get_bones_at_time(1)[0].position[0] == 1
    assert reader.get_bones_at_time(1.1) is None
    reader.loop = True
    assert reader.get_bones_at_time(2)[0].position[0] == 0
    with pytest.raises(ValueError):
        CSVDataReader(path, playback_speed=0)


def test_tasks_result_adapter():
    from xrt_devices._mediapipe_tasks import hand_result, pose_result
    point = NS(x=1, y=2, z=3)
    result = hand_result(NS(hand_landmarks=[[point]], hand_world_landmarks=[[point]],
                            handedness=[[NS(category_name='Left', score=0.9)]]))
    assert result.multi_hand_world_landmarks[0].landmark[0] is point
    assert result.multi_handedness[0].classification[0].label == 'Left'
    assert pose_result(NS(pose_landmarks=[], pose_world_landmarks=[])).pose_landmarks is None


def test_mediapipe_hand_alignment_without_camera():
    from xrt_devices._mediapipe_processing import MediaPipeTeleopDevice
    from xrt_devices._mediapipe_tasks import HandLandmark, hand_result
    device = MediaPipeTeleopDevice.__new__(MediaPipeTeleopDevice)
    device.mp_hands = NS(HandLandmark=HandLandmark)
    points = [NS(x=i*.01, y=i*.02, z=i*.03) for i in range(21)]
    hands = hand_result(NS(hand_landmarks=[points], hand_world_landmarks=[points],
                           handedness=[[NS(category_name='Left', score=.99)]]))
    body = {'body_frame': {'origin': np.zeros(3), 'x_axis': np.eye(3)[:, 0],
                          'y_axis': np.eye(3)[:, 1], 'z_axis': np.eye(3)[:, 2]},
            'right': {'W': np.array([1., 2., 3.]), 'wrist_image': np.array([0., 0.])},
            'left': {'W': np.zeros(3), 'wrist_image': np.array([1., 1.])}}
    result = device._get_hand_centric_coordinates(hands, body)
    np.testing.assert_allclose(result['right']['landmarks']['wrist'], [1, 2, 3])
    assert len(result['right']['landmarks']) == 21


def test_recorder_round_trip_single_frame(tmp_path):
    from xrt_devices.recording.writer import CSVRecorder
    from xrt_devices.recording.csv_reader import CSVDataReader
    path = tmp_path / 'capture.csv'
    with CSVRecorder(path) as writer:
        writer.write(0, deserialize_pose_data(packet()))
    reader = CSVDataReader(path, loop=False)
    np.testing.assert_allclose(reader.get_bones_at_time(0)[0].position, (3, -1, 2))
    assert reader.get_bones_at_time(1) is None
    reader.loop = True
    np.testing.assert_allclose(reader.get_bones_at_time(1)[0].position, (3, -1, 2))
    with pytest.raises(FileExistsError):
        CSVRecorder(path)


def test_server_start_close():
    device = XRDevice(host='127.0.0.1', port=0)
    with device:
        assert device.server._uvicorn.started
        thread = device._server_thread
    assert not thread.is_alive()
    assert device.get_frame() is None


def test_webrtc_client_pose_and_feedback():
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription
    from httpx import ASGITransport, AsyncClient
    from xrt_devices.streaming.webrtc_server import WebRTCServer

    async def exercise():
        received = asyncio.Event()
        feedback = asyncio.Event()
        def on_pose(message, state=None):
            assert deserialize_pose_data(message)[0].position == (3, -1, 2)
            received.set()
        server = WebRTCServer(datachannel_handlers={'body_pose': on_pose})
        client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        body = client.createDataChannel('body_pose', ordered=False, maxRetransmits=0)
        haptics = client.createDataChannel('haptics')
        @body.on('open')
        def opened():
            body.send(packet())
        @haptics.on('message')
        def message(payload):
            assert json.loads(payload) == {'force': 1}
            feedback.set()
        try:
            await client.setLocalDescription(await client.createOffer())
            async with AsyncClient(transport=ASGITransport(app=server.app), base_url='http://localhost') as http:
                response = await http.post('/offer', json={'sdp': client.localDescription.sdp, 'type': 'offer'})
                assert response.status_code == 200
                answer = response.json()
                await client.setRemoteDescription(RTCSessionDescription(**answer))
                await asyncio.wait_for(received.wait(), timeout=15)
                for _ in range(100):
                    if server.send('haptics', '{"force": 1}'):
                        break
                    await asyncio.sleep(.01)
                await asyncio.wait_for(feedback.wait(), timeout=5)
                assert (await http.post('/offer', json={})).status_code == 409
        finally:
            await client.close()
            await asyncio.gather(*(pc.close() for pc in list(server.pcs)))
    asyncio.run(exercise())
