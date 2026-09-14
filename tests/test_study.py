import csv
import json
import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest

from xrt_devices.study import StudyXRDevice
from xrt_devices.study_process import WebRTCServerProxy


def packet():
    rng = np.random.default_rng(17)
    return struct.pack('<i', 84) + b''.join(
        struct.pack('<i7f', i, *rng.normal(size=3), 0, 0, 0, 1) for i in range(84))


def test_study_recording_takes_actions_tags_and_custom_processing(tmp_path):
    from xrt_devices.recording.csv_reader import CSVDataReader
    def process(bones, tags):
        assert 7 in tags
        return {'right_sew': np.arange(18), 'left_gripper_val': .5}
    device = StudyXRDevice(process_bones_to_action_fn=process)
    device._apriltag(struct.pack('<ii7f', 1, 7, 1, 2, 3, 0, 0, 0, 1))
    for take in ('one', 'two'):
        directory = tmp_path/take
        device.configure_recording(record_data=True, output_dir=directory, started=False)
        device.process_packet(packet())
        device.configure_recording(record_data=False, output_dir=directory, started=False)
        path, = directory.glob('*.csv')
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len([r for r in rows if r['data_type']=='bone']) == 84
        assert {r['id'] for r in rows if r['data_type']=='action'} >= {'right_sew', 'left_gripper_val'}
        reader = CSVDataReader(path, loop=False, apriltag_quaternion_mode_override='internal')
        np.testing.assert_allclose(reader.get_tags_at_time(0)[7]['position'], [3, -1, 2])
        assert (directory/'csv_start_time.txt').is_file()
    device.close()


def test_telemetry_clock_domains_validation_and_disconnect():
    device = StudyXRDevice()
    sent = []
    channel = SimpleNamespace(readyState='open')
    device.server = SimpleNamespace(channels={1: {'unity_cmds': channel}},
                                    send=lambda ch, data: sent.append(json.loads(data)))
    device._connection_changed('connected')
    device._feedback_tick()
    ping, = sent
    device._event(json.dumps(dict(ping, command='latency_pong', t1_ms=900000, t2_ms=900000)))
    assert 0 <= device.network_rtt_ms() < 100
    device._event(json.dumps({'command':'video_latency', 'net_ms':float('nan')}))
    assert device.video_latency() is None
    for _ in range(300):
        device._event(json.dumps({'command':'video_latency', 'net_ms':3}))
    assert device.video_latency()['net_ms'] == 3
    assert device.poll_unity_state() == []
    device._event(json.dumps({'command':'toggle', 'enabled':True}))
    device._connection_changed('disconnected')
    assert device.poll_unity_state() == []
    assert device.video_latency() is None
    assert device.network_rtt_ms() is None
    assert not device.unity_cmds_ready()


def test_disconnect_rotates_recording_and_feedback_is_latest(tmp_path):
    device = StudyXRDevice(record_data=True, output_dir=tmp_path)
    device._connection_changed('connected')
    device.process_packet(packet())
    first = device.output_file
    device._connection_changed('disconnected')
    device._worker_tick()
    assert not device.started
    assert device.record_data
    device._connection_changed('connected')
    device.process_packet(packet())
    assert device.output_file != first
    for n in range(1000):
        device.send_haptics({'value':n})
    assert len(device._stream_feedback) == 1
    assert json.loads(device._stream_feedback['haptics']) == {'value':999}
    device.close()
    assert len(list(tmp_path.glob('*.csv'))) == 2


def test_study_proxy_start_failure_cleans_up():
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        device = WebRTCServerProxy(host='127.0.0.1', port=listener.getsockname()[1])
        with pytest.raises(RuntimeError, match='did not start|process failed|process exited'):
            device.start()
        assert not device.worker_alive
        assert device.get_controller_state() is None
        assert not device.is_connected
