import asyncio
import multiprocessing as mp
import os
import socket
import struct
import time
import json
import csv

from xrt_devices.study import StudyXRDevice
from xrt_devices.study_process import WebRTCServerProxy

import pytest

from xrt_devices.process_device import XRDeviceProcess, _LatestMailbox


def _publish_burst(mailbox):
    for sequence in range(1000):
        mailbox.put({"sequence": sequence})


def test_slow_consumer_receives_latest_not_backlog():
    context = mp.get_context("spawn")
    mailbox = _LatestMailbox(context)
    child = context.Process(target=_publish_burst, args=(mailbox,))
    child.start()
    child.join(timeout=15)
    try:
        assert not child.is_alive()
        assert child.exitcode == 0
        version, value = mailbox.get()
        assert value == {"sequence": 999}
        assert mailbox.get(version) == (version, None)
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        child.close()


@pytest.mark.parametrize("factory", [XRDeviceProcess, StudyXRDevice, WebRTCServerProxy])
def test_isolated_webrtc_pose_and_shutdown(factory, tmp_path):
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription
    from httpx import AsyncClient
    # Reserve an ephemeral port; close immediately before server startup.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    device = factory(host="127.0.0.1", port=port, stale_after=2)
    try:
        device.start()
        if hasattr(device, "diagnostics"):
            assert device.diagnostics()["pid"] != os.getpid()

        async def exercise():
            client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            channel = client.createDataChannel("body_pose", ordered=False, maxRetransmits=0)
            # Full synthetic body, matching conversion test fixtures.
            import numpy as np
            rng = np.random.default_rng(17)
            packet = struct.pack("<i", 84) + b"".join(
                struct.pack("<i7f", i, *rng.normal(size=3), 0, 0, 0, 1)
                for i in range(84))
            study = isinstance(device, (StudyXRDevice, WebRTCServerProxy))
            received = []
            if study:
                state = client.createDataChannel("unity_state")
                for name in ("unity_cmds", "haptics", "motor_stats"):
                    feedback = client.createDataChannel(name)
                    def on_feedback(message, name=name):
                        event = json.loads(message)
                        if event.get('command') == 'latency_ping':
                            state.send(json.dumps(dict(event, command='latency_pong', t1_ms=10000, t2_ms=10000)))
                        else:
                            received.append((name, event))
                    feedback.on('message', on_feedback)
                device.configure_recording(record_data=True, output_dir=str(tmp_path), started=False)
            opened = asyncio.Event()
            @channel.on("open")
            def on_open():
                opened.set()
            try:
                await client.setLocalDescription(await client.createOffer())
                async with AsyncClient(timeout=30) as http:
                    response = await http.post(f"http://127.0.0.1:{port}/offer",
                        json={"sdp": client.localDescription.sdp, "type": "offer"})
                    response.raise_for_status()
                    await client.setRemoteDescription(RTCSessionDescription(**response.json()))
                await asyncio.wait_for(opened.wait(), 15)
                for _ in range(100):
                    channel.send(packet)
                    await asyncio.sleep(.02)
                    frame = device.get_frame()
                    if frame is not None:
                        assert frame.action["skeleton"] is not None
                        assert device.is_connected
                        break
                else:
                    pytest.fail("No pose received from isolated process")
                if study:
                    for _ in range(100):
                        if device.unity_cmds_ready():
                            break
                        await asyncio.sleep(.02)
                    assert device.send_unity_command('{"command":"test"}')
                    assert device.send_haptics('{"strength":0.1}')
                    assert device.send_motor_stats('{"temperature":30}')
                    state.send('{"command":"toggle","enabled":true}')
                    state.send('{"command":"video_latency","decode_ms":2}')
                    events = []
                    for _ in range(200):
                        events.extend(device.poll_unity_state())
                        if len(received) == 3 and events and device.network_rtt_ms() is not None:
                            break
                        await asyncio.sleep(.02)
                    assert {name for name, _ in received} == {'unity_cmds','haptics','motor_stats'}
                    assert events == [{'command':'toggle','enabled':True}]
                    assert device.video_latency()['decode_ms'] == 2
                    assert device.network_rtt_ms() is not None
                    device.configure_recording(record_data=False, output_dir=str(tmp_path), started=False)
                    assert list(tmp_path.glob('*.csv'))
                    for path in tmp_path.glob('*.csv'):
                        with path.open() as stream:
                            assert len(list(csv.DictReader(stream))) >= 84
            finally:
                await client.close()
                for _ in range(100):
                    if not device.is_connected:
                        break
                    await asyncio.sleep(.02)
                assert not device.is_connected
                assert device.get_frame() is None
        asyncio.run(exercise())
        if isinstance(device, (StudyXRDevice, WebRTCServerProxy)):
            asyncio.run(exercise())
    finally:
        device.close()
    if isinstance(device, XRDeviceProcess):
        assert device._process is None
    assert device.get_frame() is None


def test_stale_snapshot_is_not_returned():
    device = XRDeviceProcess(stale_after=.1)
    from xrt_devices.types import DeviceFrame
    device._mailbox.put(dict(connected=True, frame=DeviceFrame("xrt", 1, time.monotonic()-1, {})))
    assert device.get_frame() is None


def test_periodic_telemetry_does_not_fill_operator_queue():
    import json
    from xrt_devices.xr_teleop_device import XRDevice
    device = XRDevice()
    device._connection_changed("connected")
    for n in range(1000):
        device._event(json.dumps({"command": "video_latency", "net_ms": n}))
    assert device.is_connected
    assert device.event_overflows == 0
    assert device.telemetry["video_latency"]["net_ms"] == 999
    assert device.poll_unity_state() == []
