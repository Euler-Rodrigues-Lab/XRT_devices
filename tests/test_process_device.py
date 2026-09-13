import asyncio
import multiprocessing as mp
import os
import socket
import struct
import time

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


def test_isolated_webrtc_pose_and_shutdown():
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription
    from httpx import AsyncClient
    # Reserve an ephemeral port; close immediately before server startup.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    device = XRDeviceProcess(host="127.0.0.1", port=port, stale_after=2)
    try:
        device.start()
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
            finally:
                await client.close()
        asyncio.run(exercise())
    finally:
        device.close()
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
