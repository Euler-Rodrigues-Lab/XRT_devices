"""XRT pose input and feedback, with bounded processing and explicit shutdown."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import inspect
import struct
import queue
import threading
import time

from .processing import bones_to_action
from .schemas.body_pose import deserialize_pose_data
from .schemas.skeleton import bones_to_skeleton
from .types import DeviceFrame


class XRDevice:
    """One headset per device. Construction is offline; start() opens the server.

    get_frame() returns a fresh snapshot or None. get_controller_state()/input2action()
    expose the established action dictionary for robot-package adapters.
    """

    _default_process_bones_to_action = staticmethod(bones_to_action)

    def __init__(self, *, host="0.0.0.0", port=8080, stale_after=0.25,
                 process_bones_to_action_fn=None, recorder=None):
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self.host, self.port, self.stale_after = host, port, stale_after
        self._process = process_bones_to_action_fn or bones_to_action
        self._accepts_tags = "tags" in inspect.signature(self._process).parameters
        self._lock = threading.Lock()
        self._pending = queue.Queue(maxsize=1)
        self._events = queue.Queue(maxsize=256)
        self._feedback = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._frame = None
        self._sequence = 0
        self._generation = 0
        self._connected = False
        self._worker = self._server_thread = None
        self.server = None
        self.dropped_frames = self.invalid_packets = 0
        self.event_overflows = 0
        self.last_error = None
        self.telemetry = {}
        self._tags = {}
        self.recorder = recorder  # caller owns its lifetime
        self._recording_origin = None

    def submit_pose(self, message, state=None):
        """Transport callback: retain only the latest raw packet."""
        item = (message, time.monotonic(), self._generation)
        try:
            self._pending.put_nowait(item)
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            self.dropped_frames += 1
            try:
                self._pending.put_nowait(item)
            except queue.Full:
                self.dropped_frames += 1

    def process_packet(self, message, *, received_at=None, generation=None):
        """Decode a packet synchronously; also useful for deterministic offline tests."""
        received_at = time.monotonic() if received_at is None else received_at
        bones = deserialize_pose_data(message)
        if self.recorder is not None:
            if self._recording_origin is None:
                self._recording_origin = received_at
            self.recorder.write(received_at - self._recording_origin, bones)
        with self._lock:
            tags = deepcopy(self._tags)
        action = self._process(bones, tags=tags) if self._accepts_tags else self._process(bones)
        if action is not None:
            action = dict(action, skeleton=bones_to_skeleton(bones))
        with self._lock:
            if generation is not None and generation != self._generation:
                return None
            if action is None:
                self._frame = None
                return None
            action = deepcopy(action)
            action["tags"] = tags
            self._sequence += 1
            self._frame = DeviceFrame("xrt", self._sequence, received_at, action)
            frame = deepcopy(self._frame)
        self._after_frame(bones, frame)
        return frame

    def _after_frame(self, bones, frame):
        """Optional application-neutral recording hook, outside the snapshot lock."""

    def _feedback_tick(self):
        """Optional telemetry, called on the transport event loop."""

    def _worker_tick(self):
        """Optional maintenance outside the transport event loop."""

    def _run(self):
        while not self._stop.is_set():
            try:
                self._worker_tick()
                message, received_at, generation = self._pending.get(timeout=0.05)
                self.process_packet(message, received_at=received_at, generation=generation)
            except queue.Empty:
                continue
            except Exception as exc:
                self.invalid_packets += 1
                self.last_error = str(exc)
                with self._lock:
                    self._frame = None

    def _connection_changed(self, state):
        with self._lock:
            self._connected = state == "connected"
            if not self._connected:
                self._generation += 1
                self._frame = None
                self._tags = {}
        if not self._connected:
            for q in (self._events, self._feedback):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

    def _event(self, message, state=None):
        try:
            event = json.loads(message)
            if not isinstance(event, dict):
                raise ValueError("Unity state must be a JSON object")
            # These are latest-value telemetry, not operator events. Enqueuing
            # periodic reports eventually invalidates tracking on queue overflow.
            if event.get("command") in ("latency_pong", "video_latency"):
                self.telemetry[event["command"]] = event
                return
            self._events.put_nowait(event)
        except queue.Full:
            self.event_overflows += 1
            self._connection_changed("disconnected")
            self.last_error = "operator event queue overflow; tracking invalidated"
        except (ValueError, TypeError) as exc:
            self.last_error = str(exc)

    def _apriltag(self, message, state=None):
        # Unity sends a little-endian count followed by <i7f> tag records,
        # including a four-byte zero count when no tags are visible.
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation
            if len(message) < 4:
                raise ValueError("Truncated AprilTag header")
            count = struct.unpack_from("<i", message)[0]
            if not 0 <= count <= 256 or len(message) != 4 + 32 * count:
                raise ValueError("Invalid AprilTag packet length/count")
            tags = {}
            for offset in range(4, len(message), 32):
                tag_id, x, y, z, qx, qy, qz, qw = struct.unpack_from("<i7f", message, offset)
                position = np.array([z, -x, y])
                quaternion = np.array([-qz, qx, -qy, qw])
                if tag_id in tags or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                    raise ValueError("Duplicate or nonfinite AprilTag record")
                tags[tag_id] = dict(position=position, quaternion=quaternion,
                                    rotation_matrix=Rotation.from_quat(quaternion).as_matrix())
        except (ValueError, TypeError, struct.error) as exc:
            self.invalid_packets += 1
            self.last_error = str(exc)
            return
        with self._lock:
            self._tags = tags

    def start(self):
        if self._worker is not None:
            return self
        from .streaming.webrtc_server import WebRTCServer
        self.server = WebRTCServer(
            host=self.host, port=self.port, connection_callback=self._connection_changed,
            datachannel_handlers={"body_pose": self.submit_pose, "unity_state": self._event,
                                  "apriltag_pose": self._apriltag,
                                  "haptics": lambda message: None,
                                  "motor_stats": lambda message: None,
                                  "unity_cmds": lambda message: None},
        )
        # This input device is single-source; reject a second headset's offer.
        @asynccontextmanager
        async def lifespan(app):
            async def send_loop():
                while True:
                    self._feedback_tick()
                    for _ in range(32):
                        try:
                            label, payload = self._feedback.get_nowait()
                        except queue.Empty:
                            break
                        self.server.send(label, payload)
                    await asyncio.sleep(0.005)
            task = asyncio.create_task(send_loop())
            try:
                async with self.server.lifespan(app):
                    yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.server.app.router.lifespan_context = lifespan
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True, name="xrt-pose")
        self._server_thread = threading.Thread(target=self.server.run, daemon=True, name="xrt-server")
        self._worker.start()
        self._server_thread.start()
        deadline = time.monotonic() + 10
        while self._server_thread.is_alive() and time.monotonic() < deadline:
            if self.server._uvicorn and self.server._uvicorn.started:
                return self
            time.sleep(0.01)
        self.close()
        raise RuntimeError("XRT server did not start; check the bind address and port")

    start_control = start

    @property
    def is_connected(self):
        return self._connected

    def get_frame(self):
        with self._lock:
            if self._frame is None or time.monotonic() - self._frame.received_at > self.stale_after:
                return None
            return deepcopy(self._frame)

    def get_controller_state(self):
        frame = self.get_frame()
        return None if frame is None else frame.action

    input2action = get_controller_state

    def poll_unity_state(self):
        result = []
        while True:
            try:
                result.append(self._events.get_nowait())
            except queue.Empty:
                return result

    def send_feedback(self, channel, payload):
        """Bounded, ordered enqueue. False means disconnected; Full is explicit."""
        if not self.is_connected:
            return False
        if channel not in ("haptics", "motor_stats", "unity_cmds"):
            raise ValueError("unsupported feedback channel")
        self._feedback.put_nowait((channel, payload if isinstance(payload, str) else json.dumps(payload)))
        return True

    def send_haptics(self, payload):
        return self.send_feedback("haptics", payload)

    def send_motor_stats(self, payload):
        return self.send_feedback("motor_stats", payload)

    def send_unity_command(self, payload):
        return self.send_feedback("unity_cmds", payload)

    def close(self):
        self._stop.set()
        if self.server is not None:
            self.server.stop()
        for thread in (self._worker, self._server_thread):
            if thread is not None:
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError(f"{thread.name} did not stop")
        self._worker = self._server_thread = None
        self._connection_changed("closed")
        while not self._pending.empty():
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break

    cleanup = close

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()


XRRTCBodyPoseDevice = XRDevice
