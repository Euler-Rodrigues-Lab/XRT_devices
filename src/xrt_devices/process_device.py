"""Isolate WebRTC/inference conversion from the robot solver's Python GIL.

The private shared-memory mailbox contains only the latest snapshot. No pipe or
Queue feeder can accumulate old poses when the consumer stalls. Pickle is used
only between our own parent/child, never to decode network input.
"""
from copy import deepcopy
import multiprocessing as mp
import pickle
import time
import warnings


class _LatestMailbox:
    def __init__(self, context, capacity=1024 * 1024):
        self.data = context.RawArray("B", capacity)
        self.size = context.RawValue("I", 0)
        self.version = context.RawValue("Q", 0)
        self.lock = context.Lock()

    def put(self, value):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self.data):
            raise ValueError("XRT snapshot exceeds shared mailbox capacity")
        if not self.lock.acquire(timeout=0.01):
            return False
        try:
            self.data[:len(payload)] = payload
            self.size.value = len(payload)
            self.version.value += 1
        finally:
            self.lock.release()
        return True

    def get(self, previous=0):
        if not self.lock.acquire(timeout=0.01):
            return previous, None
        try:
            version = self.version.value
            if version == previous:
                return previous, None
            payload = bytes(self.data[:self.size.value])
        finally:
            self.lock.release()
        return version, pickle.loads(payload)


def _serve(mailbox, stop, options, recording_path):
    from .xr_teleop_device import XRDevice
    from contextlib import ExitStack
    try:
        with ExitStack() as resources:
            recorder = None
            if recording_path is not None:
                from .recording.writer import CSVRecorder
                recorder = resources.enter_context(CSVRecorder(recording_path))
            device = XRDevice(recorder=recorder, **options)
            resources.callback(device.close)
            device.start()
            last_key, last_publish = None, 0.0
            while not stop.is_set():
                frame = device.get_frame()
                now = time.monotonic()
                key = (device.is_connected, None if frame is None else frame.sequence)
                if key != last_key or now - last_publish >= 0.1:
                    mailbox.put(dict(ready=True, connected=device.is_connected,
                                     frame=frame, heartbeat=now,
                                     dropped=device.dropped_frames,
                                     invalid=device.invalid_packets, error=device.last_error))
                    last_key, last_publish = key, now
                stop.wait(0.005)
    except BaseException as exc:
        mailbox.put(dict(ready=False, connected=False, frame=None,
                         heartbeat=time.monotonic(), fatal=f"{type(exc).__name__}: {exc}"))


class XRDeviceProcess:
    """Robot-input facade; transport and conversion run in a spawned process.

    This facade supports the robot adapters' read-only pose/recording contract,
    not the full study feedback API. Direct XRDevice remains available for that.
    """

    def __init__(self, *, recording_path=None, **options):
        self.options = options
        self.recording_path = recording_path
        self.stale_after = options.get("stale_after", 0.25)
        self._context = mp.get_context("spawn")
        self._mailbox = _LatestMailbox(self._context)
        self._stop = self._context.Event()
        self._process = None
        self._version = 0
        self._state = {}

    def _refresh(self):
        self._version, state = self._mailbox.get(self._version)
        if state is not None:
            self._state = state
        if self._state.get("fatal"):
            raise RuntimeError(f"XRT input process failed: {self._state['fatal']}")
        if self._process is not None and not self._process.is_alive():
            raise RuntimeError(f"XRT input process exited ({self._process.exitcode})")

    def start(self):
        if self._process is not None:
            return self
        self._stop.clear()
        self._state = {}
        self._process = self._context.Process(
            target=_serve, args=(self._mailbox, self._stop, self.options, self.recording_path),
            name="xrt-input", daemon=True)
        self._process.start()
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                self._refresh()
                if self._state.get("ready"):
                    return self
                time.sleep(0.01)
            raise RuntimeError("Timed out starting isolated XRT input")
        except BaseException:
            self.close()
            raise

    @property
    def is_connected(self):
        self._refresh()
        return bool(self._state.get("connected") and
                    time.monotonic() - self._state.get("heartbeat", 0) < self.stale_after)

    def get_frame(self):
        self._refresh()
        frame = self._state.get("frame")
        if (not self._state.get("connected") or frame is None or
                time.monotonic() - frame.received_at > self.stale_after):
            return None
        return deepcopy(frame)

    def diagnostics(self):
        self._refresh()
        frame = self._state.get("frame")
        return dict(pid=None if self._process is None else self._process.pid,
                    sequence=None if frame is None else frame.sequence,
                    receive_age_ms=None if frame is None else
                    1000 * (time.monotonic() - frame.received_at),
                    dropped=self._state.get("dropped", 0), invalid=self._state.get("invalid", 0))

    def close(self):
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=12)
            if self._process.is_alive():
                warnings.warn("XRT input shutdown timed out; terminating child. "
                              "Pending recording rows may be lost.", RuntimeWarning)
                self._process.terminate()
                self._process.join(timeout=3)
            if self._process.is_alive():
                raise RuntimeError("XRT input process did not stop")
            self._process.close()
            self._process = None
        self._state = {}
