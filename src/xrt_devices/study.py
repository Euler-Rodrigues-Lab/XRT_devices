"""Study transport compatibility: take recording and headset telemetry.

Study UI, haptics policy, robot commands and sensor orchestration stay in the app.
Construction is offline; callers explicitly start and close the device.
"""
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time

from .xr_teleop_device import XRDevice
from .recording.writer import CSVRecorder


class _StateFactory:
    def __init__(self, device):
        self.device = device

    @property
    def instance(self):
        return self.device if self.device.is_connected else None


class StudyXRDevice(XRDevice):
    def __init__(self, env=None, *, record_data=False, output_dir="recordings",
                 ndigits=3, server_data_channels=None, **options):
        if env is not None:
            raise ValueError("Robot environments belong in the application")
        if server_data_channels:
            raise ValueError("Study feedback channels must be created by the headset")
        super().__init__(**options)
        self.state_factory = _StateFactory(self)
        self._record_lock = threading.RLock()
        self.record_data = record_data
        self.output_dir = output_dir
        self.ndigits = ndigits
        self.started = False
        self._take = None
        self._finish_take = threading.Event()
        self._take_origin = None
        self.output_file = None
        self._finished_at = 0.0
        self._health_lock = threading.RLock()
        self._arrivals = []
        self._rtt = None
        self._video = None
        self._ping_id = 0
        self._pings = {}
        self._last_ping = 0.0
        self._stream_feedback = {}

    def _worker_tick(self):
        if self._finish_take.is_set():
            self._finish_take.clear()
            try:
                self.cleanup_recording()
            except Exception:
                self.record_data = False
                raise

    def _after_frame(self, bones, frame):
        self._worker_tick()
        self._finished_at = time.monotonic()
        with self._record_lock:
            if not self.record_data:
                return
            if not self.started or self._take is None:
                self.cleanup_recording()
                output = Path(self.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                self.output_file = str(output / f"body_pose_data_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
                self._take = CSVRecorder(self.output_file, ndigits=self.ndigits)
                self._take_origin = frame.received_at
                epoch = time.time() - (time.monotonic() - frame.received_at)
                (output / 'csv_start_time.txt').write_text(
                    f"Start time (epoch): {epoch}\n"
                    f"Start time (datetime): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))}\n")
                self.started = True
            self._take.write(frame.received_at - self._take_origin, bones, frame.action)

    def configure_recording(self, *, record_data, output_dir, started):
        with self._record_lock:
            if not record_data or not started or output_dir != self.output_dir:
                self.cleanup_recording()
            self.output_dir = output_dir
            self.record_data = bool(record_data)
            self.started = bool(started) and self._take is not None

    def cleanup_recording(self):
        with self._record_lock:
            if self._take is not None:
                self._take.close()
                self._take = None
            self.started = False

    def submit_pose(self, message, state=None):
        with self._health_lock:
            now = time.monotonic()
            self._arrivals.append(now)
            # Bound memory even when nobody polls the study panel.
            self._arrivals = self._arrivals[-2000:]
        super().submit_pose(message, state)

    def _send_latest(self, channel, payload):
        if not self.is_connected:
            return False
        with self._health_lock:
            self._stream_feedback[channel] = payload if isinstance(payload, str) else json.dumps(payload)
        return True

    def send_haptics(self, payload):
        return self._send_latest('haptics', payload)

    def send_motor_stats(self, payload):
        return self._send_latest('motor_stats', payload)

    def _feedback_tick(self):
        with self._health_lock:
            pending, self._stream_feedback = self._stream_feedback, {}
        for channel, payload in pending.items():
            if self.is_connected:
                self.server.send(channel, payload)
        now = time.monotonic()
        if not self.unity_cmds_ready() or now - self._last_ping < 1:
            return
        self._last_ping = now
        with self._health_lock:
            self._ping_id += 1
            self._pings = {k: v for k, v in self._pings.items() if now * 1000 - v < 2000}
            self._pings[self._ping_id] = now * 1000
            payload = dict(type="unity_command", command="latency_ping",
                           ping_id=self._ping_id, t0_ms=now * 1000)
        self.server.send("unity_cmds", json.dumps(payload))

    def _event(self, message, state=None):
        try:
            event = json.loads(message)
            if not isinstance(event, dict):
                raise ValueError("Unity state must be a JSON object")
            command = event.get('command')
            with self._health_lock:
                if command == 'latency_pong':
                    t0 = self._pings.pop(event.get('ping_id'), None)
                    if t0 is None:
                        return
                    processing = float(event['t2_ms']) - float(event['t1_ms'])
                    rtt = time.monotonic() * 1000 - t0 - processing
                    if math.isfinite(rtt) and processing >= 0 and 0 <= rtt <= 2000:
                        self._rtt = rtt if self._rtt is None else .7 * self._rtt + .3 * rtt
                    return
                if command == 'video_latency':
                    video = {k: float(event.get(k, 0)) for k in ('jitter_ms', 'decode_ms', 'net_ms')}
                    if all(math.isfinite(v) and v >= 0 for v in video.values()):
                        self._video = video
                    return
        except (KeyError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            return
        super()._event(message, state)

    def network_rtt_ms(self):
        with self._health_lock:
            return self._rtt

    def video_latency(self):
        with self._health_lock:
            return deepcopy(self._video)

    def link_health(self, min_window_s=3.0):
        with self._health_lock:
            now = time.monotonic()
            self._arrivals = [t for t in self._arrivals if now-t <= min_window_s]
            if len(self._arrivals) < 2:
                return None
            gaps = [b-a for a, b in zip(self._arrivals, self._arrivals[1:])]
            elapsed = max(now-self._arrivals[0], .001)
            return dict(fps=len(gaps)/elapsed, stalls_per_s=sum(g > .05 for g in gaps)/elapsed,
                        max_gap_ms=max(gaps)*1000, rtt_ms=self._rtt)

    def unity_cmds_ready(self):
        return bool(self.is_connected and self.server is not None and any(
            getattr(channels.get('unity_cmds'), 'readyState', None) == 'open'
            for channels in list(self.server.channels.values())))

    def pose_age_s(self):
        with self._lock:
            return None if self._frame is None else max(0, time.monotonic()-self._frame.received_at)

    def get_pose_seq(self):
        with self._lock:
            return self._sequence, self._finished_at

    def get_arrival_t(self):
        with self._lock:
            return 0.0 if self._frame is None else self._frame.received_at

    def _connection_changed(self, state):
        super()._connection_changed(state)
        if state != 'connected':
            with self._health_lock:
                self._arrivals.clear()
                self._rtt = self._video = None
                self._pings.clear()
                self._stream_feedback.clear()
            if state in ('disconnected', 'closed', 'failed'):
                # Preserve the take boundary on disconnect without blocking ICE
                # callbacks while the CSV sink drains. A reconnect opens a new file.
                self._finish_take.set()

    def close(self):
        try:
            super().close()
        finally:
            self.cleanup_recording()

    stop = close
    cleanup = close
