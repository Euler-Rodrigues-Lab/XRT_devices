"""Process-isolated study transport with latest poses and ordered control RPCs."""
from copy import deepcopy
import queue
import threading
import time

from .process_device import XRDeviceProcess
from .study import _StateFactory


def _serve_study(mailbox, stop, options, commands, replies, events):
    from .study import StudyXRDevice
    device = None
    try:
        device = StudyXRDevice(**options)
        device.start()
        last_key, last_publish = None, 0.0
        while not stop.is_set():
            for _ in range(32):
                try:
                    request_id, method, kwargs = commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    result = getattr(device, method)(**kwargs)
                    replies.put((request_id, result, None), timeout=1)
                except Exception as exc:
                    replies.put((request_id, None, f'{type(exc).__name__}: {exc}'), timeout=1)
            for event in device.poll_unity_state():
                # Never silently drop an operator event. Overflow fails the worker,
                # and the parent invalidates its snapshot via the fatal state.
                events.put_nowait((device._generation, event))
            frame = device.get_frame()
            now = time.monotonic()
            key = (device.is_connected, None if frame is None else frame.sequence)
            if key != last_key or now-last_publish >= .1:
                mailbox.put(dict(ready=True, connected=device.is_connected, frame=frame,
                                 heartbeat=now, generation=device._generation, dropped=device.dropped_frames,
                                 invalid=device.invalid_packets, error=device.last_error,
                                 link_health=device.link_health(), rtt=device.network_rtt_ms(),
                                 video=device.video_latency(), cmds_ready=device.unity_cmds_ready(),
                                 pose_seq=device.get_pose_seq(), arrival=device.get_arrival_t()))
                last_key, last_publish = key, now
            stop.wait(.005)
    except BaseException as exc:
        mailbox.put(dict(ready=False, connected=False, frame=None,
                         heartbeat=time.monotonic(), fatal=f'{type(exc).__name__}: {exc}'))
    finally:
        try:
            if device is not None:
                device.close()
        finally:
            # The parent may stop polling during shutdown; never wait on a
            # feeder flushing stale UI events into a full pipe at process exit.
            replies.cancel_join_thread()
            events.cancel_join_thread()


class WebRTCServerProxy(XRDeviceProcess):
    """Study API over a spawned device; start waits for a listening server.

    Control calls acknowledge child execution, including complete recording drain.
    Pose snapshots never queue behind control messages or a stalled parent.
    """
    def __init__(self, init_msg=None, log_fn=None, **options):
        options = dict(init_msg or {}, **options)
        super().__init__(**options)
        self._log = log_fn or (lambda *_: None)
        self._rpc_lock = threading.Lock()
        self._refresh_lock = threading.RLock()
        self._request_id = 0
        self._commands = self._replies = self._events = None
        self._record_data = bool(options.get('record_data', False))
        self._output_dir = options.get('output_dir', 'recordings')
        self._started = False
        self.state_factory = _StateFactory(self)

    def _refresh(self):
        # Pose, feedback and Qt status threads may all refresh concurrently.
        # Serialize cache updates so an older read cannot overwrite a newer one.
        with self._refresh_lock:
            super()._refresh()

    @property
    def worker_alive(self):
        process = self._process
        try:
            return process is not None and process.is_alive()
        except ValueError:  # process handle was closed concurrently
            return False

    def start(self, timeout=20):
        if self._process is not None:
            return self
        self._stop.clear()
        self._state = {}
        # A fresh mailbox prevents a prior ready/fatal snapshot from winning startup.
        from .process_device import _LatestMailbox
        self._mailbox = _LatestMailbox(self._context)
        self._version = 0
        self._commands = self._context.Queue(maxsize=256)
        self._replies = self._context.Queue(maxsize=256)
        self._events = self._context.Queue(maxsize=256)
        self.options.update(record_data=self._record_data, output_dir=self._output_dir)
        self._process = self._context.Process(target=_serve_study,
            args=(self._mailbox, self._stop, self.options, self._commands, self._replies, self._events),
            name='xrt-study', daemon=True)
        self._process.start()
        try:
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                self._refresh()
                if self._state.get('ready'):
                    return self
                time.sleep(.01)
            raise RuntimeError('Timed out starting study transport')
        except BaseException:
            self.close()
            raise

    def _rpc(self, method, **kwargs):
        if self._process is None:
            raise RuntimeError('Study transport is not started')
        with self._rpc_lock:
            self._refresh()
            self._request_id += 1
            request_id = self._request_id
            self._commands.put((request_id, method, kwargs), timeout=1)
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                self._refresh()
                try:
                    reply_id, result, error = self._replies.get(timeout=.05)
                except queue.Empty:
                    continue
                if reply_id != request_id:
                    continue
                if error:
                    raise RuntimeError(f'Study transport {method}: {error}')
                return result
            raise RuntimeError(f'Timed out waiting for study transport {method}')

    @property
    def is_connected(self):
        try:
            return super().is_connected
        except (RuntimeError, ValueError):
            return False

    def get_frame(self):
        try:
            return super().get_frame()
        except (RuntimeError, ValueError):
            return None

    def get_controller_state(self):
        frame = self.get_frame()
        return None if frame is None else frame.action

    input2action = get_controller_state

    def pose_age_s(self):
        self._refresh()
        frame = self._state.get('frame')
        return None if frame is None else max(0, time.monotonic()-frame.received_at)

    def get_pose_seq(self):
        self._refresh()
        return self._state.get('pose_seq', (0, 0.0))

    def get_arrival_t(self):
        self._refresh()
        return self._state.get('arrival', 0.0)

    def link_health(self, min_window_s=3):
        return deepcopy(self._state.get('link_health')) if self.is_connected else None

    def network_rtt_ms(self):
        return self._state.get('rtt') if self.is_connected else None

    def video_latency(self):
        return deepcopy(self._state.get('video')) if self.is_connected else None

    def unity_cmds_ready(self):
        return bool(self.is_connected and self._state.get('cmds_ready'))

    def poll_unity_state(self):
        connected = self.is_connected
        result = []
        if self._events is not None:
            while True:
                try:
                    generation, event = self._events.get_nowait()
                    if connected and generation == self._state.get('generation'):
                        result.append(event)
                except queue.Empty:
                    break
        return result

    def send_unity_command(self, payload):
        return self._rpc('send_unity_command', payload=payload)

    def send_haptics(self, payload):
        return self._rpc('send_haptics', payload=payload)

    def send_motor_stats(self, payload):
        return self._rpc('send_motor_stats', payload=payload)

    def configure_recording(self, *, record_data, output_dir, started=False):
        if self._process is not None:
            self._rpc('configure_recording', record_data=record_data,
                      output_dir=output_dir, started=started)
        self._record_data, self._output_dir, self._started = bool(record_data), output_dir, bool(started)

    def cleanup_recording(self):
        if self._process is not None:
            self._rpc('configure_recording', record_data=False,
                      output_dir=self._output_dir, started=False)
        self._record_data = self._started = False

    def _push_record(self):
        self.configure_recording(record_data=self._record_data,
                                 output_dir=self._output_dir, started=self._started)

    @property
    def record_data(self):
        return self._record_data

    @record_data.setter
    def record_data(self, value):
        self._record_data = bool(value)
        self._push_record()

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        self._push_record()

    @property
    def started(self):
        return self._started

    @started.setter
    def started(self, value):
        self._started = bool(value)
        self._push_record()

    def close(self):
        try:
            super().close()
        finally:
            for q in (self._commands, self._replies, self._events):
                if q is not None:
                    q.cancel_join_thread()
                    q.close()
            self._commands = self._replies = self._events = None

    stop = close
    cleanup = close
