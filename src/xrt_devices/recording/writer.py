"""Bounded CSV sink compatible with CSVDataReader. Formatting stays off ingestion."""
from copy import deepcopy
import csv
import numpy as np
from pathlib import Path
import queue
import threading


class CSVRecorder:
    def __init__(self, path, *, capacity=256, ndigits=None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.ndigits = ndigits
        self.path = Path(path)
        self._queue = queue.Queue(capacity)
        self._stop = threading.Event()
        self._error = None
        self.dropped_frames = 0
        # Exclusive creation prevents overwriting an existing recording.
        self._stream = self.path.open('x', newline='')
        self._thread = threading.Thread(target=self._run, daemon=True, name='xrt-recorder')
        self._thread.start()

    def write(self, elapsed_time, bones, action=None):
        if self._error is not None:
            raise RuntimeError('CSV writer failed') from self._error
        if self._stop.is_set():
            raise RuntimeError('CSV recorder is closed')
        snapshot = (float(elapsed_time), [(b.id, tuple(b.position), tuple(b.rotation)) for b in bones], deepcopy(action))
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            self.dropped_frames += 1
            return False
        return True

    def _run(self):
        try:
            writer = csv.writer(self._stream)
            writer.writerow(('time_elapsed', 'data_type', 'id', 'pos_x', 'pos_y', 'pos_z',
                             'rot_x', 'rot_y', 'rot_z', 'rot_w'))
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    timestamp, bones, action = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                for bone_id, position, rotation in bones:
                    self._row(writer, timestamp, 'bone', bone_id, (*position, *rotation))
                for tag_id, tag in (action or {}).get('tags', {}).items():
                    self._row(writer, timestamp, 'apriltag', tag_id,
                              (*tag['position'], *tag['quaternion']))
                for key, value in (action or {}).items():
                    if isinstance(value, np.ndarray):
                        flat = value.flatten()
                        position = flat[:3] if len(flat) >= 3 else (0, 0, 0)
                        rotation = [flat[i] if len(flat) > i else 0 for i in (9, 10, 11)]
                        self._row(writer, timestamp, 'action', key, (*position, *rotation, 1))
                    elif key in ('left_gripper_val', 'right_gripper_val'):
                        self._row(writer, timestamp, 'action', key, (float(value), 0, 0, 0, 0, 0, 1))
        except Exception as exc:
            self._error = exc
        finally:
            self._stream.close()

    def _row(self, writer, timestamp, kind, identifier, values):
        if self.ndigits is not None:
            timestamp = round(timestamp, self.ndigits)
            values = [round(float(v), self.ndigits) for v in values]
        writer.writerow((timestamp, kind, identifier, *values))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError('CSV writer did not stop')
        if self._error is not None:
            raise RuntimeError('CSV writer failed') from self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
