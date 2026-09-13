# XRT devices

Python input devices for XRT headsets and MediaPipe cameras. No robot SDK,
private solver, Torch, or original project checkout is required. This first
release transfers human-pose processing and CSV playback; Rust is deferred.

## Install

```sh
python -m pip install -e '.[xr,recording]'
# For camera input, additionally:
python -m pip install -e '.[mediapipe]'
```

## XRT headset

```python
import time
from xrt_devices import XRDevice

with XRDevice(port=8080) as device:
    while True:
        frame = device.get_frame()
        if frame is not None:
            print(frame.sequence, frame.action.get('left_sew'))
        time.sleep(0.02)
```

Connect [XRT-Client](https://github.com/yunho-c/XRT-Client) to this host, port 8080.
Signaling is `POST /offer`. One headset per instance is supported. Client-created
channels: `body_pose`, `apriltag_pose`, `unity_state`, `haptics`, `motor_stats`,
`unity_cmds`. Feedback methods are `send_haptics`, `send_motor_stats`, and
`send_unity_command`; drain UI events with `poll_unity_state`.

Construction does not open a server. `start()`/`start_control()` starts it;
`close()`/`cleanup()` stops it. `XRRTCBodyPoseDevice` aliases `XRDevice`, but this is
not a drop-in replacement for every study-specific legacy constructor or method.
No robot commands are issued by this package.

### Robot-process isolation

The shared `XRDeviceAdapter` used by G1/RBY1 now starts WebRTC and pose conversion
in a spawned process by default. A fixed-size shared-memory latest-snapshot slot
replaces queued IPC, so a slow solver cannot accumulate historical input frames.
`process_isolated=False` remains available for in-process embedding/tests. The
isolated facade supports pose input and CSV recording; study feedback/event APIs
still require direct `XRDevice` integration. Start adapters under a Python
`if __name__ == "__main__":` guard, as both robot CLIs already do.

Both live robot CLIs accept `--input_diagnostics`. Every five seconds this reports
consumer read rate, receive age, input PID and dropped/invalid packet counters.
Receive age starts at the server callback, not headset capture: it is not an
end-to-end network latency measurement. Live headset performance still requires
validation, particularly under sustained solver/render load.

## MediaPipe camera

On first camera startup, omitted model paths automatically download Google's
version-1 Pose Landmarker Lite and Hand Landmarker Tasks bundles. They are reused
from `~/.cache/xrt_devices/models` (respects `XDG_CACHE_HOME`); override the cache
with `XRT_DEVICES_MODEL_DIR`. No downloads occur on import or for XRT input.
Pass `pose_model` and `hand_model` explicitly for custom models or offline setup.
Explicit missing files fail instead of triggering a download. Interrupted or
invalid downloads are not cached. Model files are not bundled in the package.

```python
import time
from xrt_devices import MediaPipeTeleopDevice

with MediaPipeTeleopDevice(display=False) as device:
    while True:
        frame = device.get_frame()
        if frame is not None:
            print(frame.action['left_sew'])
        time.sleep(0.02)
```

The Tasks API runs pose and hands sequentially on the same image in one background
worker (VIDEO tracking mode). Human landmark/wrist processing is transferred from
the established Python device. `display=True` enables an OpenCV text/debug window;
skeleton drawing is not yet implemented for Tasks results. Camera construction
starts capture; use the context manager to release it.

## CSV playback

```python
from xrt_devices.recording.csv_reader import CSVDataReader
from xrt_devices.processing import bones_to_action

reader = CSVDataReader('capture.csv.gz', loop=False)
bones, tags = reader.get_bones_and_tags_at_time(0.0)
action = bones_to_action(bones, tags) if bones else None
```

Supports `time_elapsed` and numeric `timestamp` columns, compressed CSV, bone rows,
AprilTag rows and optional paper-board poses. Coordinates in existing recordings
are already converted; they are not converted from Unity a second time.

For live bone recording, use `CSVRecorder` (exclusive file creation, bounded queue,
drained on close). Recording runs outside the network callback:

```python
from xrt_devices import XRDevice
from xrt_devices.recording.writer import CSVRecorder

with CSVRecorder('capture.csv') as recorder:
    with XRDevice(recorder=recorder) as device:
        # Run your application's polling loop here.
        ...
```

Inspect `recorder.dropped_frames` if recording throughput falls behind. This writer
records processed bone frames, not every raw network arrival, action rows or tags.

## Frame contract and limitations

Both devices expose `get_frame() -> DeviceFrame | None`, with source, sequence,
local monotonic receive time and the established action dictionary. XR frames use
an 18-value SEW vector per arm (S/E/W followed by row-major 3x3 wrist rotation),
finger dictionaries and available torso/head fields. MediaPipe exposes the same
arm vector shape plus its existing tracking/hand fields; torso/head parity with
XRT is not claimed. `get_controller_state()` provides legacy dictionaries.

Wire poses use Unity metres and xyzw quaternions; conversion is right-handed
X-forward/Y-left/Z-up. A latest-value queue bounds XR backlog. Invalid/stale poses
return no frame; controllers must implement their own engagement and hold policy.
Operator events have a separate bounded queue. Feedback needs the client's channel
to be open; this release does not acknowledge delivery. Do not equate connection
state or pose visibility with permission to actuate hardware.

The optional `xrt_devices.integrations.geo_kin` module provides shared XR/MediaPipe
adapters, typed frame conversion and CSV/NPZ playback for G1 and RBY1. It needs
`geo_kin_core` supplied by the robot package; the base device package does not import it.

Not yet migrated: study-specific process proxy, extended study recording, video/panorama,
latency UI and study application entry points. G1 and RBY1 demos use this package;
the study applications still await dependency updates. Headset/camera hardware validation
is still required; offline tests cannot certify live interoperability.

## Development

```sh
uv sync --extra xr --extra recording --extra test
uv run pytest
uv run python -m build
```

See [MIGRATION.md](MIGRATION.md) for transfer status and upstream provenance.
