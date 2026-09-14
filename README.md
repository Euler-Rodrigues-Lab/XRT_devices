# XRT Devices

Python input devices for XR (WebRTC) headsets and MediaPipe camera tracking. This package provides lightweight, robot-agnostic interfaces for real-time human pose streaming, bone processing, and CSV recording/playback.

- **Prior Work & Teleop System**: [XR-Robot-Teleop Project Page](https://xr-robot-teleop-website.pages.dev/)
- **SEW-Mimic Theory & Retargeting**: [SEW-Mimic Website](https://sew-mimic.com/) | [arXiv:2602.01632](https://arxiv.org/abs/2602.01632)

---

## Overview

- **Headset Teleoperation**: Real-time WebRTC receiver for Meta Quest / OpenXR body and hand tracking via [XRT-Client](https://github.com/yunho-c/XRT-Client).
- **Camera Teleoperation**: MediaPipe Tasks integration for webcam-based upper-body and hand tracking without specialized VR hardware.
- **Recording & Playback**: High-throughput background CSV recording and deterministic offline playback.
- **Process Isolation**: Zero-copy shared-memory isolation preventing slow consumer or solver loops from causing network lag or frame accumulation.
- **Keypoint Visualization**: For detailed keypoint definitions, hierarchy, and coordinate axes, see the interactive [Rerun replay visualization](https://xr-robot-teleop-website.pages.dev/).
- **Geometric Frame Definitions**: For the mathematical formulation of Shoulder-Elbow-Wrist (SEW) frames, orientation alignment, and closed-form retargeting, see the [SEW-Mimic paper](#citation) and [sew-mimic.com](https://sew-mimic.com/).

---

## Install

```sh
# Basic XR headset & CSV recording/playback support:
python -m pip install -e '.[xr,recording]'

# For MediaPipe camera input:
python -m pip install -e '.[mediapipe]'
```

---

## XRT Headset

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

Connect [XRT-Client](https://github.com/yunho-c/XRT-Client) to this host, port `8080`.
Signaling is handled via `POST /offer`. One headset per instance is supported. Client-created WebRTC data channels include: `body_pose`, `apriltag_pose`, `unity_state`, `haptics`, `motor_stats`, and `unity_cmds`. Feedback methods include `send_haptics`, `send_motor_stats`, and `send_unity_command`; poll UI events with `poll_unity_state`.

Construction does not automatically open the server. Use the context manager or call `start()` / `cleanup()`.

### Robot-Process Isolation

The shared `XRDeviceAdapter` used by downstream robot repositories (e.g. G1, RB-Y1) starts WebRTC streaming and pose conversion in a spawned dedicated background process by default. A fixed-size shared-memory latest-snapshot slot replaces queued IPC, preventing slow solver or render cycles from accumulating historical input frames.

- Pass `process_isolated=False` for in-process embedding or debugging.
- Start adapters under a Python `if __name__ == "__main__":` guard.
- Pass `--input_diagnostics` in supported CLIs to monitor consumer read rate, receive age, input PID, and packet stats every 5 seconds.

---

## MediaPipe Camera

On first camera startup, omitted model paths automatically download Google's Pose Landmarker Lite and Hand Landmarker Tasks bundles to `~/.cache/xrt_devices/models` (configurable via `XRT_DEVICES_MODEL_DIR`).

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

The Tasks API processes pose and hand landmarks sequentially in a dedicated worker thread (VIDEO tracking mode). Pass `display=True` for an OpenCV debug window.

---

## CSV Playback & Recording

### Offline Playback

```python
from xrt_devices.recording.csv_reader import CSVDataReader
from xrt_devices.processing import bones_to_action

reader = CSVDataReader('capture.csv.gz', loop=False)
bones, tags = reader.get_bones_and_tags_at_time(0.0)
action = bones_to_action(bones, tags) if bones else None
```

Supports `time_elapsed` and numeric `timestamp` columns, gzip-compressed CSV, bone rows, AprilTag rows, and board poses.

### Live Recording

```python
from xrt_devices import XRDevice
from xrt_devices.recording.writer import CSVRecorder

with CSVRecorder('capture.csv') as recorder:
    with XRDevice(recorder=recorder) as device:
        # Run polling loop here
        ...
```

`CSVRecorder` runs file I/O on a background thread with bounded memory queues, logging `recorder.dropped_frames` if disk writes fall behind.

---

## Frame Contract & Coordinate Conventions

- **Coordinate System**: Wire poses use Unity coordinate space (metres, xyzw quaternions) and are converted to right-handed standard robotics convention ($X$-forward, $Y$-left, $Z$-up).
- **SEW Representation**: Both devices produce an 18-value SEW vector per arm ($S$, $E$, $W$ position coordinates followed by row-major $3 \times 3$ wrist rotation matrix), along with hand landmark dictionaries and torso/head poses where available.
- **Integration**: The optional `xrt_devices.integrations.geo_kin` module provides shared adapters and typed frame conversion for `geo_kin_core`.

---

## Citation & References

If you use `xrt_devices` or the SEW teleoperation framework in your research, please cite the following:

```bibtex
@article{kong2026closedform,
  title={A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation},
  author={Kong, Chuizheng and Cho, Yunho and Jung, Wonsuhk and Wibowo, Idris and Shinde, Parth and Vinodh-Sangeetha, Sundhar and Chung, Long Kiu and Chen, Zhenyang and others},
  journal={arXiv preprint arXiv:2602.01632},
  year={2026},
  url={https://arxiv.org/abs/2602.01632}
}
```

- **Project Web**: [https://sew-mimic.com/](https://sew-mimic.com/)
- **Prior Teleop System**: [https://xr-robot-teleop-website.pages.dev/](https://xr-robot-teleop-website.pages.dev/)
- **Upstream Client**: [XR-Robot-Teleop-Client](https://github.com/yunho-c/XR-Robot-Teleop-Client)

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) for details.
# MediaPipe coordinate conventions

Pose and Hand Tasks provide metric world landmarks with separate origins. Hands
are made wrist-relative and attached to the corresponding pose wrist before
conversion to the body frame (+X forward, +Y anatomical left, +Z up).
Hand-to-arm assignment uses the same-image pose wrists, not a hardcoded swap of
the classifier's selfie labels. Overlapping/occluded wrists can still be ambiguous.
When hips are outside the image or low-confidence, the upper-body frame assumes
anchors directly below the shoulders along camera-down. Keep the camera upright;
this fallback is not measured lower-body tracking and does not enable base control.
