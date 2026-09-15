# XRT Devices

Python input devices for XR (WebRTC) headsets and MediaPipe camera tracking. This package provides lightweight, robot-agnostic interfaces for real-time human pose streaming, bone processing, and CSV recording/playback.

- **Prior Work & Teleop System**: [XR-Robot-Teleop Project Page](https://xr-robot-teleop-website.pages.dev/)
- **SEW-Mimic Theory & Retargeting**: [SEW-Mimic Website](https://sew-mimic.com/) | [arXiv:2602.01632](https://arxiv.org/abs/2602.01632)

---

## Overview

- **Headset Teleoperation**: Real-time WebRTC receiver for Meta Quest / OpenXR body and hand tracking via [XRT-Client](https://github.com/yunho-c/XRT-Client).
- **Camera Teleoperation**: MediaPipe Tasks integration for webcam-based upper-body and hand tracking without specialized VR hardware.
- **Recording & Playback**: High-throughput background CSV recording and deterministic offline playback.
- **Process Isolation**: A shared-memory latest-snapshot mailbox keeps pose IPC bounded when a consumer or solver loop slows down.
- **Keypoint Visualization**: For detailed keypoint definitions, hierarchy, and coordinate axes, see the interactive [Rerun replay visualization](https://xr-robot-teleop-website.pages.dev/).
- **Geometric Frame Definitions**: For the mathematical formulation of Shoulder-Elbow-Wrist (SEW) frames, orientation alignment, and closed-form retargeting, see the [SEW-Mimic paper](#citation) and [sew-mimic.com](https://sew-mimic.com/).

---

## Input conventions: SEED bones, IOBT, and MediaPipe

These tables describe **the inputs this package actually consumes and the fields it
currently emits**. “SEED” means the supported bone CSV/CSV.GZ export, including the
frozen WARP fixture; it does not mean arbitrary raw BVH or a different SEED exporter.
“IOBT” (inside-out body tracking) means the Quest body/hand skeleton carried by XRT-Client's `body_pose`
channel. A CSV is a storage format: an IOBT recording can also be a CSV, so the
filename extension does not determine the anatomical convention.

### Source coordinates and time

| Property | SEED bone CSV export | Live IOBT through XRT-Client | MediaPipe Pose + Hand Tasks |
|---|---|---|---|
| Input representation | Rows grouped by `time_elapsed` or `timestamp`; `bone_id` or `id`; position and quaternion columns | Little-endian `int32 count`, then `count` records of `<i7f>`: ID, XYZ position, XYZW quaternion | 33 Pose landmarks and 21 landmarks per detected hand; image and metric world results are separate |
| Skeleton identifiers | Uses this package's `FullBodyBoneId` namespace. The canonical WARP CSV has 71 distinct IDs, including root | Same `FullBodyBoneId` namespace, up to 84 defined pose IDs (0–83); packet count is explicit | Pose and Hand have separate index spaces; Hand index 0 is a wrist, not body/root ID 0 |
| Position units used by processing | Metres, already in the exported internal coordinate system; reader does not rescale | Metres expected from the Unity capture; receiver does not rescale | Metric **world** landmarks for geometry; normalized image XY only for association, visibility checks, and drawing |
| Axes at ingestion | Already right-handed FLU: +X forward, +Y left, +Z up | Wire input is Unity left-handed: +X right, +Y up, +Z forward. Decode once with `p_internal=(z,-x,y)` | Keep the native MediaPipe world vectors until constructing the body frame; do **not** apply the Unity permutation. The upright-camera fallback treats native +Y as camera-down |
| Quaternion order | `rot_x, rot_y, rot_z, rot_w`; read as stored | Wire XYZW becomes internal `(-qz,qx,-qy,qw)` | No landmark quaternion is supplied; wrist orientation is constructed from hand points |
| Position reference | Exported world/global bone positions, not parent-relative offsets | Common capture/world frame expected by the receiver, not parent-relative bone offsets | Pose-world origin is the hip midpoint; Hand-world origin is the hand's geometric center. They are **not one shared translated world frame** |
| Conversion on replay | Bone rows are already converted: no second Unity-to-FLU transform | A CSV recorded by XRT_devices stores decoded internal bones; replay follows the same rule as SEED CSV | Hand points are made wrist-relative and attached to the matching Pose wrist before body/wrist transforms |
| Timing | Timestamp values are seconds. Playback uses the stored sampling times; WARP explicitly samples at 30 Hz | `DeviceFrame.received_at` is host monotonic receive time, not headset capture time. The body packet carries no source timestamp | Both Tasks run sequentially on the same RGB image. The device frame timestamp is host monotonic camera capture time; Tasks receive increasing millisecond timestamps |

Google documents the separate metric origins in its
[Pose result contract](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python#handle_and_display_results)
and [Hand result contract](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results).
The body/hand alignment below is this package's implementation, not a claim that
the two independently estimated metric spaces are perfectly calibrated.

### Body landmarks and derived frames

In this table, bone names omit `FullBody_`; paired IDs are **left / right**.
`S`, `E`, and `W` denote shoulder, elbow, and wrist positions supplied to retargeting.

| Quantity | SEED / frozen WARP convention | Current IOBT default | MediaPipe convention |
|---|---|---|---|
| SEW shoulder `S` | `ArmUpper` **10 / 15** | `ArmUpper` **10 / 15**; not clavicle/`Shoulder` | Pose `SHOULDER` **11 / 12** |
| SEW elbow `E` | `ArmLower` **11 / 16** | Same | Pose `ELBOW` **13 / 14** |
| SEW wrist `W` | `HandWrist` **19 / 45** | Same; not `HandWristTwist` **12 / 17** or palm **18 / 44** | Pose `WRIST` **15 / 16** provides the arm wrist; Hand `WRIST` **0** is aligned to it |
| Upper-body origin `o_B` | Mean of left/right `ArmUpper` **10,15**, with `body_frame="upper_arms"` | Mean of left/right `Shoulder` **8,13**, with `body_frame="hips"` | Mean of Pose shoulders **11,12** |
| Upper-body +Y | Right-to-left `ArmUpper` line | Right-to-left `UpperLeg`/hip line **77 → 70** | Right-to-left Pose shoulder line **12 → 11** |
| Initial up reference | `o_B - SpineMiddle(3)` | `o_B - SpineMiddle(3)` | Shoulder midpoint minus Pose hip midpoint **23,24** |
| Upper-body +X, +Z | Normalize `X = Y × up`; `Z = X × Y`; SVD orthogonalization | Same construction, with the hip-derived Y axis | Normalize `X = Y × up`; recompute `Z = X × Y` to make an orthonormal body frame |
| Lower-body frame | Origin at mean of `UpperLeg` **70,77**; +Y right-to-left hips, up reference toward `SpineMiddle(3)` | Same | No lower-body retargeting frame emitted |
| Legs | Hip **70 / 77**, knee **71 / 78**, ankle **73 / 80**; optional foot points affect ankle orientation | Same; additional foot landmarks may be available | Pose estimates have leg points, but this adapter does not emit HKA/ankle/base control |
| Head | `Head(7)` quaternion, with the existing intrinsic `ZYX=(-π/2,-π/2,0)` alignment, expressed in the upper-body frame | Same | No head-orientation output from this adapter |
| Torso output | `R_lower_upper = R_world_lower.T @ R_world_upper` | Same | Raw action has `R_torso` for the camera-relative body frame. It is **not** `R_lower_upper`; the typed adapter does not turn it into waist/base tracking |
| Missing hips | The bone processor expects the required full-body anchors | Same | If either hip is off-image or has visibility ≤0.75, use a reference 0.5 m in native camera-down below the shoulder midpoint. This establishes an upright upper-body reference only |
| Left/right naming | Anatomical side encoded by the bone ID | Anatomical side encoded by the bone ID | Associate Hand image wrists to anatomical Pose wrists by minimum total squared image-XY distance, one-to-one. Hand classifier labels do not decide which arm receives a hand |

**The body-frame choice is explicit, not inferred from CSV versus live input.**
`bones_to_action()` and `OfflineCSVAdapter` default to `body_frame="hips"`.
WARP's converter deliberately passes `body_frame="upper_arms"`; this reproduces
its existing 420-frame fixture to absolute tolerance `1e-12`. Changing only this
option changes the body basis and hence SEW/head/torso coordinates, even when the
raw bones are identical. To replay a current IOBT capture, keep the `hips` default.

```python
from xrt_devices.integrations.geo_kin import OfflineCSVAdapter

paper_source = OfflineCSVAdapter("seed.csv.gz", loop=False, body_frame="upper_arms")
headset_source = OfflineCSVAdapter("headset.csv", loop=False, body_frame="hips")
```

### Hand landmark names and indices

The following are **raw identifiers**, before the legacy SEED mapping and fingertip
adjustment described next. XR finger chains include metacarpal bone positions that
have no separate counterpart in the 21-point MediaPipe hand model.

| Landmark chain | FullBody IDs, left / right (SEED and IOBT namespace) | MediaPipe Hand indices |
|---|---|---|
| Wrist | `HandWrist`: **19 / 45** | `WRIST`: **0** |
| Thumb | `ThumbMetacarpal, Proximal, Distal, Tip`: **20–23 / 46–49** | `THUMB_CMC, MCP, IP, TIP`: **1–4** |
| Index | `IndexMetacarpal, Proximal, Intermediate, Distal, Tip`: **24–28 / 50–54** | `INDEX_FINGER_MCP, PIP, DIP, TIP`: **5–8** |
| Middle | `MiddleMetacarpal, Proximal, Intermediate, Distal, Tip`: **29–33 / 55–59** | `MIDDLE_FINGER_MCP, PIP, DIP, TIP`: **9–12** |
| Ring | `RingMetacarpal, Proximal, Intermediate, Distal, Tip`: **34–38 / 60–64** | `RING_FINGER_MCP, PIP, DIP, TIP`: **13–16** |
| Little/pinky | `LittleMetacarpal, Proximal, Intermediate, Distal, Tip`: **39–43 / 65–69** | `PINKY_MCP, PIP, DIP, TIP`: **17–20** |

Names in the emitted solver dictionaries are **compatibility keys**. For SEED in
particular, a key named `mcp` is not necessarily the raw bone named `Proximal`.
For each row below, entries are the source of the emitted **MCP / PIP / tip** keys.

| Emitted finger keys | SEED legacy branch (left scapula absent) | IOBT branch (left scapula present) | MediaPipe |
|---|---|---|---|
| `thumb_mcp / thumb_pip / thumb_tip` | `ThumbMetacarpal / ThumbProximal / ThumbTip` | `ThumbProximal / ThumbDistal* / ThumbTip*` | `THUMB_MCP(2) / THUMB_IP(3) / THUMB_TIP(4)` |
| `index_finger_mcp / index_finger_pip / index_finger_tip` | `IndexIntermediate / IndexTip* / IndexTip*` | `IndexProximal / IndexIntermediate / IndexTip*` | **5 / 6 / 8** |
| `middle_finger_mcp / middle_finger_pip / middle_finger_tip` | `MiddleIntermediate / MiddleTip* / MiddleTip*` | `MiddleProximal / MiddleIntermediate / MiddleTip*` | **9 / 10 / 12** |
| `ring_finger_mcp / ring_finger_pip / ring_finger_tip` | `RingIntermediate / RingTip* / RingTip*` | `RingProximal / RingIntermediate / RingTip*` | **13 / 14 / 16** |
| `pinky_mcp / pinky_pip / pinky_tip` | `LittleIntermediate / LittleTip* / LittleTip*` | `LittleProximal / LittleIntermediate / LittleTip*` | **17 / 18 / 20** |
| Additional output keys | The normalized finger dictionary contains the three slots above | Same; raw metacarpal/DIP bones still exist in the capture skeleton | Thumb also retains `thumb_cmc(1)` and `thumb_ip(3)`; other fingers retain their DIP points **7,11,15,19** |

`thumb_pip` is the shared solver alias for MediaPipe's **thumb IP**, not an extra
anatomical thumb joint. Both `thumb_ip` and `thumb_pip` retain the same position.
The outer dictionary keys are `thumb`, `index`, `middle`, `ring`, and `pinky`.

Two separate existing heuristics govern the bone path:

- **Finger-slot mapping:** absence of `LeftScapula(9)` selects the legacy SEED/VMD
  mapping, for both hands. The canonical SEED fixture lacks scapula IDs. This is
  a presence heuristic, not a reliable source-format tag; a reduced IOBT packet
  without that bone would also select the legacy mapping.
- **Tip adjustment (`*`):** presence of `LeftHandThumbDistal(22)` enables the
  existing adjustment for both hands: every tip becomes
  `p_tip + R_tip @ (0, side_sign × 0.01, 0)` metres, with `side_sign=+1` for left
  and `-1` for right. The thumb's `Distal` position is replaced by the original,
  unextended thumb-tip position before the new tip is inserted. **The canonical
  SEED fixture has ID 22, so this adjustment applies there too**, despite using
  the legacy finger-slot mapping. MediaPipe has no such 1 cm tip extrusion.

Those adjustments affect the derived finger payload, not the stored raw bones
or the raw skeleton overlay. They remain unchanged to preserve recorded behavior.
An absent bone must not be replaced by a different enum's similarly numbered point.

### Wrist frame, transforms, and validity

All three inputs emit SEW positions in a human body frame whose intended axes are
**+X forward, +Y anatomical left, +Z up**. Finger positions are in a **wrist-local
frame**, which has a different axis definition:

| Property | SEED bones | IOBT bones | MediaPipe hands |
|---|---|---|---|
| Wrist origin | `HandWrist` | `HandWrist` | Pose wrist, after attaching Hand-world points to it |
| Wrist +X | Wrist toward mean of available index/middle/ring/little **Proximal** raw bone points | Same | Wrist toward mean of index/middle/ring/pinky MCP points |
| Wrist +Y | SVD palm-plane normal; sign aligned to `(index − wrist) × (little − wrist)` | Same | Same sign rule with index/pinky MCP landmarks |
| Wrist +Z | `X × Y`, followed by SVD frame orthogonalization | Same | `X × Y`; reject the frame when its determinant differs from 1 by more than 0.1 |
| Source of SEW wrist orientation | Derived from landmark geometry, not the raw wrist quaternion | Same | Derived from metric hand geometry; no source wrist quaternion |
| Hand-to-body placement | Raw body and hand bones share a capture frame | Same | `p_hand_aligned = p_pose_wrist + (p_hand_world − p_hand_world_wrist)`; then transform to the body frame |
| Missing finger data | No valid coordinates yields an empty finger dictionary; missing slots in a partially valid finger are zero-filled by the legacy mapper | Same | Missing/low-confidence/invalid hand geometry yields no fresh finger dictionary; fingers are not held over dropouts |
| Wrist dropout | Offline replay samples recorded frames; no MediaPipe-style wrist hold | No wrist hold; live snapshots expire after 0.25 s by default | Last valid wrist orientation can be held for 0.5 s from its capture timestamp, retained in native camera axes and re-expressed in the current body frame. No identity fallback or motion prediction |
| Engagement/validity | Offline sampling has no live connection gate | Disconnect invalidates current tracking; malformed packets do not produce a fresh valid pose | Both arms' shoulder/elbow/wrist visibility must exceed 0.75. Wrist estimation requires hand confidence >0.5; uninitialized/expired wrist orientation produces invalid SEW and the typed adapter returns that arm as unavailable |

The wrist +Y sign rule is applied as written to each anatomical hand; there is no
extra “invert right hand” step in the palm-normal construction. The side sign in
the fingertip adjustment is a separate operation.

Matrix direction is defined by the **implementation**, not by ambiguous “world-to-body”
wording: columns of `R_world_body` are the body axes expressed in the source world.
Thus `p_body = R_world_body.T @ (p_world - o_body)`. Likewise
`R_body_wrist = R_world_body.T @ R_world_wrist`, and
`p_wrist = R_body_wrist.T @ (p_body - W_body)`.
The 18-value SEW array is `[S_xyz, E_xyz, W_xyz, R_body_wrist.flatten()]`, with the
3×3 matrix flattened in **row-major** order; the final nine values are not a
quaternion or Euler angles. Device normalization does not apply robot shoulder
scales, retarget offsets, IK, or actuator joint ordering.

Implementation references: [bone processing](src/xrt_devices/processing.py),
[bone IDs](src/xrt_devices/schemas/openxr_skeletons.py),
[Unity conversion](src/xrt_devices/utils/coordinate.py),
[CSV parsing](src/xrt_devices/recording/csv_reader.py),
[MediaPipe processing](src/xrt_devices/_mediapipe_processing.py), and
[typed adapter](src/xrt_devices/integrations/geo_kin.py).

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

## Study transport and video

```python
from xrt_devices.study import StudyXRDevice
from xrt_devices.study_process import WebRTCServerProxy

# Use StudyXRDevice for direct transport, or the same API in a separate process.
# Spawned processes require this guard in executable Python scripts.
if __name__ == "__main__":
    device = WebRTCServerProxy({"host": "0.0.0.0", "port": 8080})
    device.start()
    try:
        device.configure_recording(record_data=True, output_dir="recordings/take1", started=False)
        # Run the app loop here: read get_controller_state(), poll_unity_state(),
        # and telemetry; send_haptics()/send_motor_stats()/send_unity_command()
        # transport application payloads.
    finally:
        device.close()
```

Use `configure_recording(record_data=False, output_dir=..., started=False)` to
finish a take before selecting another directory. CSV rows retain the legacy
bone/action format and include AprilTags; `csv_start_time.txt` records wall-clock
alignment while elapsed times use the monotonic receive clock. Recording overflow
is counted by the recorder; shutdown failures are explicit.
`network_rtt_ms()` measures ping/pong round trips with headset processing time
removed; it does not claim synchronized one-way latency. `video_latency()` returns
headset-reported components. UI events are ordered separately from latest telemetry.

`from xrt_devices.video import zed_stream_script` locates the optional Bash
launcher in installed wheels. Run it with Bash on a host providing GStreamer,
ZED SDK, NVIDIA NVENC and MediaMTX. Importing this resource starts no camera or GPU.
