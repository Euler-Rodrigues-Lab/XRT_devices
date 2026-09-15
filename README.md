# XRT Devices

Python input devices for XR (WebRTC) headsets, MediaPipe camera tracking, and offline BVH motion. This package provides lightweight, robot-agnostic interfaces for real-time human pose streaming, bone processing, and CSV recording/playback.

- **Prior Work & Teleop System**: [XR-Robot-Teleop Project Page](https://xr-robot-teleop-website.pages.dev/)
- **SEW-Mimic Theory & Retargeting**: [SEW-Mimic Website](https://sew-mimic.com/) | [arXiv:2602.01632](https://arxiv.org/abs/2602.01632)

---

## Overview

- **Headset Teleoperation**: Real-time WebRTC receiver for Meta Quest / OpenXR body and hand tracking via [XRT-Client](https://github.com/yunho-c/XRT-Client).
- **Camera Teleoperation**: MediaPipe Tasks integration for webcam-based upper-body and hand tracking without specialized VR hardware.
- **Recording & Playback**: High-throughput background CSV recording and deterministic CSV/BVH playback.
- **Process Isolation**: A shared-memory latest-snapshot mailbox keeps pose IPC bounded when a consumer or solver loop slows down.
- **Keypoint Visualization**: For detailed keypoint definitions, hierarchy, and coordinate axes, see the interactive [Rerun replay visualization](https://xr-robot-teleop-website.pages.dev/).
- **Geometric Frame Definitions**: For the mathematical formulation of Shoulder-Elbow-Wrist (SEW) frames, orientation alignment, and closed-form retargeting, see the [SEW-Mimic paper](#citation) and [sew-mimic.com](https://sew-mimic.com/).

---

## Skeleton conventions: BONES-SEED (SOMA), OpenXR, LaFAN1, and MediaPipe

These tables compare **source keypoint definitions** and **the keypoints consumed
by processing**. Body-frame configuration is documented separately below. BVH playback uses the original repository's LaFAN1 mapping by
default, with explicit overrides for other joint names, units, and world axes.
“BONES-SEED” means the supported bone CSV/CSV.GZ export, including the
recording used by WARP; it does not mean arbitrary raw BVH or a different BONES-SEED exporter.
“OpenXR (Quest IOBT)” (inside-out body tracking) means the Quest body/hand skeleton carried by XRT-Client's `body_pose`
channel. A CSV is a storage format: an OpenXR (Quest IOBT) recording can also be a CSV, so the
filename extension does not determine the anatomical convention.

### Source body keypoints

Names below are exact enum suffixes: prepend `FullBody_`; `{Left,Right}` and
paired IDs are **left / right**. BONES-SEED exports and OpenXR (Quest IOBT) share this **identifier
namespace**, so it is combined here. A shared ID does not prove identical source
anatomy, point availability, or exporter placement. BONES-SEED uses the SOMA
rig before conversion; it must not be conflated with a VMD/MMD skeleton.

| Source keypoint | BONES-SEED / OpenXR (Quest IOBT) `FullBodyBoneId` | LaFAN1 joint names | MediaPipe Pose |
|---|---|---|---|
| Root | `Root` **0** (`Start` is an enum alias) | No separate mapping; BVH root is `Hips` | No root landmark |
| Pelvis | `Hips` **1** | `Hips` | No pelvis landmark; hip midpoint is derived |
| Lower spine | `SpineLower` **2** | `Spine` | No corresponding landmark |
| Middle spine | `SpineMiddle` **3** | `Spine1` | No corresponding landmark |
| Upper spine | `SpineUpper` **4** | `Spine2` | No corresponding landmark |
| Chest | `Chest` **5** | `Chest`, if present | No corresponding landmark |
| Neck | `Neck` **6** | `Neck` | No corresponding landmark |
| Head | `Head` **7** | `Head` | Face landmarks **0–10**; no single equivalent head joint |
| Shoulder / clavicle anchor | `{Left,Right}Shoulder` **8 / 13** | `LeftShoulder / RightShoulder` | No separate clavicle anchor |
| Scapula | `{Left,Right}Scapula` **9 / 14** | `LeftScapula / RightScapula`, if present | No corresponding landmark |
| Upper-arm joint | `{Left,Right}ArmUpper` **10 / 15** | `LeftArm / RightArm` | `LEFT_SHOULDER / RIGHT_SHOULDER` **11 / 12** |
| Lower-arm joint | `{Left,Right}ArmLower` **11 / 16** | `LeftForeArm / RightForeArm` | `LEFT_ELBOW / RIGHT_ELBOW` **13 / 14** |
| Wrist twist | `{Left,Right}HandWristTwist` **12 / 17** | No mapping | No corresponding landmark |
| Palm | `{Left,Right}HandPalm` **18 / 44** | No mapping | No palm-center landmark |
| Wrist | `{Left,Right}HandWrist` **19 / 45** | `LeftHand / RightHand` | `LEFT_WRIST / RIGHT_WRIST` **15 / 16** |
| Hip joint | `{Left,Right}UpperLeg` **70 / 77** | `LeftUpLeg / RightUpLeg` | `LEFT_HIP / RIGHT_HIP` **23 / 24** |
| Knee joint | `{Left,Right}LowerLeg` **71 / 78** | `LeftLeg / RightLeg` | `LEFT_KNEE / RIGHT_KNEE` **25 / 26** |
| Ankle twist | `{Left,Right}FootAnkleTwist` **72 / 79** | No mapping | No corresponding landmark |
| Ankle joint | `{Left,Right}FootAnkle` **73 / 80** | `LeftFoot / RightFoot`; original mapping also accepts `LeftFootMod / RightFootMod` | `LEFT_ANKLE / RIGHT_ANKLE` **27 / 28** |
| Subtalar | `{Left,Right}FootSubtalar` **74 / 81** | No mapping | No exact equivalent; `LEFT_HEEL / RIGHT_HEEL` **29 / 30** are heel points |
| Transverse foot | `{Left,Right}FootTransverse` **75 / 82** | No mapping | No corresponding landmark |
| Ball of foot | `{Left,Right}FootBall` **76 / 83** | `LeftToe / RightToe` | `LEFT_FOOT_INDEX / RIGHT_FOOT_INDEX` **31 / 32** are toe landmarks, not an exact ball-joint equivalent |

The BONES-SEED recording used by WARP omits IDs **5, 9, 12, 14, 17, 18, 44, 72, 74, 75,
79, 81, 82**. These omissions are intentional in the supplied BONES-SEED
export contract; availability must still be checked per export. `FullBody_End(84)` is a sentinel, not a keypoint.

BVH joint names come from each file's hierarchy; the names above are the original
repository's `BVH_TO_FULL_BODY` mapping, not universal BVH identifiers. Optional
mapping entries do not imply that every LaFAN1 file contains those joints. Resolve
parent-relative offsets and rotation channels through forward kinematics before
using joint positions as world-space bones. The original reader defaults to
`unit_scale=0.01`, converts Y-up with `(x,y,z) → (x,-z,y)`, and reads Euler angles
in degrees in declared channel order; arbitrary BVH exporters need explicit unit,
axis, and joint-name conventions. Frame times come from `Frame Time`.

### Body keypoints consumed by processing

BONES-SEED and OpenXR (Quest IOBT) use the same body-keypoint selections, so they share one column.
WARP is a downstream consumer, not another input skeleton convention. Paired IDs
are **left / right**; enum names omit `FullBody_Left` / `FullBody_Right` for paired
points and `FullBody_` for spine/head points. Frame origins, axes, and rotations
are processing results and are not keypoints in this table.

| Processing keypoint | BONES-SEED / OpenXR (Quest IOBT) | LaFAN1 | MediaPipe |
|---|---|---|---|
| Arm shoulder | `ArmUpper` **10 / 15** | `LeftArm / RightArm` | Pose `LEFT_SHOULDER / RIGHT_SHOULDER` **11 / 12** |
| Arm elbow | `ArmLower` **11 / 16** | `LeftForeArm / RightForeArm` | Pose `LEFT_ELBOW / RIGHT_ELBOW` **13 / 14** |
| Arm wrist | `HandWrist` **19 / 45** | `LeftHand / RightHand` | Pose `LEFT_WRIST / RIGHT_WRIST` **15 / 16**; Hand `WRIST(0)` is attached to the Pose wrist |
| Upper-body reference landmarks | `ArmUpper` **10 / 15**, `SpineMiddle` **3** only | `LeftArm / RightArm`, `Spine1` only | Pose shoulders **11 / 12** and hip midpoint from **23 / 24** (or the documented off-image fallback) |
| Lower-body reference landmarks | `UpperLeg` **70 / 77**, `SpineMiddle` **3** only | `LeftUpLeg / RightUpLeg`, `Spine1` only | No lower-body frame emitted |
| Hip | `UpperLeg` **70 / 77** | `LeftUpLeg / RightUpLeg` | Pose `LEFT_HIP / RIGHT_HIP` **23 / 24**; used for the upper-body reference, not emitted as leg targets |
| Knee | `LowerLeg` **71 / 78** | `LeftLeg / RightLeg` | No processed leg target; Pose detects **25 / 26** |
| Ankle | `FootAnkle` **73 / 80** | `LeftFoot / RightFoot` | No processed leg target; Pose detects **27 / 28** |
| Optional foot geometry | `FootSubtalar` **74 / 81**, `FootTransverse` **75 / 82**, `FootBall` **76 / 83** | `LeftToe / RightToe` maps to foot ball; BVH ankle orientation uses the foot joint rotation | No processed foot target |
| Head | `Head` **7** | `Head` | No processed head-orientation target |
| Body-center anchor | `SpineLower` **2** | `Spine` | No processed body-center target |

#### Active frame dependencies

[`_get_body_frame`](src/xrt_devices/processing.py) computes **only the upper-body
frame**:

```python
shoulder_center = (left_arm_upper + right_arm_upper) / 2
y_axis = left_arm_upper - right_arm_upper
torso_vector = shoulder_center - spine_middle
```

Only `ArmUpper` **10 / 15** and `SpineMiddle` **3** determine this result.
The original function also reads `Shoulder`, `UpperLeg`, `Hips`, and `SpineUpper`,
and computes shoulder averages and a hip midpoint, but those values do not feed
the returned origin or rotation. They are not upper-body frame anchors. XRT omits
these unused calculations and does not require those points for this helper.
The arm shoulder target likewise uses `ArmUpper`, not `Shoulder` or their average.

[`_get_lower_body_frame`](src/xrt_devices/processing.py) separately uses
`UpperLeg` **70 / 77** and `SpineMiddle` **3**. Its origin is the hip midpoint;
its lateral direction is the right-to-left hip line. The full bone action still
requires a valid lower-body frame to produce its torso-relative rotation.
`SpineLower(2)` supplies the body-center output; it does not define either frame.
The bone action emits `R_torso` as an alias of `R_lower_upper`, matching the
original client. MediaPipe's `R_torso` remains its camera-relative body rotation
and is not interpreted as a measured lower-to-upper torso rotation.

Live OpenXR (Quest IOBT), BONES-SEED CSV replay, and BVH all use the upper-arm convention.
`body_frame="upper_arms"` remains accepted for existing explicit callers such as
[WARP's converter](../WARP_retargeting/src/warp_retargeting/transcode_csv.py), but
is now also the default. The previous `body_frame="hips"` upper-body variant is
rejected: hips define the separate lower-body frame, not the upper-body frame.
The hand-slot selection described below is a separate bone-presence heuristic.
Raw bone gripper actions follow the original client: thumb–index tip distance
above 0.05 m emits `-1`, otherwise `+1`; the distance values are unchanged.

### Hand landmark names and indices

The following are **raw enum identifiers**, before the legacy BONES-SEED mapping and
fingertip adjustment described next. They describe the shared namespace, not a
guarantee that BONES-SEED and OpenXR (Quest IOBT) store the same anatomical point at every ID. XR finger
chains include metacarpal bone positions that have no separate counterpart in the
21-point MediaPipe hand model. In the chain notation below, expand each suffix
with its finger name and `FullBody_LeftHand` or `FullBody_RightHand`: for example,
index `Proximal` means `FullBody_LeftHandIndexProximal(25)` or
`FullBody_RightHandIndexProximal(51)`. MediaPipe suffixes likewise retain the
listed finger prefix.

| Landmark chain | FullBody IDs, left / right (BONES-SEED and OpenXR (Quest IOBT) namespace) | LaFAN1 joint names | MediaPipe Hand indices |
|---|---|---|---|
| Wrist | `HandWrist`: **19 / 45** | `LeftHand / RightHand` | `WRIST`: **0** |
| Thumb | `ThumbMetacarpal, Proximal, Distal, Tip`: **20–23 / 46–49** | No finger-joint mapping | `THUMB_CMC, MCP, IP, TIP`: **1–4** |
| Index | `IndexMetacarpal, Proximal, Intermediate, Distal, Tip`: **24–28 / 50–54** | No finger-joint mapping | `INDEX_FINGER_MCP, PIP, DIP, TIP`: **5–8** |
| Middle | `MiddleMetacarpal, Proximal, Intermediate, Distal, Tip`: **29–33 / 55–59** | No finger-joint mapping | `MIDDLE_FINGER_MCP, PIP, DIP, TIP`: **9–12** |
| Ring | `RingMetacarpal, Proximal, Intermediate, Distal, Tip`: **34–38 / 60–64** | No finger-joint mapping | `RING_FINGER_MCP, PIP, DIP, TIP`: **13–16** |
| Little/pinky | `LittleMetacarpal, Proximal, Intermediate, Distal, Tip`: **39–43 / 65–69** | No finger-joint mapping | `PINKY_MCP, PIP, DIP, TIP`: **17–20** |

The BONES-SEED SOMA mapping and the OpenXR tracker share output IDs, not
necessarily source-joint placement. Retain the exporter mapping when interpreting
anatomical joints; CSV IDs alone do not identify the source skeleton.
The LaFAN1 mapping has no fingers, so it cannot supply the palm geometry used by
the XR wrist estimator; the original BVH reader uses the `LeftHand / RightHand`
global joint rotations (with optional orientation offsets) instead.

Implementation: [LaFAN1 reader and joint mapping](src/xrt_devices/recording/bvh_reader.py).
Source skeleton definitions are distinct from the derived solver slots below.

Names in the emitted solver dictionaries are **compatibility keys**. For BONES-SEED in
particular, a key named `mcp` is not necessarily the raw bone named `Proximal`.
For each row below, entries are the source of the emitted **MCP / PIP / tip** keys.

| Emitted finger keys | Legacy branch (left scapula absent) | Scapula-present branch | MediaPipe |
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

- **Finger-slot mapping:** absence of `LeftScapula(9)` selects the legacy reduced-skeleton
  mapping, for both hands. The canonical BONES-SEED fixture lacks scapula IDs. This is
  a presence heuristic, not a reliable source-format tag; a reduced OpenXR (Quest IOBT) packet
  without that bone would also select the legacy mapping.
- **Tip adjustment (`*`):** presence of `LeftHandThumbDistal(22)` enables the
  existing adjustment for both hands: every tip becomes
  `p_tip + R_tip @ (0, side_sign × 0.01, 0)` metres, with `side_sign=+1` for left
  and `-1` for right. The thumb's `Distal` position is replaced by the original,
  unextended thumb-tip position before the new tip is inserted. **The canonical
  BONES-SEED fixture has ID 22, so this adjustment applies there too**, despite using
  the legacy finger-slot mapping. MediaPipe has no such 1 cm tip extrusion.

Those adjustments affect the derived finger payload, not the stored raw bones
or the raw skeleton overlay. They remain unchanged to preserve recorded behavior.
An absent bone must not be replaced by a different enum's similarly numbered point.

### Wrist frame, transforms, and validity

XR/CSV, BVH, and MediaPipe emit SEW positions in a human body frame whose intended axes are
**+X forward, +Y anatomical left, +Z up**. Finger positions are in a **wrist-local
frame**, which has a different axis definition:

| Property | BONES-SEED bones | OpenXR (Quest IOBT) bones | LaFAN1 wrists | MediaPipe hands |
|---|---|---|---|---|
| Wrist origin | `HandWrist` | `HandWrist` | `LeftHand / RightHand` global positions | Pose wrist, after attaching Hand-world points to it |
| Wrist +X | Wrist toward mean of available index/middle/ring/little **Proximal** raw bone points | Same | BVH hand-joint local +X, after optional wrist offset | Wrist toward mean of index/middle/ring/pinky MCP points |
| Wrist +Y | SVD palm-plane normal; sign aligned to `(index − wrist) × (little − wrist)` | Same | BVH hand-joint local +Y, after optional wrist offset | Same sign rule with index/pinky MCP landmarks |
| Wrist +Z | `X × Y`, followed by SVD frame orthogonalization | Same | BVH hand-joint local +Z, after optional wrist offset | `X × Y`; reject the frame when its determinant differs from 1 by more than 0.1 |
| Source of SEW wrist orientation | Derived from landmark geometry, not the raw wrist quaternion | Same | Global hand-joint FK rotation; optional post-multiplied orientation offset | Derived from metric hand geometry; no source wrist quaternion |
| Hand-to-body placement | Raw body and hand bones share a capture frame | Same | All joints resolved through the same hierarchy and world transform | `p_hand_aligned = p_pose_wrist + (p_hand_world − p_hand_world_wrist)`; then transform to the body frame |
| Missing finger data | No valid coordinates yields an empty finger dictionary; missing slots in a partially valid finger are zero-filled by the legacy mapper | Same | No finger payload from the LaFAN1 mapping | Missing/low-confidence/invalid hand geometry yields no fresh finger dictionary; fingers are not held over dropouts |
| Wrist dropout | Offline replay samples recorded frames; no MediaPipe-style wrist hold | No wrist hold; live snapshots expire after 0.25 s by default | Offline sample hold for one `Frame Time`; no dropout prediction | Last valid wrist orientation can be held for 0.5 s from its capture timestamp, retained in native camera axes and re-expressed in the current body frame. No identity fallback or motion prediction |
| Engagement/validity | Offline sampling has no live connection gate | Disconnect invalidates current tracking; malformed packets do not produce a fresh valid pose | Required mapped arm/body anchors; degenerate body basis yields no frame | Both arms' shoulder/elbow/wrist visibility must exceed 0.75. Wrist estimation requires hand confidence >0.5; uninitialized/expired wrist orientation produces invalid SEW and the typed adapter returns that arm as unavailable |

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
[BVH parsing and playback](src/xrt_devices/recording/bvh_reader.py),
[MediaPipe processing](src/xrt_devices/_mediapipe_processing.py), and
[typed adapter](src/xrt_devices/integrations/geo_kin.py).

### Skeleton convention is not a file container

- **BONES-SEED** uses the **SOMA** skeleton convention (names, hierarchy and rest
  offsets). SOMA can be stored in BVH or USD; it is not synonymous with BVH.
- **LaFAN1** names the dataset/skeleton mapping used by this package's BVH reader.
  A CMU, Mixamo or SOMA BVH needs its own mapping; a `.bvh` suffix does not imply
  LaFAN1 compatibility.
- **OpenXR** here means the Quest full-body extension skeleton transported by
  XRT-Client, not a claim that every OpenXR runtime emits this 84-ID namespace.
- CSV and CSV.GZ describe storage/compression, not a skeleton convention.
  The legacy internal `is_vmd` heuristic is not a dataset identifier.

#### BONES-SEED production conversion contract

The supplied exporter contract is SOMA proportional BVH from HuggingFace at
`soma_proportional/bvh/{YYMMDD}/{name}.bvh`: 120 fps, centimetres, Y-up.
The exporter performs FK on local rotations, applies the axis transform, and
writes global positions/quaternions directly to CSV; CSV.GZ is gzip of that CSV,
with no intermediate skeleton format. Production options are
`--variant proportional --axis-transform soma_to_flu --stride 1`, yielding
metres, Z-up FLU, and `time_elapsed,bone_id,pos_*,rot_*` (quaternion XYZW).

One exporter mapping dictionary maps 70 bones and synthesizes Root (ID 0), for
71 bones/frame. Intentionally omitted: Scapula, WristTwist, Palm, Subtalar,
Transverse, AnkleTwist and Chest (ID 5). SOMA Neck1 is omitted; Neck2 supplies the
single neck. These are exporter details supplied by the dataset workflow, not
a claim that this package includes that raw SOMA converter. The CSV reader
consumes the already-converted export; do not run the LaFAN1 default mapping on
raw SOMA BVH. The default OpenXR namespace includes IDs that may be absent in
actual tracker packets.

### Source coordinates and time

| Property | BONES-SEED bone CSV export | Live OpenXR (Quest IOBT) through XRT-Client | LaFAN1 | MediaPipe Pose + Hand Tasks |
|---|---|---|---|---|
| Input representation | Rows grouped by `time_elapsed` or `timestamp`; `bone_id` or `id`; position and quaternion columns | Little-endian `int32 count`, then `count` records of `<i7f>`: ID, XYZ position, XYZW quaternion | BVH hierarchy (`ROOT`, `JOINT`, `OFFSET`, `CHANNELS`) and sampled `MOTION` rows | 33 Pose landmarks and 21 landmarks per detected hand; image and metric world results are separate |
| Skeleton identifiers | Uses this package's `FullBodyBoneId` namespace. The canonical WARP CSV has 71 distinct IDs, including root | Same `FullBodyBoneId` namespace, up to 84 defined pose IDs (0–83); packet count is explicit | Named hierarchy; explicit LaFAN1-to-`FullBodyBoneId` mapping, configurable per file | Pose and Hand have separate index spaces; Hand index 0 is a wrist, not body/root ID 0 |
| Position units used by processing | Metres, already in the exported internal coordinate system; reader does not rescale | Metres expected from the Unity capture; receiver does not rescale | `unit_scale=0.01` by default (cm → m); explicit override for other exporters | Metric **world** landmarks for geometry; normalized image XY only for association, visibility checks, and drawing |
| Axes at ingestion | Already right-handed FLU: +X forward, +Y left, +Z up | Wire input is Unity left-handed: +X right, +Y up, +Z forward. Decode once with `p_internal=(z,-x,y)` | Right-handed Y-up → Z-up: `(x,-z,y)` by default; configurable `world_rotation` | Keep the native MediaPipe world vectors until constructing the body frame; do **not** apply the Unity permutation. The upright-camera fallback treats native +Y as camera-down |
| Quaternion order | `rot_x, rot_y, rot_z, rot_w`; read as stored | Wire XYZW becomes internal `(-qz,qx,-qy,qw)` | Euler degrees in declared channel order; FK outputs world matrices and XYZW bone quaternions | No landmark quaternion is supplied; wrist orientation is constructed from hand points |
| Position reference | Exported world/global bone positions, not parent-relative offsets | Common capture/world frame expected by the receiver, not parent-relative bone offsets | Parent-relative offsets/channels resolved to global poses by forward kinematics | Pose-world origin is the hip midpoint; Hand-world origin is the hand's geometric center. They are **not one shared translated world frame** |
| Conversion on replay | Bone rows are already converted: no second Unity-to-FLU transform | A CSV recorded by XRT_devices stores decoded internal bones; replay follows the same rule as BONES-SEED CSV | Scale and world rotation applied once while evaluating the hierarchy | Hand points are made wrist-relative and attached to the matching Pose wrist before body/wrist transforms |
| Timing | Timestamp values are seconds. Playback uses the stored sampling times; WARP explicitly samples at 30 Hz | `DeviceFrame.received_at` is host monotonic receive time, not headset capture time. The body packet carries no source timestamp | `Frame Time` seconds per sample; floor sampling, speed scaling, optional looping | Both Tasks run sequentially on the same RGB image. The device frame timestamp is host monotonic camera capture time; Tasks receive increasing millisecond timestamps |

Google documents the separate metric origins in its
[Pose result contract](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python#handle_and_display_results)
and [Hand result contract](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results).
The body/hand alignment below is this package's implementation, not a claim that
the two independently estimated metric spaces are perfectly calibrated.

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

## CSV / BVH Playback & Recording

### CSV Playback

```python
from xrt_devices.recording.csv_reader import CSVDataReader
from xrt_devices.processing import bones_to_action

reader = CSVDataReader('capture.csv.gz', loop=False)
bones, tags = reader.get_bones_and_tags_at_time(0.0)
action = bones_to_action(bones, tags) if bones else None
```

Supports `time_elapsed` and numeric `timestamp` columns, gzip-compressed CSV, bone rows, AprilTag rows, and board poses.

### BVH Playback

BVH parsing and raw actions need only the base NumPy/SciPy dependencies:

```python
from xrt_devices.recording.bvh_reader import BVHDataReader

reader = BVHDataReader("motion.bvh", playback_speed=1.0, loop=False)
action = reader.get_action_at_time(0.0)
bones = reader.get_bones_at_time(0.0)
```

For consumers of `geo_kin_core`, use the common motion-source interface:

```python
from xrt_devices.integrations.geo_kin import open_motion_source

source = open_motion_source(bvh_file="motion.bvh", loop=False)
frame, bones = source.get_frame_at_time(0.0)
# source.frame_at_time(t) returns only the RetargetFrame.
```

`OfflineBVHAdapter` also exposes this interface directly. Existing `frames=` and
`csv_file=` calls remain supported; pass exactly one source to `open_motion_source`.
BVH reader options are forwarded by the adapter and factory:

- `unit_scale=0.01`: centimetres to metres, including root translation and offsets.
- `world_rotation`: a proper 3×3 rotation applied to the source world; defaults to
  `(x,y,z) → (x,-z,y)`. Pass `np.eye(3)` for already Z-up data. This rotates global
  joint orientations on the left; joint-local axes retain their BVH definitions.
- `joint_mapping`: replacement dictionary from BVH names to `FullBodyBoneId`
  values. Unmapped joints remain in the visualization skeleton. Ambiguous mappings
  of two present joints to the same ID are rejected, including files containing
  both `LeftFoot` and `LeftFootMod` unless an explicit mapping selects one.
- `body_frame="upper_arms"`: the shared upper-body convention; this is the only
  accepted value. Hip landmarks are used only for the separate lower-body frame.
- `wrist_orientation_offsets={"left": R_left, "right": R_right}`: optional
  proper 3×3 rotations post-multiplied onto the BVH wrist orientations.

Sampling holds each frame for `Frame Time`, uses the preceding frame without
interpolation, and scales elapsed time by `playback_speed`. Duration is
`number_of_frames * Frame Time`; at that boundary a non-looping source returns
`None` (or `(None, None)` from `get_frame_at_time`), while looping restarts.
Negative/nonfinite times, invalid units/speed, malformed hierarchy, nonfinite
motion, and inconsistent frame/channel counts are rejected.

Raw bone playback supports partial skeletons. Arm retargeting requires mapped
`ArmUpper`, `ArmLower`, and `HandWrist` on both sides, plus `SpineMiddle`.
Clavicle/`Shoulder` and hip points are not required for BVH arm targets.
Missing required joints raise a descriptive error; a degenerate body basis yields
no action/frame. Leg/torso and head fields are emitted only when their anchors
exist. Wrist, ankle, and head rotations retain BVH joint-local axes, expressed in
the corresponding body frame; XR-specific head/foot alignment is not applied.
The default LaFAN1 mapping supplies no fingers or gripper values. The skeleton
preserves all named BVH joints and their actual parents; `End Site` offsets are
parsed but are not emitted as tracked joints. Robot scaling and IK stay downstream.

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
