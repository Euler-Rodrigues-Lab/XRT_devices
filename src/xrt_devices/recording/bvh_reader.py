"""BVH playback with explicit LaFAN joint, unit, and world-axis conventions.

The LaFAN mapping follows SEW-Geometric-Teleop's offline_lafan_bvh_reader.
No robot scales or IK offsets are applied. Only numpy/scipy are required.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from ..schemas.body_pose import Bone


BVH_TO_FULL_BODY = {
    "Hips": 1, "Spine": 2, "Spine1": 3, "Spine2": 4, "Chest": 5,
    "Neck": 6, "Head": 7,
    "LeftShoulder": 8, "LeftScapula": 9, "LeftArm": 10, "LeftForeArm": 11,
    "LeftHand": 19, "RightShoulder": 13, "RightScapula": 14,
    "RightArm": 15, "RightForeArm": 16, "RightHand": 45,
    "LeftUpLeg": 70, "LeftLeg": 71, "LeftFoot": 73, "LeftFootMod": 73,
    "LeftToe": 76, "RightUpLeg": 77, "RightLeg": 78,
    "RightFoot": 80, "RightFootMod": 80, "RightToe": 83,
}
Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


class BVHParseError(ValueError):
    """Invalid or unsupported BVH structure or motion data."""


@dataclass
class BVHJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int


def _rotation(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(value), 1., atol=1e-7)):
        raise ValueError(f"{name} must be a finite proper 3x3 rotation")
    return value.copy()


class BVHDataReader:
    """Sample BVH frames using floor(time / Frame Time), without interpolation.

    Defaults: LaFAN names, centimetres, right-handed Y-up -> Z-up world.
    ``joint_mapping`` replaces the name-to-FullBody-ID map. ``world_rotation``
    rotates the source world, leaving each joint's local axes unchanged.
    Duration is N * Frame Time; non-looping playback ends at that boundary.
    """

    def __init__(self, bvh_file, playback_speed=1., loop=True, *, unit_scale=.01,
                 world_rotation=None, joint_mapping=None, body_frame="upper_arms",
                 wrist_orientation_offsets=None):
        self.bvh_file = Path(bvh_file)
        for name, value in (("playback_speed", playback_speed), ("unit_scale", unit_scale)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if body_frame != "upper_arms":
            raise ValueError("upper-body frame must use body_frame='upper_arms'")
        self.playback_speed, self.unit_scale = float(playback_speed), float(unit_scale)
        self.loop, self.body_frame = loop, body_frame
        self.world_rotation = _rotation(
            Y_UP_TO_Z_UP if world_rotation is None else world_rotation, "world_rotation")
        self.joint_mapping = dict(BVH_TO_FULL_BODY if joint_mapping is None else joint_mapping)
        if any(not isinstance(v, (int, np.integer)) or not 0 <= v < 84
               for v in self.joint_mapping.values()):
            raise ValueError("joint_mapping values must be FullBody IDs 0–83")
        self.wrist_offsets = {}
        for side, matrix in (wrist_orientation_offsets or {}).items():
            if side not in ("left", "right"):
                raise ValueError("wrist_orientation_offsets keys must be left/right")
            self.wrist_offsets[side] = _rotation(matrix, f"{side} wrist offset")
        self._parse()
        self._mapped = [(i, self.joint_mapping[j.name]) for i, j in enumerate(self.joints)
                        if j.name in self.joint_mapping]
        ids = [bone_id for _, bone_id in self._mapped]
        if len(set(ids)) != len(ids):
            raise ValueError("multiple BVH joints map to the same bone ID; supply joint_mapping")

    def _parse(self):
        text = self.bvh_file.read_text(encoding="utf-8-sig")
        parts = re.split(r"\bMOTION\b", text)
        if len(parts) != 2:
            raise BVHParseError("expected one MOTION section")
        tokens = iter(re.findall(r"[{}]|[^\s{}]+", parts[0]))
        self.joints = []
        names = set()
        channel_count = 0

        def expect(word):
            got = next(tokens)
            if got != word:
                raise BVHParseError(f"expected {word}, got {got}")

        def offset():
            expect("OFFSET")
            values = np.array([float(next(tokens)) for _ in range(3)])
            if not np.isfinite(values).all():
                raise BVHParseError("non-finite OFFSET")
            return values

        def joint(parent):
            nonlocal channel_count
            name = next(tokens)
            if name in names:
                raise BVHParseError(f"duplicate joint name: {name}")
            names.add(name)
            expect("{")
            xyz = offset()
            expect("CHANNELS")
            count = int(next(tokens))
            if not 0 <= count <= 6:
                raise BVHParseError("CHANNELS count must be 0–6")
            channels = tuple(next(tokens) for _ in range(count))
            allowed = {a + kind for a in "XYZ" for kind in ("position", "rotation")}
            if len(set(channels)) != count or not set(channels) <= allowed:
                raise BVHParseError("invalid or duplicate channel")
            index = len(self.joints)
            self.joints.append(BVHJoint(name, parent, xyz, channels, channel_count))
            channel_count += count
            while True:
                token = next(tokens)
                if token == "}":
                    break
                if token == "JOINT":
                    joint(index)
                elif token == "End":
                    expect("Site")
                    expect("{")
                    offset()  # End sites are terminal offsets, not tracked joints.
                    expect("}")
                else:
                    raise BVHParseError(f"unexpected hierarchy token: {token}")

        try:
            expect("HIERARCHY")
            expect("ROOT")
            joint(-1)
            if next(tokens, None) is not None:
                raise BVHParseError("unexpected trailing hierarchy data")
            lines = [line.strip() for line in parts[1].splitlines() if line.strip()]
            match = re.fullmatch(r"Frames\s*:\s*(\d+)", lines[0])
            timing = re.fullmatch(r"Frame\s+Time\s*:\s*(\S+)", lines[1])
            if not match or not timing:
                raise BVHParseError("expected Frames and Frame Time headers")
            self.num_frames = int(match[1])
            self.frame_time = float(timing[1])
            if self.num_frames <= 0 or not np.isfinite(self.frame_time) or self.frame_time <= 0:
                raise BVHParseError("frame count and Frame Time must be positive and finite")
            rows = [[float(v) for v in line.split()] for line in lines[2:]]
            if len(rows) != self.num_frames or any(len(row) != channel_count for row in rows):
                raise BVHParseError("motion row count or channel count does not match headers")
            self.motion = np.asarray(rows, dtype=float)
            if not np.isfinite(self.motion).all():
                raise BVHParseError("non-finite motion data")
        except (StopIteration, IndexError, ValueError) as exc:
            if isinstance(exc, BVHParseError):
                raise
            raise BVHParseError(f"invalid BVH: {exc}") from exc

    def get_duration(self):
        return self.num_frames * self.frame_time

    def _frame_index(self, elapsed_time):
        if not np.isfinite(elapsed_time) or elapsed_time < 0:
            raise ValueError("elapsed_time must be finite and nonnegative")
        offset = float(elapsed_time) * self.playback_speed
        if not np.isfinite(offset):
            raise ValueError("scaled playback time must be finite")
        if not self.loop and offset >= self.get_duration():
            return None
        if self.loop:
            offset %= self.get_duration()
        return min(int(np.floor(offset / self.frame_time)), self.num_frames - 1)

    def transforms_at_time(self, elapsed_time):
        """World joint matrices in metres; None when playback has finished."""
        index = self._frame_index(elapsed_time)
        if index is None:
            return None
        world = np.eye(4)
        world[:3, :3] = self.world_rotation
        transforms = []
        for joint in self.joints:
            local = np.eye(4)
            local[:3, 3] = joint.offset * self.unit_scale
            # Compose in declared order, including interleaved translation channels.
            for k, channel in enumerate(joint.channels):
                value = self.motion[index, joint.channel_start + k]
                operation = np.eye(4)
                if channel.endswith("position"):
                    operation["XYZ".index(channel[0]), 3] = value * self.unit_scale
                else:
                    operation[:3, :3] = Rotation.from_euler(channel[0], value, degrees=True).as_matrix()
                local = local @ operation
            parent = world if joint.parent < 0 else transforms[joint.parent]
            transforms.append(parent @ local)
        return np.asarray(transforms)

    def _bones(self, transforms):
        return [Bone(int(bid), transforms[i, :3, 3].copy(),
                     Rotation.from_matrix(transforms[i, :3, :3]).as_quat())
                for i, bid in self._mapped]

    def get_bones_at_time(self, elapsed_time):
        transforms = self.transforms_at_time(elapsed_time)
        return None if transforms is None else self._bones(transforms)

    def get_action_and_bones_at_time(self, elapsed_time):
        transforms = self.transforms_at_time(elapsed_time)
        if transforms is None:
            return None, None
        bones = self._bones(transforms)
        poses = {bid: transforms[i] for i, bid in self._mapped}
        required = {10, 11, 19, 15, 16, 45, 3}
        missing = required - poses.keys()
        if missing:
            raise ValueError(f"BVH retargeting missing required FullBody IDs: {sorted(missing)}")
        p = {bid: pose[:3, 3] for bid, pose in poses.items()}
        origin = (p[10] + p[15]) / 2
        lateral = p[10] - p[15]
        upper = _body_basis(lateral, origin - p[3])
        if upper is None:
            return None, bones
        action = {"R_world_upper_body": upper, "p_world_upper_body": origin,
                  "skeleton": {"names": tuple(j.name for j in self.joints),
                               "parents": np.array([j.parent for j in self.joints]),
                               "positions": transforms[:, :3, 3].copy()}}
        for side, ids in (("left", (10, 11, 19)), ("right", (15, 16, 45))):
            points = [upper.T @ (p[bid] - origin) for bid in ids]
            wrist = upper.T @ poses[ids[-1]][:3, :3] @ self.wrist_offsets.get(side, np.eye(3))
            action[f"{side}_sew"] = np.concatenate([*points, wrist.ravel()])
        if 7 in poses:
            action["head_rotation"] = upper.T @ poses[7][:3, :3]
        if {70, 77} <= poses.keys():
            center = (p[70] + p[77]) / 2
            lower = _body_basis(p[70] - p[77], p[3] - center)
            if lower is not None:
                action["R_lower_upper"] = lower.T @ upper
                for side, ids in (("left", (70, 71, 73)), ("right", (77, 78, 80))):
                    if set(ids) <= poses.keys():
                        leg = {key: lower.T @ (p[bid] - center) for key, bid in zip("HKA", ids)}
                        leg.update(ankle_rot=lower.T @ poses[ids[-1]][:3, :3],
                                   A_world=p[ids[-1]].copy(), hip_center_world=center.copy())
                        action[f"{side}_hka"] = leg
        if 2 in p:
            action["body_center"] = p[2].copy()
            if {73, 80} <= p.keys():
                action["ankle_to_body"] = p[2] - (p[73] + p[80]) / 2
        return action, bones

    def get_action_at_time(self, elapsed_time):
        return self.get_action_and_bones_at_time(elapsed_time)[0]


def _body_basis(lateral, up):
    norm = np.linalg.norm(lateral)
    if norm < 1e-8:
        return None
    y = lateral / norm
    x = np.cross(y, up)
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return None
    x /= norm
    return np.column_stack((x, y, np.cross(x, y)))
