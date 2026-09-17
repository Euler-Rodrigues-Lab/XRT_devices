"""SOMA (BONES SEED) BVH to OpenXR FullBody bone CSV.

BONES ships the SEED motion-capture dataset as BVH files carrying the SOMA rig:
78 joints, centimetres, 120 fps, Y-up with +X anatomical left and +Z forward.
This module maps that rig onto the ``FullBodyBoneId`` namespace and writes the
CSV layout of the SEED exports replayed by ``CSVDataReader`` and WARP::

    time_elapsed,bone_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w

- positions in metres, right-handed FLU (+X forward, +Y left, +Z up);
- rotations are global XYZW quaternions of ``SOMA_TO_FLU @ R``: the source
  world is rotated, joint-local axes keep their SOMA definitions, exactly as
  every ``BVHDataReader`` world rotation behaves;
- the Hips position channels replace the Hips ``OFFSET`` (BONES parser
  semantics); adding both would lift the body by about one metre;
- 70 SOMA joints map to FullBody IDs. ``Root`` (0) is synthesized at the Hips
  x/y and the lowest SOMA joint height with identity rotation. SOMA's inert
  ``Root``, ``Neck1``, ``HeadEnd``, ``Jaw``, both eyes and both toe ends are
  dropped;
- shipped SEED exports carry ``LeftArm``/``RightArm`` in IDs 8/13 as well as
  10/15 (``shoulders="duplicate_arm"``); ``shoulders="clavicle"`` writes SOMA's
  ``LeftShoulder``/``RightShoulder`` there instead;
- one row per bone in the shipped order, ``%.4f`` values, ``time_elapsed`` is
  frame index times the file's literal ``Frame Time``; ``stride`` subsamples.

Usage::

    python -m xrt_devices.recording.soma motion.bvh motion.csv.gz
    python -m xrt_devices.recording.soma motion.bvh motion.csv --stride 4 --shoulders clavicle

``soma_bvh_reader`` returns a ``BVHDataReader`` preconfigured for SOMA files so
a motion can also be replayed without conversion. Only numpy/scipy are needed.
SEED motion data is licensed separately by BONES and is not part of this package.
"""
import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh_reader import BVHDataReader

# SOMA BVH world: +X anatomical left, +Y up, +Z forward  ->  FLU (x, y, z) = (z, x, y).
SOMA_TO_FLU = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])

SOMA_TO_FULL_BODY = {
    "Hips": 1, "Spine1": 2, "Spine2": 3, "Chest": 4, "Neck2": 6, "Head": 7,
    "LeftShoulder": 8, "LeftArm": 10, "LeftForeArm": 11, "LeftHand": 19,
    "RightShoulder": 13, "RightArm": 15, "RightForeArm": 16, "RightHand": 45,
    "LeftLeg": 70, "LeftShin": 71, "LeftFoot": 73, "LeftToeBase": 76,
    "RightLeg": 77, "RightShin": 78, "RightFoot": 80, "RightToeBase": 83,
}
for _side, _base in (("Left", 20), ("Right", 46)):
    # Thumb 1,2,3,End -> Metacarpal, Proximal, Distal, Tip.
    SOMA_TO_FULL_BODY.update({f"{_side}HandThumb{k}": _base + i
                              for i, k in enumerate(("1", "2", "3", "End"))})
    # Other fingers 1,2,3,4,End -> Metacarpal, Proximal, Intermediate, Distal, Tip.
    for _f, _finger in enumerate(("Index", "Middle", "Ring", "Pinky")):
        SOMA_TO_FULL_BODY.update({f"{_side}Hand{_finger}{k}": _base + 4 + 5 * _f + i
                                  for i, k in enumerate(("1", "2", "3", "4", "End"))})
del _side, _base, _f, _finger

SOMA_UNMAPPED = ("Root", "Neck1", "HeadEnd", "Jaw", "LeftEye", "RightEye", "LeftToeEnd", "RightToeEnd")

# Row order of the shipped SEED exports: spine, left arm, right arm, legs, fingers.
SEED_ROW_ORDER = (0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 19, 13, 15, 16, 45,
                  70, 71, 73, 76, 77, 78, 80, 83, *range(20, 44), *range(46, 70))
SEED_HEADER = ("time_elapsed", "bone_id", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w")

SOMA_READER_KWARGS = {"unit_scale": .01, "world_rotation": SOMA_TO_FLU,
                      "joint_mapping": SOMA_TO_FULL_BODY, "translation_channels": "replace"}


def soma_bvh_reader(bvh_file, **kwargs):
    """``BVHDataReader`` with SOMA units, axes, joint mapping and Hips translation."""
    return BVHDataReader(bvh_file, **{**SOMA_READER_KWARGS, **kwargs})


def convert_soma_bvh(bvh_file, out_file, *, stride=1, shoulders="duplicate_arm"):
    """Write a SOMA BVH motion as a SEED FullBody bone CSV (``.csv`` or ``.csv.gz``)."""
    if shoulders not in ("duplicate_arm", "clavicle"):
        raise ValueError("shoulders must be 'duplicate_arm' or 'clavicle'")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    reader = soma_bvh_reader(bvh_file, loop=False)
    names = [joint.name for joint in reader.joints]
    missing = sorted(set(SOMA_TO_FULL_BODY) - set(names))
    if missing:
        raise ValueError(f"{bvh_file} is not a SOMA hierarchy; missing joints such as {missing[:4]}")
    hips = names.index("Hips")
    body = [index for index, name in enumerate(names) if name != "Root"]
    indices = [index for index, _ in reader.mapped_joints]
    out_file = Path(out_file)
    opener = gzip.open if out_file.suffix == ".gz" else open
    with opener(out_file, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SEED_HEADER)
        for frame in range(0, reader.num_frames, int(stride)):
            transforms = reader.transforms_at_frame(frame)
            quaternions = Rotation.from_matrix(transforms[indices, :3, :3]).as_quat()
            rows = {bone_id: (transforms[index, :3, 3], quaternions[k])
                    for k, (index, bone_id) in enumerate(reader.mapped_joints)}
            root = np.array([*transforms[hips, :2, 3], transforms[body, 2, 3].min()])
            rows[0] = (root, np.array([0., 0., 0., 1.]))
            if shoulders == "duplicate_arm":
                rows[8], rows[13] = rows[10], rows[15]
            elapsed = "%.4f" % (frame * reader.frame_time)
            for bone_id in SEED_ROW_ORDER:
                position, rotation = rows[bone_id]
                writer.writerow((elapsed, bone_id, *("%.4f" % value for value in (*position, *rotation))))
    return out_file


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m xrt_devices.recording.soma",
        description="Convert a SOMA (BONES SEED) BVH motion into the SEED FullBody bone CSV "
                    "replayed by CSVDataReader and WARP.")
    parser.add_argument("bvh", help="SOMA BVH motion file")
    parser.add_argument("out", help="output path; .gz suffix selects gzip")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame (source is 120 fps)")
    parser.add_argument("--shoulders", choices=("duplicate_arm", "clavicle"), default="duplicate_arm",
                        help="IDs 8/13: copy the upper arms like shipped SEED exports (default) "
                             "or write SOMA's clavicles")
    args = parser.parse_args(argv)
    print(convert_soma_bvh(args.bvh, args.out, stride=args.stride, shoulders=args.shoulders))


if __name__ == "__main__":
    main()
