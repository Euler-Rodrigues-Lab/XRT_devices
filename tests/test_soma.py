"""SOMA (BONES SEED) BVH conversion: mapping, axes, SEED export layout, playback."""
import csv
import gzip
import os
from pathlib import Path
import re

import numpy as np
import pytest

from xrt_devices.recording.bvh_reader import BVHDataReader
from xrt_devices.recording.soma import (
    SEED_HEADER, SEED_ROW_ORDER, SOMA_TO_FLU, SOMA_TO_FULL_BODY, SOMA_UNMAPPED,
    convert_soma_bvh, main, soma_bvh_reader,
)


# Real SOMA joint names and tree; invented offsets (centimetres, SOMA axes:
# +X anatomical left, +Y up, +Z forward) laid out as a world-aligned T pose.
def _chain(names, offsets):
    node = (names[-1], offsets[-1], [])
    for name, offset in zip(reversed(names[:-1]), reversed(offsets[:-1])):
        node = (name, offset, [node])
    return node


def _hand(side, s):
    fingers = []
    fingers.append(_chain([f"{side}HandThumb{k}" for k in ("1", "2", "3", "End")],
                          [(2 * s, -1, 3), (3 * s, 0, 2), (2.5 * s, 0, 1.5), (2.5 * s, 0, 1)]))
    for finger, z, lengths in (("Index", 1.5, (6, 3.5, 2.5, 2.5)), ("Middle", .5, (6.5, 4, 3, 2.5)),
                               ("Ring", -.5, (6, 4, 2.5, 2)), ("Pinky", -1.5, (5.5, 3, 1.5, 2))):
        names = [f"{side}Hand{finger}{k}" for k in ("1", "2", "3", "4", "End")]
        offsets = [(3 * s, 0, z)] + [(length * s, 0, 0) for length in lengths]
        fingers.append(_chain(names, offsets))
    return fingers


def _arm(side, s):
    hand = (f"{side}Hand", (26 * s, 0, 0), _hand(side, s))
    return (f"{side}Shoulder", (5 * s, 12, 0), [(f"{side}Arm", (14 * s, 0, 0),
            [(f"{side}ForeArm", (28 * s, 0, 0), [hand])])])


def _leg(side, s):
    return _chain([f"{side}Leg", f"{side}Shin", f"{side}Foot", f"{side}ToeBase", f"{side}ToeEnd"],
                  [(9 * s, -3, 0), (0, -43, 0), (0, -42, 0), (0, -6, 14), (0, -1.5, 6)])


SOMA_TREE = ("Root", (0, 0, 0), [("Hips", (0, 95, 0), [
    ("Spine1", (0, 8, 0), [("Spine2", (0, 8, 0), [("Chest", (0, 8, 0), [
        ("Neck1", (0, 15, 0), [("Neck2", (0, 5, 0), [("Head", (0, 6, 0), [
            ("HeadEnd", (0, 12, 0), []), ("Jaw", (0, -3, 4), []),
            ("LeftEye", (3, 3, 8), []), ("RightEye", (-3, 3, 8), [])])])]),
        _arm("Left", 1), _arm("Right", -1)])])]),
    _leg("Left", 1), _leg("Right", -1)])])

FRAMES = [{"Hips": [1, 96, 2, 0, 0, 0]},
          {"Hips": [1.5, 96.2, 2.5, 0, 0, 0]},
          {"Hips": [2, 96.4, 3, 0, 0, 0], "LeftHand": [0, 0, 30]}]


def write_soma_bvh(path, frames=FRAMES):
    order = []

    def emit(node, depth, root=False):
        name, offset, children = node
        pad = " " * depth
        count = 6 if name in ("Root", "Hips") else 3
        order.append((name, count))
        spec = ("Xposition Yposition Zposition " if count == 6 else "") + "Zrotation Yrotation Xrotation"
        lines = [f"{pad}{'ROOT' if root else 'JOINT'} {name}", f"{pad}{{",
                 f"{pad} OFFSET {offset[0]} {offset[1]} {offset[2]}", f"{pad} CHANNELS {count} {spec}"]
        for child in children:
            lines += emit(child, depth + 1)
        return lines + [f"{pad}}}"]

    lines = ["HIERARCHY"] + emit(SOMA_TREE, 0, root=True) + ["MOTION", f"Frames: {len(frames)}",
                                                              "Frame Time: 0.008333"]
    for overrides in frames:
        row = [v for name, count in order for v in overrides.get(name, [0] * count)]
        lines.append(" ".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")
    return order


@pytest.fixture
def soma_bvh(tmp_path):
    path = tmp_path / "soma.bvh"
    write_soma_bvh(path)
    return path


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.reader(stream))


def test_mapping_covers_the_soma_hierarchy(tmp_path):
    names = [name for name, _ in write_soma_bvh(tmp_path / "soma.bvh")]
    assert len(names) == 78 and len(SOMA_TO_FULL_BODY) == 70
    assert set(names) - set(SOMA_TO_FULL_BODY) == set(SOMA_UNMAPPED)
    assert len(set(SOMA_TO_FULL_BODY.values())) == 70
    assert len(SEED_ROW_ORDER) == 71
    assert sorted(SEED_ROW_ORDER) == sorted({0, *SOMA_TO_FULL_BODY.values()})
    np.testing.assert_allclose(SOMA_TO_FLU @ SOMA_TO_FLU.T, np.eye(3))
    assert np.linalg.det(SOMA_TO_FLU) == pytest.approx(1)


def test_export_matches_seed_layout(soma_bvh, tmp_path):
    out = convert_soma_bvh(soma_bvh, tmp_path / "seed.csv")
    rows = read_rows(out)
    assert rows[0] == list(SEED_HEADER)
    body = rows[1:]
    assert len(body) == 3 * 71
    for frame, expected_time in enumerate(("0.0000", "0.0083", "0.0167")):
        chunk = body[frame * 71:(frame + 1) * 71]
        assert [int(r[1]) for r in chunk] == list(SEED_ROW_ORDER)
        assert {r[0] for r in chunk} == {expected_time}
        by_id = {int(r[1]): r[2:] for r in chunk}
        assert by_id[8] == by_id[10] and by_id[13] == by_id[15]
        assert by_id[0][:2] == by_id[1][:2] and by_id[0][3:] == ["0.0000", "0.0000", "0.0000", "1.0000"]
        assert float(by_id[0][2]) < float(by_id[76][2]) < float(by_id[73][2])
    assert all(re.fullmatch(r"-?\d+\.\d{4}", v) for r in body for v in (r[0], *r[2:]))
    # Root height is the lowest SOMA joint, including the unmapped toe ends.
    reader = soma_bvh_reader(soma_bvh)
    transforms = reader.transforms_at_frame(0)
    lowest = min(transforms[i, 2, 3] for i, j in enumerate(reader.joints) if j.name != "Root")
    assert body[0][4] == "%.4f" % lowest
    assert reader.joints[int(np.argmin(transforms[:, 2, 3][1:])) + 1].name == "LeftToeEnd"


def test_axes_and_hips_translation(soma_bvh):
    reader = soma_bvh_reader(soma_bvh)
    transforms = reader.transforms_at_frame(0)
    p = {j.name: transforms[i, :3, 3] for i, j in enumerate(reader.joints)}
    # Hips = position channel only (1, 96, 2) cm, mapped SOMA (x,y,z) -> FLU (z, x, y).
    np.testing.assert_allclose(p["Hips"], [.02, .01, .96], atol=1e-12)
    assert p["LeftArm"][1] > 0 > p["RightArm"][1]
    assert p["LeftToeBase"][0] > p["LeftFoot"][0] and p["LeftEye"][0] > p["Head"][0]
    assert p["Head"][2] > p["Hips"][2] > p["LeftFoot"][2]
    added = soma_bvh_reader(soma_bvh, translation_channels="add").transforms_at_frame(0)
    assert added[1, 2, 3] == pytest.approx(.95 + .96)
    assert [bid for _, bid in reader.mapped_joints] == [SOMA_TO_FULL_BODY[j.name]
                                                        for j in reader.joints if j.name in SOMA_TO_FULL_BODY]


def test_clavicle_option_and_stride(soma_bvh, tmp_path):
    out = convert_soma_bvh(soma_bvh, tmp_path / "clavicle.csv", shoulders="clavicle", stride=2)
    rows = read_rows(out)[1:]
    assert sorted({r[0] for r in rows}) == ["0.0000", "0.0167"] and len(rows) == 2 * 71
    by_id = {int(r[1]): r[2:] for r in rows[:71]}
    assert by_id[8] != by_id[10] and by_id[13] != by_id[15]
    reader = soma_bvh_reader(soma_bvh)
    clavicle = reader.transforms_at_frame(0)[[j.name for j in reader.joints].index("LeftShoulder"), :3, 3]
    assert by_id[8][:3] == ["%.4f" % v for v in clavicle]
    with pytest.raises(ValueError):
        convert_soma_bvh(soma_bvh, tmp_path / "bad.csv", shoulders="both")
    with pytest.raises(ValueError):
        convert_soma_bvh(soma_bvh, tmp_path / "bad.csv", stride=0)


def test_gzip_round_trip_through_csv_reader_and_processing(soma_bvh, tmp_path):
    pytest.importorskip("pandas")
    from xrt_devices.recording.csv_reader import CSVDataReader
    from xrt_devices.processing import bones_to_action
    out = convert_soma_bvh(soma_bvh, tmp_path / "seed.csv.gz")
    reader = CSVDataReader(str(out), loop=False)
    bones = reader.get_bones_at_time(0)
    assert sorted(b.id for b in bones) == sorted(SEED_ROW_ORDER)
    action = bones_to_action(bones)
    for key in ("left_sew", "right_sew"):
        assert action[key].shape == (18,) and np.isfinite(action[key]).all()
    assert "left_fingers" in action and "left_hka" in action
    assert reader.get_duration() == pytest.approx(.0167)


def test_direct_playback_matches_export(soma_bvh, tmp_path):
    rows = read_rows(convert_soma_bvh(soma_bvh, tmp_path / "seed.csv"))[1:72]
    exported = {int(r[1]): np.array(r[2:5], dtype=float) for r in rows}
    action, bones = soma_bvh_reader(soma_bvh, loop=False).get_action_and_bones_at_time(0)
    assert {b.id for b in bones} == set(SOMA_TO_FULL_BODY.values())
    for bone in bones:
        if bone.id not in (8, 13):
            np.testing.assert_allclose(bone.position, exported[bone.id], atol=5e-5)
    assert np.isfinite(action["left_sew"]).all() and action["skeleton"]["names"][0] == "Root"


def test_translation_channels_on_generic_bvh(tmp_path):
    path = tmp_path / "pelvis.bvh"
    path.write_text("HIERARCHY\nROOT Pelvis { OFFSET 0 10 0 CHANNELS 3 Xposition Yposition Zposition }\n"
                    "MOTION\nFrames: 1\nFrame Time: 0.5\n0 5 0\n")
    added = BVHDataReader(path, joint_mapping={"Pelvis": 1}).get_bones_at_time(0)[0].position
    replaced = BVHDataReader(path, joint_mapping={"Pelvis": 1},
                             translation_channels="replace").get_bones_at_time(0)[0].position
    np.testing.assert_allclose(added, [0, 0, .15], atol=1e-14)
    np.testing.assert_allclose(replaced, [0, 0, .05], atol=1e-14)
    with pytest.raises(ValueError, match="translation_channels"):
        BVHDataReader(path, translation_channels="ignore")
    with pytest.raises(IndexError):
        BVHDataReader(path).transforms_at_frame(1)


def test_cli(soma_bvh, tmp_path, capsys):
    out = tmp_path / "cli.csv.gz"
    main([str(soma_bvh), str(out), "--stride", "3"])
    assert str(out) in capsys.readouterr().out
    assert len(read_rows(out)) == 1 + 71


@pytest.mark.skipif(not (os.environ.get("XRT_SOMA_GOLDEN_BVH") and os.environ.get("XRT_SOMA_GOLDEN_CSV")),
                    reason="set XRT_SOMA_GOLDEN_BVH and XRT_SOMA_GOLDEN_CSV to compare with a shipped SEED export")
def test_golden_seed_export(tmp_path):
    out = convert_soma_bvh(os.environ["XRT_SOMA_GOLDEN_BVH"], tmp_path / "golden.csv")
    expected = Path(os.environ["XRT_SOMA_GOLDEN_CSV"])
    reference = gzip.open(expected, "rb").read() if expected.suffix == ".gz" else expected.read_bytes()
    assert out.read_bytes() == reference
