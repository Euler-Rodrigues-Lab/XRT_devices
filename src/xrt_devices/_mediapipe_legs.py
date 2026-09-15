"""Measured MediaPipe leg targets in a pelvis frame; no occlusion inference."""
import numpy as np


def leg_targets(result):
    targets = {'left_hka': None, 'right_hka': None}
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return targets
    image = result.pose_landmarks.landmark
    world = result.pose_world_landmarks.landmark

    def point(i):
        p, w = image[i], world[i]
        xyz = np.array([w.x, w.y, w.z], dtype=float)
        if (getattr(p, 'visibility', 0) < .75
                or not 0 <= p.x <= 1 or not 0 <= p.y <= 1
                or not np.isfinite(xyz).all()):
            return None
        return xyz

    def unit(v):
        norm = np.linalg.norm(v)
        return None if norm < 1e-6 else v / norm

    anchors = [point(i) for i in (11, 12, 23, 24)]
    if any(p is None for p in anchors):
        return targets
    ls, rs, lh, rh = anchors
    origin = (lh + rh) / 2
    y = unit(lh - rh)
    if y is None:
        return targets
    x = unit(np.cross(y, (ls + rs) / 2 - origin))
    if x is None:
        return targets
    rotation = np.column_stack((x, y, np.cross(x, y)))
    for side, ids in [('left', (23, 25, 27, 29, 31)),
                      ('right', (24, 26, 28, 30, 32))]:
        points = [point(i) for i in ids]
        if any(p is None for p in points):
            continue
        hip, knee, ankle, heel, toe = points
        fx = unit(toe - heel)
        if fx is None:
            continue
        fy = unit(np.cross(ankle - heel, fx))
        if fy is None:
            continue
        foot = np.column_stack((fx, fy, np.cross(fx, fy)))
        targets[side + '_hka'] = dict(
            H=rotation.T @ (hip-origin), K=rotation.T @ (knee-origin),
            A=rotation.T @ (ankle-origin), ankle_rot=rotation.T @ foot)
    return targets
