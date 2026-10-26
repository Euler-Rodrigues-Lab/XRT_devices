"""Elbow-independent XR hand-pose acquisition using the established hand frames."""
import numpy as np
from xrt_devices.processing import (
    _get_body_frame, _get_hand_frame, _get_finger_coordinates,
    _convert_bones_to_finger_dict, simplified_finger_bones,
)
from xrt_devices.schemas.openxr_skeletons import FullBodyBoneId


def bones_to_eef_action(bones, tags=None):
    positions = {b.id: np.asarray(b.position, float) for b in bones if np.isfinite(b.position).all()}
    body_ids = (FullBodyBoneId.FullBody_LeftArmUpper, FullBodyBoneId.FullBody_RightArmUpper, FullBodyBoneId.FullBody_SpineMiddle)
    if any(key not in positions for key in body_ids):
        return None
    left, right, spine = (positions[key] for key in body_ids)
    if np.linalg.norm(np.cross(left-right, (left+right)*0.5-spine)) < 1e-8:
        return None
    origin, rotation = _get_body_frame(positions)
    if origin is None or rotation is None:
        return None
    absolute = _get_finger_coordinates(bones, 'absolute', simplified_finger_bones)
    action = {'p_world_upper_body': origin, 'R_world_upper_body': rotation}
    for side in ('left', 'right'):
        wrist = positions.get(getattr(FullBodyBoneId, f'FullBody_{side.title()}HandWrist'))
        required = [getattr(FullBodyBoneId, f'FullBody_{side.title()}Hand'+suffix)
                    for suffix in ('Wrist','IndexProximal','LittleProximal')]
        if any(key not in positions or not np.isfinite(positions[key]).all() for key in required):
            continue
        hand_rotation = _get_hand_frame(side, positions)
        if wrist is None or hand_rotation is None:
            continue
        local = {key: None if value is None else hand_rotation.T @ (value-wrist)
                 for key, value in absolute.items()}
        action[f'{side}_hand_pose'] = {'position': wrist, 'rotation': hand_rotation}
        named = {FullBodyBoneId(key).name: value for key, value in positions.items()}
        action[f'{side}_fingers'] = complete_hand_landmarks(
            named, side, wrist, hand_rotation, _convert_bones_to_finger_dict(local, side))
    return action


def complete_hand_landmarks(positions, side, wrist, rotation, existing=None):
    """Recover all measured XR finger joints in wrist coordinates.

    `positions` maps full OpenXR bone names to world positions. Existing payload
    values take precedence so legacy MCP/PIP/TIP data remains unchanged.
    """
    result = {finger: dict(points) for finger, points in (existing or {}).items()}
    for finger, bone in [('thumb','Thumb'), ('index','Index'), ('middle','Middle'),
                         ('ring','Ring'), ('pinky','Little')]:
        prefix = finger if finger in ('thumb','pinky') else finger+'_finger'
        slots = [('cmc','Metacarpal'),('mcp','Proximal'),('ip','Distal'),('tip','Tip')] if finger=='thumb' else [
            ('mcp','Proximal'),('pip','Intermediate'),('dip','Distal'),('tip','Tip')]
        points = result.setdefault(finger, {})
        for slot, suffix in slots:
            point = positions.get(f'FullBody_{side.title()}Hand{bone}{suffix}')
            if point is not None and np.isfinite(point).all():
                points.setdefault(prefix+'_'+slot, np.asarray(rotation).T @ (np.asarray(point)-wrist))
    return result
