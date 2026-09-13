"""Adapt Tasks results to the transferred landmark-processing code.

Both models process the same camera image sequentially on one worker. VIDEO mode
provides tracking without asynchronous pose/hand results from different images.
"""
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
import time
from ._drawing import draw_landmarks, POSE_CONNECTIONS, HAND_CONNECTIONS


class PoseLandmark(IntEnum):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


HandLandmark = IntEnum("HandLandmark", dict(zip((
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
), range(21))))


def pose_result(result):
    return NS(
        pose_landmarks=NS(landmark=result.pose_landmarks[0]) if result.pose_landmarks else None,
        pose_world_landmarks=NS(landmark=result.pose_world_landmarks[0]) if result.pose_world_landmarks else None,
    )


def hand_result(result):
    return NS(
        multi_hand_landmarks=[NS(landmark=items) for items in result.hand_landmarks],
        multi_hand_world_landmarks=[NS(landmark=items) for items in result.hand_world_landmarks],
        multi_handedness=[NS(classification=[NS(label=c.category_name, score=c.score) for c in items])
                          for items in result.handedness],
    )


def make_api(pose_model, hand_model):
    for path in (pose_model, hand_model):
        if not Path(path).is_file():
            raise FileNotFoundError(f"MediaPipe Tasks model not found: {path}")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    class Detector:
        def __init__(self, hand=False):
            self.hand = hand
            self.timestamp = -1
            options = dict(base_options=BaseOptions(model_asset_path=str(hand_model if hand else pose_model)),
                           running_mode=vision.RunningMode.VIDEO)
            if hand:
                self.model = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(num_hands=2, **options))
            else:
                self.model = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(num_poses=1, **options))
        def process(self, rgb):
            self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
            result = self.model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp)
            return hand_result(result) if self.hand else pose_result(result)
        def close(self):
            self.model.close()

    # Tasks landmarks are ordinary objects, not Solutions protobuf messages.
    noop = lambda *args, **kwargs: None
    styles = NS(**{name: noop for name in (
        "get_default_pose_landmarks_style", "get_default_hand_landmarks_style",
        "get_default_hand_connections_style")})
    return NS(solutions=NS(
        pose=NS(Pose=lambda **kw: Detector(), PoseLandmark=PoseLandmark, POSE_CONNECTIONS=POSE_CONNECTIONS),
        hands=NS(Hands=lambda **kw: Detector(hand=True), HandLandmark=HandLandmark, HAND_CONNECTIONS=HAND_CONNECTIONS),
        drawing_utils=NS(draw_landmarks=draw_landmarks), drawing_styles=styles))
