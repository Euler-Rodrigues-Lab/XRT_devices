"""Draw Tasks normalized landmarks without the retired Solutions drawing API."""
import math

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (18, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
                    (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
                    (0, 17), (17, 18), (18, 19), (19, 20))


def draw_landmarks(image, landmarks, connections, *args, **kwargs):
    import cv2
    height, width = image.shape[:2]
    points = {}
    for index, point in enumerate(landmarks.landmark):
        visibility = getattr(point, "visibility", None)
        presence = getattr(point, "presence", None)
        if ((visibility is not None and visibility < 0.5) or
                (presence is not None and presence < 0.5)):
            continue
        if math.isfinite(point.x) and math.isfinite(point.y) and 0 <= point.x <= 1 and 0 <= point.y <= 1:
            points[index] = (min(int(point.x * width), width - 1),
                             min(int(point.y * height), height - 1))
    for start, end in connections:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], (0, 220, 0), 2)
    for point in points.values():
        cv2.circle(image, point, 3, (0, 100, 255), -1)
