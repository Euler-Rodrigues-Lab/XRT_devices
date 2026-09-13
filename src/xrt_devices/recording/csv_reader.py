import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ..schemas.body_pose import Bone


class CSVDataReader:
    """Class to read body pose data from CSV file and simulate real-time playback."""

    def __init__(self, csv_file_path, playback_speed=1.0, loop=True, apriltag_quaternion_mode_override=None):
        self.csv_file_path = csv_file_path
        self.playback_speed = playback_speed
        if not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("playback_speed must be finite and positive")
        self.loop = loop
        self.apriltag_quaternion_mode_override = apriltag_quaternion_mode_override
        self.data = None
        self.current_index = 0
        self.start_time = None
        self.csv_start_timestamp = None
        self.time_column = None
        self.is_relative_time = False
        self.apriltag_quaternion_mode = "internal"
        self.load_data()

    def load_data(self):
        """Load CSV data and prepare for playback."""
        print(f"Loading CSV data from {self.csv_file_path}...")
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Loaded {len(self.data)} rows of body pose data")

        # Auto-detect time column format
        if 'time_elapsed' in self.data.columns:
            self.time_column = 'time_elapsed'
            self.is_relative_time = True
            print("Detected relative time format (time_elapsed)")
        elif 'timestamp' in self.data.columns:
            self.time_column = 'timestamp'
            self.is_relative_time = False
            print("Detected absolute timestamp format")
        else:
            raise ValueError("CSV must contain either 'time_elapsed' or 'timestamp' column")

        # Get unique time values to understand data structure
        if self.data.empty or not np.isfinite(self.data[self.time_column].to_numpy(dtype=float)).all():
            raise ValueError("CSV must have nonempty, finite timestamps")
        self.data = self.data.sort_values(self.time_column, kind="stable")
        unique_times = self.data[self.time_column].unique()
        print(f"Data contains {len(unique_times)} unique time values")

        # Store the first time value for timing calculations
        self.csv_start_timestamp = unique_times[0]

        if self.apriltag_quaternion_mode_override in ("internal", "legacy_quat_only", "raw_rows_present"):
            self.apriltag_quaternion_mode = self.apriltag_quaternion_mode_override
        else:
            self.apriltag_quaternion_mode = self._infer_apriltag_quaternion_mode()
        print(f"AprilTag quaternion mode: {self.apriltag_quaternion_mode}")

    def _infer_apriltag_quaternion_mode(self):
        """Infer how apriltag quaternion rows should be interpreted for backward compatibility."""
        if self.data is None or len(self.data) == 0 or "data_type" not in self.data.columns:
            return "internal"

        data_type = self.data["data_type"].astype(str).str.lower()
        if (data_type == "apriltag_raw").any():
            return "raw_rows_present"

        apriltag_rows = self.data[data_type == "apriltag"]
        if len(apriltag_rows) == 0:
            return "internal"

        if self.time_column is None:
            return "internal"

        unique_times = apriltag_rows[self.time_column].dropna().unique()[:200]
        if len(unique_times) == 0:
            return "internal"

        score_internal = 0.0
        score_legacy_q = 0.0
        n_scored = 0
        for t in unique_times:
            frame = apriltag_rows[apriltag_rows[self.time_column] == t]
            if len(frame) < 3:
                continue

            ids = []
            pos = []
            quat = []
            for _, row in frame.iterrows():
                tag_id = _parse_apriltag_id(find_either_or(row, 'bone_id', 'id'))
                if tag_id is None:
                    continue
                ids.append(tag_id)
                pos.append(np.array([
                    _safe_float(row.get('pos_x', 0.0)),
                    _safe_float(row.get('pos_y', 0.0)),
                    _safe_float(row.get('pos_z', 0.0)),
                ], dtype=float))
                quat.append(np.array([
                    _safe_float(row.get('rot_x', 0.0)),
                    _safe_float(row.get('rot_y', 0.0)),
                    _safe_float(row.get('rot_z', 0.0)),
                    _safe_float(row.get('rot_w', 1.0), default=1.0),
                ], dtype=float))

            if len(ids) < 3:
                continue

            spread_internal = _paper_center_spread_from_tag_rows(ids, pos, quat, quat_mode="internal")
            spread_legacy_q = _paper_center_spread_from_tag_rows(ids, pos, quat, quat_mode="legacy_quat_only")

            if np.isfinite(spread_internal) and np.isfinite(spread_legacy_q):
                score_internal += spread_internal
                score_legacy_q += spread_legacy_q
                n_scored += 1

        if n_scored == 0:
            return "internal"

        # Prefer legacy-quaternion conversion only when clearly better.
        if score_legacy_q < 0.99 * score_internal:
            return "legacy_quat_only"
        return "internal"


    def get_duration(self):
        """Get the total duration of the CSV data in seconds."""
        if self.data is None or len(self.data) == 0:
            return 0.0

        unique_times = self.data[self.time_column].unique()
        return float(unique_times[-1] - unique_times[0])

    def get_bones_at_time(self, elapsed_time):
        """Get bone data at a specific elapsed time from start."""
        frame_data = self._get_frame_data_at_time(elapsed_time)
        if frame_data is None:
            return None

        bones, _, _ = self._parse_frame_data(frame_data)
        return bones

    def get_tags_at_time(self, elapsed_time):
        """Get AprilTag data at a specific elapsed time from start."""
        frame_data = self._get_frame_data_at_time(elapsed_time)
        if frame_data is None:
            return {}

        _, tags, _ = self._parse_frame_data(frame_data)
        return tags

    def get_paper_board_pose_at_time(self, elapsed_time):
        """Get optional fused paper-board pose row at a specific elapsed time from start."""
        frame_data = self._get_frame_data_at_time(elapsed_time)
        if frame_data is None:
            return None

        _, _, paper_board_pose = self._parse_frame_data(frame_data)
        return paper_board_pose

    def get_bones_and_tags_at_time(self, elapsed_time):
        """Get both bone and AprilTag data at a specific elapsed time from start."""
        frame_data = self._get_frame_data_at_time(elapsed_time)
        if frame_data is None:
            return None, {}

        bones, tags, _ = self._parse_frame_data(frame_data)
        return bones, tags

    def get_bones_tags_and_board_at_time(self, elapsed_time):
        """Get bone, AprilTag, and optional paper-board data at a specific elapsed time from start."""
        frame_data = self._get_frame_data_at_time(elapsed_time)
        if frame_data is None:
            return None, {}, None

        return self._parse_frame_data(frame_data)

    def _get_frame_data_at_time(self, elapsed_time):
        if not np.isfinite(elapsed_time) or elapsed_time < 0:
            raise ValueError("elapsed_time must be finite and nonnegative")
        times = self.data[self.time_column].unique()
        duration = float(times[-1] - times[0])
        offset = elapsed_time * self.playback_speed
        if duration == 0:
            if offset > 0 and not self.loop:
                return None
            target = times[0]
        else:
            if offset > duration and not self.loop:
                return None
            target = times[0] + (offset % duration if self.loop else offset)
        closest = times[np.argmin(np.abs(times - target))]
        return self.data[self.data[self.time_column] == closest]

    def _parse_frame_data(self, frame_data):
        """Parse a frame DataFrame into Bone list, AprilTag dictionary, and optional paper-board pose."""
        bones = []
        tags = {}
        paper_board_pose = None

        for _, row in frame_data.iterrows():
            row_id = find_either_or(row, 'bone_id', 'id')
            if row_id is None or pd.isna(row_id):
                continue

            row_id_str = str(row_id).strip()
            row_id_str_l = row_id_str.lower()
            data_type = str(row.get('data_type', '')).strip().lower()

            if data_type == 'apriltag_board' or row_id_str_l in ('paper_board_pose', 'apriltag_board'):
                paper_board_pose = {
                    'position': np.array([
                        _safe_float(row.get('pos_x', 0.0)),
                        _safe_float(row.get('pos_y', 0.0)),
                        _safe_float(row.get('pos_z', 0.0)),
                    ]),
                    'quaternion': np.array([
                        _safe_float(row.get('rot_x', 0.0)),
                        _safe_float(row.get('rot_y', 0.0)),
                        _safe_float(row.get('rot_z', 0.0)),
                        _safe_float(row.get('rot_w', 1.0), default=1.0),
                    ]),
                }
                continue

            # Parse AprilTag rows recorded by xr_robot_teleop_client.
            if data_type in ('apriltag', 'apriltag_raw') or row_id_str_l.startswith('apriltag_'):
                tag_id = _parse_apriltag_id(row_id_str)
                if tag_id is None:
                    continue

                p = np.array([
                    _safe_float(row.get('pos_x', 0.0)),
                    _safe_float(row.get('pos_y', 0.0)),
                    _safe_float(row.get('pos_z', 0.0)),
                ])
                q = np.array([
                    _safe_float(row.get('rot_x', 0.0)),
                    _safe_float(row.get('rot_y', 0.0)),
                    _safe_float(row.get('rot_z', 0.0)),
                    _safe_float(row.get('rot_w', 1.0), default=1.0),
                ])

                if data_type == 'apriltag_raw':
                    p, q = _convert_unity_raw_apriltag_to_internal(p, q)
                elif data_type == 'apriltag' and self.apriltag_quaternion_mode == 'legacy_quat_only':
                    q = _legacy_convert_quaternion_only(q)

                tags[tag_id] = {
                    'position': p,
                    'quaternion': q,
                }
                continue

        # Convert to Bone objects
            if row_id_str and row_id_str[0].isalpha():  # row is not a bone (probably an action)
                continue

            try:
                bone_id = int(float(row_id))
            except (TypeError, ValueError):
                continue

            bone = Bone(
                # id=int(row['bone_id'] or row['id']),
                id=bone_id,
                position=[
                    _safe_float(row.get('pos_x', 0.0)),
                    _safe_float(row.get('pos_y', 0.0)),
                    _safe_float(row.get('pos_z', 0.0)),
                ],
                rotation=[
                    _safe_float(row.get('rot_x', 0.0)),
                    _safe_float(row.get('rot_y', 0.0)),
                    _safe_float(row.get('rot_z', 0.0)),
                    _safe_float(row.get('rot_w', 1.0), default=1.0),
                ]
            )
            bones.append(bone)

        return bones, tags, paper_board_pose

def find_either_or(d, a, b):
    if a in d:
        return d[a]
    elif b in d:
        return d[b]


def _safe_float(value, default=0.0):
    try:
        out = float(value)
        return default if np.isnan(out) else out
    except (TypeError, ValueError):
        return default


def _parse_apriltag_id(raw_id):
    tag_id_str = str(raw_id).strip()
    if tag_id_str.lower().startswith('apriltag_'):
        tag_id_str = tag_id_str.split('_', 1)[1]

    try:
        return int(float(tag_id_str))
    except (TypeError, ValueError):
        return None


def _convert_unity_raw_apriltag_to_internal(position_raw, quat_raw):
    """Convert Unity left-handed Y-up tag packet to internal right-handed Z-up frame."""
    p = np.asarray(position_raw, dtype=float).reshape(3)
    q = np.asarray(quat_raw, dtype=float).reshape(4)

    # Position conversion: Unity (x, y, z) -> internal (z, -x, y)
    p_out = np.array([p[2], -p[0], p[1]], dtype=float)

    # Legacy quaternion conversion used by historical recordings.
    q_out = _legacy_convert_quaternion_only(q)
    return p_out, q_out


def _legacy_convert_quaternion_only(quat_raw):
    """Legacy quaternion conversion without position conversion."""
    q = np.asarray(quat_raw, dtype=float).reshape(4)
    return np.array([-q[2], q[0], -q[1], q[3]], dtype=float)


def _quat_to_rot(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    return Rotation.from_quat(q / n).as_matrix()


def _tag_centers_default_board_frame():
    tag_size_m = 0.0784
    tag_gap_m = 0.02
    c = 0.5 * (tag_size_m + tag_gap_m)
    return {
        0: np.array([-c, +c, 0.0], dtype=float),
        1: np.array([+c, +c, 0.0], dtype=float),
        2: np.array([-c, -c, 0.0], dtype=float),
        3: np.array([+c, -c, 0.0], dtype=float),
    }


def _paper_center_spread_from_tag_rows(ids, positions, quaternions, quat_mode="internal"):
    """Compute spread of per-tag paper-center hypotheses for one frame."""
    centers_map = _tag_centers_default_board_frame()
    centers = []
    for tid, p, q in zip(ids, positions, quaternions):
        if tid not in centers_map:
            continue
        p = np.asarray(p, dtype=float).reshape(3)
        q = np.asarray(q, dtype=float).reshape(4)
        if quat_mode == "legacy_quat_only":
            q = _legacy_convert_quaternion_only(q)
        R = _quat_to_rot(q)
        centers.append(p - R @ centers_map[tid])

    if len(centers) < 2:
        return np.inf
    C = np.asarray(centers, dtype=float)
    c_mean = np.mean(C, axis=0)
    return float(np.mean(np.linalg.norm(C - c_mean, axis=1)))
