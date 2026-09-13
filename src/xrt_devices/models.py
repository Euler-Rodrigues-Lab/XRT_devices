"""On-demand, versioned MediaPipe model cache (no downloads at import time)."""
import logging
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

_ROOT = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "pose": f"{_ROOT}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": f"{_ROOT}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def _valid_bundle(path):
    try:
        with zipfile.ZipFile(path) as bundle:
            return bool(bundle.namelist()) and bundle.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def resolve_model(kind, path=None, *, cache_dir=None):
    """Use an explicit file unchanged, or download/cache a default Tasks bundle.

    Set XRT_DEVICES_MODEL_DIR to override the default XDG user cache location.
    Explicit missing paths fail without silently downloading a replacement.
    """
    url = MODEL_URLS[kind]
    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"MediaPipe {kind} model does not exist: {explicit}")
        return explicit
    root = Path(cache_dir or os.environ.get("XRT_DEVICES_MODEL_DIR") or
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) /
                "xrt_devices" / "models").expanduser()
    target = root / f"{kind}-float16-v1.task"
    if _valid_bundle(target):
        return target
    temporary = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        logging.getLogger(__name__).warning("Downloading MediaPipe %s model to %s", kind, target)
        with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise ValueError("Model download exceeds 100 MiB")
                    output.write(chunk)
        if not _valid_bundle(temporary):
            raise ValueError("Downloaded model is not a valid Tasks bundle")
        os.replace(temporary, target)
        return target
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not download MediaPipe {kind} model from {url}. "
            f"Check network/cache permissions or provide --{kind}_model /path/to/model.task."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
