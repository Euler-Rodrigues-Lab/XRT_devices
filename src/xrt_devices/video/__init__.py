"""Optional video launch resources; no camera or GPU imports at package load."""
from pathlib import Path


def zed_stream_script():
    """Return the installed ZED/GStreamer-to-MediaMTX Bash launcher."""
    return Path(__file__).with_name("zed_to_mediamtx.sh")
