"""Smoke-check an installed release, outside the source checkout."""
from importlib.metadata import version
assert version('eurola-xrt-devices') == "0.1.1"
import xrt_devices
assert xrt_devices.__version__ == '0.1.1'
print("Release smoke check passed")
