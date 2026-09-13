from io import BytesIO
import zipfile

import pytest

from xrt_devices import models


def test_download_and_cache(monkeypatch, tmp_path):
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as bundle:
        bundle.writestr("model.tflite", b"fixture")
    calls = []

    def download(url, timeout):
        calls.append(url)
        return BytesIO(payload.getvalue())

    monkeypatch.setattr(models, "urlopen", download)
    path = models.resolve_model("pose", cache_dir=tmp_path)
    assert models.resolve_model("pose", cache_dir=tmp_path) == path
    assert len(calls) == 1
    path.write_bytes(b"corrupt")
    models.resolve_model("pose", cache_dir=tmp_path)
    assert len(calls) == 2


def test_explicit_path_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    path = tmp_path / "custom.task"
    with pytest.raises(FileNotFoundError):
        models.resolve_model("hand", path)
    path.write_bytes(b"user supplied")
    assert models.resolve_model("hand", path) == path


@pytest.mark.parametrize("failure", [False, True])
def test_failed_download_is_not_cached(monkeypatch, tmp_path, failure):
    def download(*args, **kwargs):
        if failure:
            raise OSError("offline")
        return BytesIO(b"not a model")

    monkeypatch.setattr(models, "urlopen", download)
    with pytest.raises(RuntimeError, match="--hand_model"):
        models.resolve_model("hand", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cache_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("XRT_DEVICES_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(models, "_valid_bundle", lambda path: True)
    assert models.resolve_model("hand") == tmp_path / "hand-float16-v1.task"
