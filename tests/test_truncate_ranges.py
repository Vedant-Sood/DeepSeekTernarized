import io

import pytest

from boilerplate.testing.truncate import RangeResponseError, _Reader


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers
        self.read_called = False

    def read(self, *args):
        self.read_called = True
        return super().read(*args)


def test_remote_range_must_match_requested_offsets(monkeypatch):
    response = Response(b"abcd", 206, {"Content-Range": "bytes 8-11/20", "Content-Length": "4"})
    seen = []

    def open_range(request, timeout):
        seen.append(request.get_header("Range"))
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_range)
    assert _Reader("owner/repo").read("shard", 8, 12) == b"abcd"
    assert seen == ["bytes=8-11"]


@pytest.mark.parametrize("status,headers", [
    (200, {"Content-Length": "1000000000"}),
    (206, {"Content-Range": "bytes 0-3/20", "Content-Length": "4"}),
    (206, {"Content-Range": "bytes 8-11/20", "Content-Length": "1000000000"}),
    (206, {"Content-Length": "4"}),
])
def test_remote_range_rejects_unexpected_response_before_read(monkeypatch, status, headers):
    response = Response(b"full file bytes", status, headers)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
    with pytest.raises(RangeResponseError):
        _Reader("owner/repo").read("shard", 8, 12)
    assert not response.read_called


def test_local_range_rejects_short_read(tmp_path):
    (tmp_path / "shard").write_bytes(b"abcd")
    reader = _Reader(str(tmp_path))
    assert reader.read("shard", 1, 3) == b"bc"
    with pytest.raises(IOError, match="short read"):
        reader.read("shard", 1, 8)
