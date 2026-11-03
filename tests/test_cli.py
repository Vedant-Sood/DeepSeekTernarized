import pytest

from boilerplate.cli import _rss_bytes


@pytest.mark.parametrize(
    ("platform", "ru_maxrss"),
    [("linux", 2_097_152), ("darwin", 2_147_483_648)],
)
def test_rss_bytes_matches_known_two_gib(platform, ru_maxrss):
    assert _rss_bytes(ru_maxrss, platform) == 2_147_483_648


def test_rss_bytes_rejects_unknown_units():
    with pytest.raises(ValueError, match="unsupported ru_maxrss units"):
        _rss_bytes(1, "unknown")
