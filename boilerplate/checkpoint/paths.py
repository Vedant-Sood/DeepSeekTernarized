"""Constrain checkpoint-controlled filenames to the checkpoint directory."""

from pathlib import Path, PurePosixPath, PureWindowsPath


def checkpoint_filename(name: str) -> str:
    """Validate an index filename using the same rules on Windows and POSIX."""
    if (not isinstance(name, str) or not name or "\x00" in name or "\\" in name or
            PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive or
            any(part in (".", "..", "") for part in name.split("/"))):
        raise ValueError(f"unsafe checkpoint path: {name!r}")
    return name


def checkpoint_path(root: str | Path, name: str) -> Path:
    """Reject traversal, Windows paths on POSIX, and symlinks outside the root."""
    checkpoint_filename(name)
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"checkpoint path escapes its directory: {name!r}")
    return path
