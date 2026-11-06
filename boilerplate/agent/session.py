"""Session transcripts (JSONL, owner-only, redacted) and context compaction."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .security import Redactor

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _secure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


@dataclass
class Session:
    """Messages in OpenAI chat format. ``messages[0]`` is the system prompt."""

    directory: Path
    redact: Redactor
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    messages: list[dict] = field(default_factory=list)
    persist: bool = True

    @property
    def path(self) -> Path:
        return self.directory / f"{self.id}.jsonl"

    def _write(self, record: dict) -> None:
        if not self.persist:
            return
        _secure_dir(self.directory)
        line = self.redact(json.dumps(record, ensure_ascii=False)) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line)

    def append(self, message: dict) -> None:
        self.messages.append(message)
        self._write({"t": time.time(), "type": "message", "message": message})

    def note(self, kind: str, **data) -> None:
        self._write({"t": time.time(), "type": kind, **data})

    def replace_history(self, messages: list[dict], reason: str) -> None:
        self.messages = messages
        self._write({"t": time.time(), "type": "reset", "reason": reason, "messages": messages})

    @classmethod
    def load(cls, directory: Path, session_id: str, redact: Redactor) -> "Session":
        if not _ID.match(session_id):
            raise ValueError("bad session id")
        s = cls(directory, redact, id=session_id)
        if not s.path.is_file():
            raise FileNotFoundError(f"no session {session_id} in {directory}")
        for line in s.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "message":
                s.messages.append(rec["message"])
            elif rec.get("type") == "reset":
                s.messages = rec["messages"]
        return s

    @staticmethod
    def list(directory: Path, limit: int = 20) -> list[tuple[str, float, str]]:
        if not directory.is_dir():
            return []
        out = []
        for p in sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            first = ""
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        m = json.loads(line).get("message", {})
                    except json.JSONDecodeError:
                        continue
                    if m.get("role") == "user":
                        first = (m.get("content") or "")[:80].replace("\n", " ")
                        break
            out.append((p.stem, p.stat().st_mtime, first))
        return out


def message_tokens(m: dict, count: Callable[[str], int]) -> int:
    n = count(m.get("content") or "") + 4
    for c in m.get("tool_calls") or []:
        n += count(c["function"]["name"]) + count(c["function"].get("arguments") or "")
    return n


def _split_point(messages: list[dict], keep_recent: int) -> int:
    """Index where the kept tail starts: never between an assistant tool call and its results."""
    i = max(1, len(messages) - keep_recent)
    while i < len(messages) and messages[i]["role"] == "tool":
        i -= 1  # move back to include the assistant message that made the call
        if i <= 1:
            return 1
    return i


def digest(messages: list[dict], limit: int = 6000) -> str:
    """Mechanical summary used when the model cannot summarise."""
    lines = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            lines.append(f"- tool result ({len(m.get('content') or '')} chars)")
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(c["function"]["name"] for c in m["tool_calls"])
            lines.append(f"- assistant called {names}" + (f": {m['content'][:200]}" if m.get("content") else ""))
        else:
            lines.append(f"- {role}: {(m.get('content') or '')[:300]}")
    text = "\n".join(lines)
    return text[-limit:]


SUMMARY_PROMPT = ("Summarise the conversation so far for your own later use: the user's goal, decisions made, "
                  "files read or changed (with paths), commands run and their outcomes, and what remains to do. "
                  "Be concise and factual. Do not call tools.")


def compact(messages: list[dict], keep_recent: int, summarise: Callable[[list[dict]], str] | None) -> list[dict]:
    """System prompt + one summary message + the recent tail."""
    if len(messages) <= keep_recent + 2:
        return messages
    cut = _split_point(messages, keep_recent)
    old, tail = messages[1:cut], messages[cut:]
    if not old:
        return messages
    summary = None
    if summarise is not None:
        try:
            summary = summarise([messages[0], *old, {"role": "user", "content": SUMMARY_PROMPT}]).strip()
        except Exception:
            summary = None
    if not summary:
        summary = digest(old)
    note = {"role": "user", "content": "[Summary of earlier conversation, written to save context]\n" + summary}
    if tail and tail[0]["role"] == "user":
        tail = [{"role": "user", "content": note["content"] + "\n\n[Latest message]\n" + (tail[0].get("content") or "")},
                *tail[1:]]
        return [messages[0], *tail]
    return [messages[0], note, *tail]
