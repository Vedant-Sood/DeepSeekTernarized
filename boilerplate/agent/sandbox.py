"""Workspace confinement and the subprocess runner.

This is a policy layer inside the agent's own process, not operating-system isolation: a command
the user allows can still do anything the user can. For untrusted repositories run the agent in a
container or set ``shell.wrapper`` to a bwrap/firejail prefix.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .security import is_secret_env_name


class SandboxError(Exception):
    """A request the sandbox refuses outright (never turned into a prompt)."""



# programs that only prefix another command; denylist and allowlist look through them
_WRAPPERS = {"env", "nice", "nohup", "command", "exec", "time", "timeout", "stdbuf", "ionice",
             "chrt", "taskset", "unbuffer", "xargs", "builtin"}
_FIND_EXEC = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"}


@dataclass
class Workspace:
    root: Path
    sensitive_globs: list[str]
    sensitive_dirs: list[str]

    def __post_init__(self):
        self.root = Path(os.path.realpath(self.root))

    # -------------------------------------------------------------- paths
    def resolve(self, path: str, *, for_write: bool = False) -> Path:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise SandboxError("invalid path")
        p = Path(os.path.expanduser(path)) if path.startswith("~") else Path(path)
        if not p.is_absolute():
            p = self.root / p
        if for_write and p.is_symlink():
            raise SandboxError(f"refusing to write through symlink {path}")
        real = Path(os.path.realpath(p))
        if real != self.root and self.root not in real.parents:
            raise SandboxError(f"path {path} is outside the workspace {self.root}")
        rel = self.rel(real)
        parts = PurePosixPath(rel).parts
        if for_write and parts and parts[0] == ".git":
            raise SandboxError("writing inside .git/ is not allowed; use the git tool")
        return real

    def rel(self, real: Path) -> str:
        return "." if real == self.root else real.relative_to(self.root).as_posix()

    def is_sensitive(self, real: Path) -> bool:
        rel = self.rel(real)
        name = real.name
        if any(fnmatch.fnmatch(name.lower(), g.lower()) for g in self.sensitive_globs):
            return True
        return any(rel == d or rel.startswith(d + "/") or f"/{d}/" in f"/{rel}/" for d in self.sensitive_dirs)

    # -------------------------------------------------------------- command analysis
    def command_paths_ok(self, argv: list[str]) -> bool:
        """True when every path-looking argument stays inside the workspace and is not sensitive.
        Used to decide whether an allowlist entry may apply without asking."""
        for a in argv[1:]:
            val = a.split("=", 1)[1] if a.startswith("-") and "=" in a else a
            if val.startswith("-") or not val:
                continue
            looks_like_path = "/" in val or val.startswith(("~", ".")) or (self.root / val).exists()
            if not looks_like_path:
                continue
            try:
                real = self.resolve(val)
            except SandboxError:
                return False
            if self.is_sensitive(real):
                return False
        return True


def has_shell_syntax(command: str) -> bool:
    """True when the command uses unquoted shell operators or substitutions. Commands without them
    run with no shell at all, so quoted text such as ``python -c "a; b"`` reaches the program as is."""
    if "\n" in command or "\r" in command:
        return True
    lex = shlex.shlex(command, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        tokens = list(lex)
    except ValueError:
        return True
    if any(t and set(t) <= set("();<>|&") for t in tokens):
        return True
    # unquoted $( or backtick: compare against a lexer pass that keeps quotes
    raw = shlex.shlex(command, posix=False)
    raw.whitespace_split = True
    raw.commenters = ""
    try:
        for t in raw:
            if t[:1] in ("'", '"'):
                continue
            if "`" in t or "$(" in t or "${" in t:
                return True
    except ValueError:
        return True
    return False


def split_command(command: str) -> list[str]:
    try:
        argv = shlex.split(command, comments=False, posix=True)
    except ValueError as e:
        raise SandboxError(f"cannot parse command: {e}") from None
    if not argv:
        raise SandboxError("empty command")
    return argv


def effective_commands(argv: list[str]) -> list[list[str]]:
    """The command and every command it wraps, with program names reduced to their basename:
    ``env FOO=1 /usr/bin/curl x`` yields ``[["env", ...], ["curl", "x"]]``."""
    out = []
    cur = list(argv)
    for _ in range(8):
        if not cur:
            break
        cur = [os.path.basename(cur[0])] + cur[1:]
        out.append(cur)
        prog = cur[0]
        if prog not in _WRAPPERS:
            break
        rest = cur[1:]
        # skip the wrapper's own options and assignments / numeric arguments
        while rest and (rest[0].startswith("-") or "=" in rest[0] or rest[0].replace(".", "").isdigit()):
            opt = rest.pop(0)
            if prog in ("nice", "ionice", "taskset", "chrt", "stdbuf") and opt in ("-n", "-c", "-p") and rest:
                rest.pop(0)
        cur = rest
    return out


def matches(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if fnmatch.fnmatchcase(text, p):
            return p
    return None


def is_denylisted(argv: list[str], denylist: list[str]) -> str | None:
    for cmd in effective_commands(argv):
        line = shlex.join(cmd)
        hit = matches(denylist, line) or matches(denylist, cmd[0]) or matches(denylist, cmd[0] + " *")
        if hit:
            return hit
    return None


def is_allowlisted(argv: list[str], allowlist: list[str]) -> str | None:
    cmds = effective_commands(argv)
    if len(cmds) > 1:  # wrapped commands never match the allowlist
        return None
    cmd = cmds[0]
    if cmd[0] == "find" and any(a in _FIND_EXEC for a in cmd):
        return None
    if cmd[0] in ("sh", "bash", "zsh", "dash", "fish") or "-c" in cmd[1:2]:
        return None
    return matches(allowlist, shlex.join(cmd)) or matches(allowlist, " ".join(cmd))


def scrubbed_env(passthrough: list[str]) -> dict[str, str]:
    # GIT_CONFIG_{COUNT,KEY_n,VALUE_n} and GIT_CONFIG_PARAMETERS inject git config (often auth headers);
    # dropping only some of them makes git fail, so drop them all
    env = {k: v for k, v in os.environ.items()
           if not is_secret_env_name(k) and not k.startswith("GIT_CONFIG")}
    for k in passthrough:
        if k in os.environ:
            env[k] = os.environ[k]
    env.update(GIT_TERMINAL_PROMPT="0", PAGER="cat", GIT_PAGER="cat", TERM="dumb", NO_COLOR="1",
               PYTHONDONTWRITEBYTECODE=env.get("PYTHONDONTWRITEBYTECODE", "1"))
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    env.pop("SSH_AUTH_SOCK", None)
    return env


@dataclass
class CommandResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

    def render(self) -> str:
        parts = []
        if self.timed_out:
            parts.append("[timed out; process group killed]")
        parts.append(f"exit code: {self.exit_code}")
        if self.stdout:
            parts.append("stdout:\n" + self.stdout.rstrip("\n"))
        if self.stderr:
            parts.append("stderr:\n" + self.stderr.rstrip("\n"))
        if self.truncated:
            parts.append("[output truncated]")
        return "\n".join(parts)


def _limits(cpu_seconds: int, max_file_bytes: int, max_open_files: int):
    def fn():
        try:
            import resource

            for lim, val in ((resource.RLIMIT_CPU, cpu_seconds), (resource.RLIMIT_FSIZE, max_file_bytes),
                             (resource.RLIMIT_NOFILE, max_open_files)):
                soft, hard = resource.getrlimit(lim)
                v = val if hard == resource.RLIM_INFINITY else min(val, hard)
                resource.setrlimit(lim, (v, hard))
        except (ImportError, ValueError, OSError):
            pass
    return fn


def run_command(argv: list[str], cwd: Path, *, timeout: float, max_output_bytes: int,
                env: dict[str, str], cpu_seconds: int = 900, max_file_bytes: int = 1 << 30,
                max_open_files: int = 1024) -> CommandResult:
    """Run ``argv`` without a shell in its own process group, stdin closed, output capped."""
    kw = {}
    if sys.platform != "win32":
        kw["start_new_session"] = True
        kw["preexec_fn"] = _limits(cpu_seconds, max_file_bytes, max_open_files)
    else:  # pragma: no cover
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    except FileNotFoundError:
        return CommandResult(127, "", f"command not found: {argv[0]}")
    except PermissionError as e:
        return CommandResult(126, "", f"permission denied: {e}")

    bufs = {"out": bytearray(), "err": bytearray()}
    over = {"flag": False}

    def pump(stream, key):
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            room = max_output_bytes - len(bufs[key])
            if room > 0:
                bufs[key].extend(chunk[:room])
            if len(chunk) > room:
                over["flag"] = True
        stream.close()

    threads = [threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    except KeyboardInterrupt:
        _kill_group(proc)
        raise
    if timed_out:
        _kill_group(proc)
    for t in threads:
        t.join(timeout=5)
    return CommandResult(proc.returncode, bufs["out"].decode("utf-8", "replace"),
                         bufs["err"].decode("utf-8", "replace"), timed_out, over["flag"])


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass
