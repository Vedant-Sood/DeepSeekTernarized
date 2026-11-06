"""Built-in tools: file reading and search, edits, shell and git.

Each tool turns validated arguments into a permission ``Request`` (without side effects), can render
a preview for the approval prompt, and runs. Paths always go through ``Workspace.resolve``.
"""

from __future__ import annotations

import difflib
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .permissions import Request, rule_matches
from .sandbox import (SandboxError, Workspace, has_shell_syntax, is_allowlisted, is_denylisted, run_command,
                      scrubbed_env, split_command)


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from."""


@dataclass
class ToolContext:
    cfg: AgentConfig
    ws: Workspace


# ------------------------------------------------------------------ argument validation
_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float), "array": list, "object": dict}


def validate_args(schema: dict, args: Any) -> dict:
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(sorted(unknown))}")
    for k in schema.get("required", []):
        if k not in args:
            raise ToolError(f"missing required argument {k!r}")
    for k, v in args.items():
        spec = props[k]
        t = _TYPES[spec["type"]]
        if isinstance(v, bool) and spec["type"] in ("integer", "number"):
            raise ToolError(f"{k} must be {spec['type']}")
        if not isinstance(v, t):
            raise ToolError(f"{k} must be {spec['type']}")
        if spec["type"] == "string" and len(v) > spec.get("maxLength", 4 << 20):
            raise ToolError(f"{k} is too long")
        if spec["type"] == "integer" and not spec.get("minimum", -2**63) <= v <= spec.get("maximum", 2**63):
            raise ToolError(f"{k} out of range")
    return args


class Tool:
    name = ""
    description = ""
    schema: dict = {}
    risk = "read"

    def spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.schema}}

    def request(self, args: dict, ctx: ToolContext) -> Request:
        raise NotImplementedError

    def preview(self, args: dict, ctx: ToolContext) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in args.items())

    def run(self, args: dict, ctx: ToolContext) -> str:
        raise NotImplementedError


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_PATH = {"type": "string", "maxLength": 4096, "description": "path relative to the workspace root"}


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


class _PathTool(Tool):
    path_arg = "path"
    for_write = False

    def request(self, args, ctx):
        try:
            real = ctx.ws.resolve(args.get(self.path_arg, "."), for_write=self.for_write)
        except SandboxError as e:
            return Request(self.name, self.risk, str(args.get(self.path_arg)), hard_deny=str(e))
        return Request(self.name, self.risk, ctx.ws.rel(real), sensitive=ctx.ws.is_sensitive(real))


class ReadFile(_PathTool):
    name = "read_file"
    description = ("Read a UTF-8 text file in the workspace. Returns numbered lines. Use offset/limit "
                   "(1-based line numbers) for large files.")
    schema = _obj({"path": _PATH, "offset": {"type": "integer", "minimum": 1},
                   "limit": {"type": "integer", "minimum": 1, "maximum": 20000}}, ["path"])

    def run(self, args, ctx):
        p = ctx.ws.resolve(args["path"])
        if not p.is_file():
            raise ToolError(f"{args['path']} is not a file")
        if p.stat().st_size > ctx.cfg.limits.max_read_bytes:
            if "offset" not in args and "limit" not in args:
                raise ToolError(f"file is {p.stat().st_size} bytes; pass offset and limit to read part of it")
        with open(p, "rb") as f:
            data = f.read(ctx.cfg.limits.max_read_bytes + 1)
        if _is_binary(data):
            raise ToolError("binary file; not shown")
        lines = data[: ctx.cfg.limits.max_read_bytes].decode("utf-8", "replace").splitlines()
        start = args.get("offset", 1)
        end = start - 1 + args.get("limit", 2000)
        chunk = lines[start - 1:end]
        out = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(chunk, start))
        if end < len(lines):
            out += f"\n[{len(lines) - end} more lines; continue with offset={end + 1}]"
        return out or "(empty file)"


class ListDir(_PathTool):
    name = "list_dir"
    description = "List one directory level in the workspace (directories end with /)."
    schema = _obj({"path": _PATH}, [])

    def run(self, args, ctx):
        p = ctx.ws.resolve(args.get("path", "."))
        if not p.is_dir():
            raise ToolError(f"{args.get('path', '.')} is not a directory")
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        cap = ctx.cfg.limits.max_glob_results
        out = [e.name + ("/" if e.is_dir() else "") for e in entries[:cap]]
        if len(entries) > cap:
            out.append(f"[{len(entries) - cap} more entries]")
        return "\n".join(out) or "(empty directory)"


_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _walk_files(ctx: ToolContext, base: Path):
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for fn in sorted(filenames):
            yield Path(dirpath) / fn


class Glob(_PathTool):
    name = "glob"
    description = "Find files by glob pattern, for example '**/*.py', relative to path (default: workspace root)."
    schema = _obj({"pattern": {"type": "string", "maxLength": 512}, "path": _PATH}, ["pattern"])

    def request(self, args, ctx):
        pat = args["pattern"]
        if pat.startswith(("/", "~")) or ".." in Path(pat).parts:
            return Request(self.name, self.risk, pat, hard_deny="glob patterns must be relative and stay in the workspace")
        return super().request(args, ctx)

    def run(self, args, ctx):
        base = ctx.ws.resolve(args.get("path", "."))
        cap = ctx.cfg.limits.max_glob_results
        out = []
        for m in base.glob(args["pattern"]):
            try:
                real = ctx.ws.resolve(str(m))
            except SandboxError:
                continue
            rel = ctx.ws.rel(real)
            if rel.split("/")[0] in _SKIP_DIRS:
                continue
            out.append(rel + ("/" if real.is_dir() else ""))
            if len(out) > cap:
                break
        more = len(out) > cap
        out = sorted(out[:cap])
        return "\n".join(out) + (f"\n[more than {cap} results; narrow the pattern]" if more else "") if out else "(no matches)"


class Grep(_PathTool):
    name = "grep"
    description = ("Search file contents with a Python regular expression. Returns path:line:text. "
                   "Optional file_glob filters file names (e.g. '*.py').")
    schema = _obj({"pattern": {"type": "string", "maxLength": 500}, "path": _PATH,
                   "file_glob": {"type": "string", "maxLength": 256},
                   "ignore_case": {"type": "boolean"}}, ["pattern"])

    def run(self, args, ctx):
        import fnmatch

        try:
            rx = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as e:
            raise ToolError(f"bad regex: {e}") from None
        base = ctx.ws.resolve(args.get("path", "."))
        files = [base] if base.is_file() else _walk_files(ctx, base)
        cap, out, skipped = ctx.cfg.limits.max_grep_matches, [], 0
        for f in files:
            if args.get("file_glob") and not fnmatch.fnmatch(f.name, args["file_glob"]):
                continue
            try:
                real = ctx.ws.resolve(str(f))
                if ctx.ws.is_sensitive(real):
                    skipped += 1
                    continue
                if real.stat().st_size > ctx.cfg.limits.max_read_bytes:
                    continue
                data = real.read_bytes()
            except (SandboxError, OSError):
                continue
            if _is_binary(data):
                continue
            for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{ctx.ws.rel(real)}:{i}:{line[:300]}")
                    if len(out) >= cap:
                        out.append(f"[stopped at {cap} matches]")
                        return "\n".join(out)
        if skipped:
            out.append(f"[skipped {skipped} credential-like file(s); read them explicitly if needed]")
        return "\n".join(out) or "(no matches)"


def _diff(path: str, old: str, new: str, limit: int = 12000) -> str:
    d = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))
    return d if len(d) <= limit else d[:limit] + "\n[diff truncated]"


def _atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = p.stat().st_mode & 0o7777 if p.exists() else None
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_text(p: Path, ctx: ToolContext) -> str:
    if p.stat().st_size > ctx.cfg.limits.max_write_bytes:
        raise ToolError("file too large to edit")
    data = p.read_bytes()
    if _is_binary(data):
        raise ToolError("binary file; cannot edit")
    return data.decode("utf-8")


class WriteFile(_PathTool):
    name = "write_file"
    description = "Create or overwrite a text file in the workspace with the given content."
    schema = _obj({"path": _PATH, "content": {"type": "string", "maxLength": 8 << 20}}, ["path", "content"])
    risk = "write"
    for_write = True

    def request(self, args, ctx):
        r = super().request(args, ctx)
        if not r.hard_deny and len(args["content"].encode()) > ctx.cfg.limits.max_write_bytes:
            r.hard_deny = f"content exceeds max_write_bytes ({ctx.cfg.limits.max_write_bytes})"
        return r

    def preview(self, args, ctx):
        p = ctx.ws.resolve(args["path"], for_write=True)
        old = _read_text(p, ctx) if p.is_file() else ""
        return _diff(ctx.ws.rel(p), old, args["content"]) or "(no change)"

    def run(self, args, ctx):
        p = ctx.ws.resolve(args["path"], for_write=True)
        if p.is_dir():
            raise ToolError("path is a directory")
        existed = p.exists()
        _atomic_write(p, args["content"])
        return f"{'overwrote' if existed else 'created'} {ctx.ws.rel(p)} ({len(args['content'])} chars)"


class EditFile(_PathTool):
    name = "edit_file"
    description = ("Replace old_string with new_string in a file. old_string must match exactly (including "
                   "whitespace) and occur exactly once unless replace_all is true. Read the file first.")
    schema = _obj({"path": _PATH, "old_string": {"type": "string", "maxLength": 1 << 20},
                   "new_string": {"type": "string", "maxLength": 1 << 20},
                   "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"])
    risk = "write"
    for_write = True

    def _apply(self, args, ctx):
        p = ctx.ws.resolve(args["path"], for_write=True)
        if not p.is_file():
            raise ToolError(f"{args['path']} does not exist; use write_file to create it")
        old = _read_text(p, ctx)
        if not args["old_string"]:
            raise ToolError("old_string must not be empty")
        if args["old_string"] == args["new_string"]:
            raise ToolError("old_string and new_string are identical")
        n = old.count(args["old_string"])
        if n == 0:
            raise ToolError("old_string not found; read the file and copy the text exactly")
        if n > 1 and not args.get("replace_all"):
            raise ToolError(f"old_string occurs {n} times; add surrounding context or set replace_all")
        new = old.replace(args["old_string"], args["new_string"]) if args.get("replace_all") else \
            old.replace(args["old_string"], args["new_string"], 1)
        if len(new.encode()) > ctx.cfg.limits.max_write_bytes:
            raise ToolError("result exceeds max_write_bytes")
        return p, old, new, n

    def preview(self, args, ctx):
        p, old, new, _ = self._apply(args, ctx)
        return _diff(ctx.ws.rel(p), old, new)

    def run(self, args, ctx):
        p, old, new, n = self._apply(args, ctx)
        _atomic_write(p, new)
        return f"edited {ctx.ws.rel(p)} ({n if args.get('replace_all') else 1} replacement(s))"


_CODE_RUNNERS = {"python", "python3", "pytest", "make", "npm", "npx", "yarn", "pnpm", "node", "cargo", "go",
                 "tox", "nox", "uv", "poetry", "bundle", "rake", "gradle", "mvn", "ruff", "mypy"}


class Shell(Tool):
    name = "shell"
    description = ("Run a command in the workspace root without a shell (no pipes, redirects, ';', '&&', "
                   "'$()' or globs unless the user enabled shell syntax). Returns exit code, stdout and "
                   "stderr. stdin is closed; interactive programs will not work.")
    schema = _obj({"command": {"type": "string", "maxLength": 8000},
                   "timeout": {"type": "integer", "minimum": 1, "maximum": 3600,
                               "description": "seconds"}}, ["command"])
    risk = "execute"

    def request(self, args, ctx):
        cmd = args["command"].strip()
        sh = ctx.cfg.shell
        try:
            argv = split_command(cmd)
        except SandboxError as e:
            return Request(self.name, self.risk, cmd, hard_deny=str(e))
        syntax = has_shell_syntax(cmd)
        if syntax and not sh.allow_shell_syntax:
            return Request(self.name, self.risk, cmd, hard_deny=(
                "shell syntax (pipes, redirects, ;, &&, $(), backticks) is disabled; run one program "
                "with arguments, or use the read/grep/glob tools"))
        hit = is_denylisted(argv, sh.denylist)
        if hit:
            return Request(self.name, self.risk, cmd, hard_deny=f"command matches the shell denylist ({hit})")
        allow = None if syntax else is_allowlisted(argv, sh.allowlist)
        if allow and not ctx.ws.command_paths_ok(argv):
            allow = None
        prog = os.path.basename(argv[0])
        return Request(self.name, self.risk, cmd, allowlisted=allow,
                       runs_workspace_code=prog in _CODE_RUNNERS or prog.startswith("python"))

    def preview(self, args, ctx):
        return f"$ {args['command']}"

    def run(self, args, ctx):
        sh = ctx.cfg.shell
        cmd = args["command"].strip()
        argv = ["/bin/sh", "-c", cmd] if has_shell_syntax(cmd) else split_command(cmd)
        timeout = min(float(args.get("timeout", sh.timeout)), sh.max_timeout)
        res = run_command(list(sh.wrapper) + argv, ctx.ws.root, timeout=timeout,
                          max_output_bytes=ctx.cfg.limits.max_output_chars * 2,
                          env=scrubbed_env(sh.env_passthrough), cpu_seconds=sh.cpu_seconds,
                          max_file_bytes=sh.max_file_bytes, max_open_files=sh.max_open_files)
        return res.render()


# git: classification of subcommands. Anything not listed is refused unless an allow rule names it.
_GIT_READ = {"status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "shortlog", "describe", "grep",
             "cat-file", "ls-tree", "merge-base", "reflog"}
_GIT_WRITE = {"add", "commit", "checkout", "switch", "restore", "stash", "merge", "cherry-pick", "revert",
              "mv", "rm", "reset", "branch", "tag", "init", "apply", "rebase"}
_GIT_BAD_OPTS = ("--output", "--ext-diff", "--exec", "--upload-pack", "--receive-pack", "--exec-path",
                 "--git-dir", "--work-tree", "--interactive", "--edit", "--patch", "--config-env", "--no-index",
                 "--open-files-in-pager", "--unsafe-paths", "--directory", "--template", "--separate-git-dir",
                 "--textconv", "--upload-archive", "--remote")
_GIT_BAD_SHORT = {"-x", "-i", "-e", "-c", "-C", "-p", "-O"}
# short options that are harmless for particular subcommands (context lines, combined diffs, patterns)
_GIT_SHORT_OK = {"grep": {"-e", "-C", "-i", "-c", "-p"}, "log": {"-p", "-c", "-i"}, "show": {"-p", "-c"},
                 "diff": {"-p", "-c"}, "blame": {"-C", "-e", "-p", "-c"}, "cherry-pick": {"-x"},
                 "tag": {"-l"}, "branch": {"-c", "-C"}}


def _bad_git_option(sub: str, arg: str) -> bool:
    if arg == "--":
        return False
    if arg.startswith("--"):
        name = arg.split("=", 1)[0]
        # git accepts unambiguous abbreviations of long options, so match prefixes as well
        return any(b == name or (len(name) >= 5 and b.startswith(name)) for b in _GIT_BAD_OPTS)
    if arg.startswith("-") and len(arg) >= 2:
        short = arg[:2]
        if short == "-O":
            return True
        return short in _GIT_BAD_SHORT and short not in _GIT_SHORT_OK.get(sub, set())
    return False


_GIT_HARDENING = ["-c", "core.fsmonitor=false", "-c", "core.pager=cat", "-c", "diff.external=",
                  "-c", "core.sshCommand=false", "-c", "credential.helper=", "-c", "protocol.allow=never",
                  "-c", "core.editor=false", "-c", "sequence.editor=false"]
# local repository config keys that make otherwise read-only git commands execute programs
_GIT_DANGEROUS_CONFIG = re.compile(
    r"^(core\.(fsmonitor|pager|sshcommand|editor|hookspath|gitproxy|askpass)|diff\..*\.(command|textconv)|"
    r"filter\..*\.(clean|smudge|process)|merge\..*\.driver|alias\..*|credential\..*|.*\.helper|"
    r"sequence\.editor|gpg\.program|gpg\..*\.program|uploadpack\..*|receivepack\..*)=", re.I)


class Git(Tool):
    name = "git"
    description = ("Run git with the given arguments, e.g. 'status', 'diff --stat', 'log -n 5 --oneline', "
                   "'add path', 'commit -m \"message\"'. Network and config subcommands are refused.")
    schema = _obj({"args": {"type": "string", "maxLength": 4000}}, ["args"])

    def _argv(self, args) -> list[str]:
        argv = split_command(args["args"]) if args["args"].strip() else []
        if argv and argv[0] == "git":
            argv = argv[1:]
        if not argv:
            raise SandboxError("missing git subcommand")
        return argv

    def _classify(self, argv: list[str]) -> tuple[str, str | None]:
        sub = argv[0]
        if sub.startswith("-"):
            return "execute", "git global options are not allowed; put the subcommand first"
        for a in argv[1:]:
            if a == "--":
                break
            if _bad_git_option(sub, a):
                return "execute", f"git option {a} is not allowed here (it can run programs, touch files "\
                    "outside the workspace, or needs an interactive terminal)"
        if sub == "reset" and "--hard" in argv:
            return "execute", "dangerous"
        if sub == "branch" and all(a in ("-a", "-r", "-v", "-vv", "--list", "--all", "--show-current") for a in argv[1:]):
            return "read", None
        if sub == "tag" and all(a in ("-l", "--list") for a in argv[1:]):
            return "read", None
        if sub == "stash" and argv[1:2] in (["list"], ["show"]):
            return "read", None
        if sub in _GIT_READ:
            return "read", None
        if sub in _GIT_WRITE:
            return "execute", None
        return "execute", "dangerous"

    def _risky_config(self, ctx) -> list[str]:
        cfgfile = ctx.ws.root / ".git" / "config"
        if not cfgfile.is_file():
            return []
        try:
            out = subprocess.run(["git", "config", "--file", str(cfgfile), "--list"], capture_output=True,
                                 text=True, timeout=10, env=scrubbed_env([])).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ["(could not read .git/config)"]
        return [line.split("=", 1)[0] for line in out.splitlines() if _GIT_DANGEROUS_CONFIG.match(line)]

    def request(self, args, ctx):
        self._risky_note = ""
        try:
            argv = self._argv(args)
        except SandboxError as e:
            return Request(self.name, "execute", args["args"], hard_deny=str(e))
        subject = shlex.join(argv)
        risk, problem = self._classify(argv)
        if problem == "dangerous":
            if not any(rule_matches(r, self.name, subject) for r in ctx.cfg.allow):
                return Request(self.name, risk, subject, hard_deny=(
                    f"git {argv[0]} is not allowed by default (network, config, history-destroying or "
                    f"unknown subcommand); the user can allow it with a rule like git({argv[0]} *)"))
        elif problem:
            return Request(self.name, risk, subject, hard_deny=problem)
        risky = self._risky_config(ctx)
        req = Request(self.name, risk, subject)
        if risky:
            # treated like a sensitive path: always ask, since these keys can run programs
            req.sensitive = True
            req.risk = "execute"
            self._risky_note = f"the repository's .git/config sets {', '.join(risky)}, which can run programs"
        return req

    def preview(self, args, ctx):
        note = getattr(self, "_risky_note", "")
        return f"$ git {args['args']}" + (f"\n[warning: {note}]" if note else "")

    def run(self, args, ctx):
        argv = self._argv(args)
        extra = ["--no-ext-diff", "--no-textconv"] if argv[0] in ("diff", "log", "show") else []
        full = ["git", *_GIT_HARDENING, argv[0], *extra, *argv[1:]]
        res = run_command(full, ctx.ws.root, timeout=ctx.cfg.shell.timeout,
                          max_output_bytes=ctx.cfg.limits.max_output_chars * 2,
                          env=scrubbed_env(ctx.cfg.shell.env_passthrough))
        return res.render()


def default_tools() -> dict[str, Tool]:
    tools = [ReadFile(), ListDir(), Glob(), Grep(), WriteFile(), EditFile(), Shell(), Git()]
    return {t.name: t for t in tools}
