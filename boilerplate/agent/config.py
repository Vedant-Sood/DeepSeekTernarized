"""Layered agent configuration.

Order: built-in defaults, then the user config (``~/.config/boilerplate/agent.yaml`` or
``--config``), then the project config (``<workspace>/.boilerplate/agent.yaml``), then command-line
flags. The project file comes from the repository being edited, which may be hostile, so unless the
workspace is listed in the user config's ``trusted_workspaces`` it may only tighten: add deny and
ask rules, add shell denylist entries and lower numeric limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODES = ("plan", "default", "accept-edits", "auto")

DEFAULT_ALLOWLIST = [
    "ls *", "ls", "cat *", "head *", "tail *", "wc *", "pwd", "echo *", "true", "false",
    "find *", "tree *", "tree", "diff *", "file *", "stat *", "which *", "sort *", "uniq *",
    "python -m pytest*", "python3 -m pytest*", "pytest*", "python -m ruff *", "ruff *",
    "python -m mypy *", "mypy *", "python -m compileall *", "python -m pip list*", "pip list*",
    "make test*", "make check*", "make lint*", "npm test*", "npm run test*", "npm run lint*",
    "cargo test*", "cargo check*", "cargo build*", "go test *", "go build *", "go vet *",
]
DEFAULT_DENYLIST = [
    "sudo *", "su", "su *", "doas *", "rm -rf /*", "rm -rf ~*", "rm -fr /*", "rm -rf .", "rm -rf *",
    "mkfs*", "dd *", "shutdown*", "reboot*", "halt*", "poweroff*", "chmod -R 777 *", "chown -R *",
    "curl *", "wget *", "ssh *", "scp *", "sftp *", "rsync *", "nc *", "ncat *", "netcat *",
    "socat *", "telnet *", "ftp *", "crontab *", "systemctl *", "launchctl *", "kill -9 -1",
    "pip install *", "pip3 install *", "python -m pip install *", "npm install *", "npm i *",
    "git push*", "git config*", "git remote*",
]
SENSITIVE_GLOBS = [
    ".env", ".env.*", "*.env", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", ".netrc", ".pgpass", ".npmrc", ".pypirc",
    "*credentials*", "*secret*", ".git-credentials", "*.kdbx",
]
SENSITIVE_DIRS = [".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config/gcloud", ".azure"]


@dataclass
class BackendConfig:
    kind: str = "openai"  # openai | local | mock
    endpoint: str = "http://127.0.0.1:8000/v1"
    model: str = "boilerplate"
    api_key_env: str | None = None
    tool_protocol: str = "native"  # native | text
    allow_insecure_http: bool = False
    timeout: float = 300.0
    temperature: float = 0.2
    max_tokens: int = 4096
    checkpoint: str | None = None
    max_seq_len: int = 8192
    max_response_bytes: int = 32 << 20


@dataclass
class ShellConfig:
    allowlist: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWLIST))
    denylist: list[str] = field(default_factory=lambda: list(DEFAULT_DENYLIST))
    allow_shell_syntax: bool = False
    timeout: float = 120.0
    max_timeout: float = 600.0
    env_passthrough: list[str] = field(default_factory=list)
    wrapper: list[str] = field(default_factory=list)  # e.g. a bwrap/firejail prefix
    cpu_seconds: int = 900
    max_file_bytes: int = 1 << 30
    max_open_files: int = 1024


@dataclass
class Limits:
    max_steps: int = 50
    max_output_chars: int = 30_000
    max_read_bytes: int = 2 << 20
    max_write_bytes: int = 1 << 20
    max_glob_results: int = 500
    max_grep_matches: int = 300


@dataclass
class ContextConfig:
    window: int = 32_768
    compact_at: float = 0.75
    keep_recent: int = 12


@dataclass
class AgentConfig:
    workspace: Path = field(default_factory=Path.cwd)
    mode: str = "default"
    backend: BackendConfig = field(default_factory=BackendConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)
    limits: Limits = field(default_factory=Limits)
    context: ContextConfig = field(default_factory=ContextConfig)
    allow: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    sensitive_globs: list[str] = field(default_factory=lambda: list(SENSITIVE_GLOBS))
    sensitive_dirs: list[str] = field(default_factory=lambda: list(SENSITIVE_DIRS))
    trusted_workspaces: list[str] = field(default_factory=list)
    sessions_dir: Path | None = None
    system_prompt_extra: str = ""
    project_config_trusted: bool = False
    notes: list[str] = field(default_factory=list)  # warnings produced while loading

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.backend.kind not in ("openai", "local", "mock"):
            raise ValueError(f"unknown backend kind {self.backend.kind!r}")
        if self.backend.tool_protocol not in ("native", "text"):
            raise ValueError("tool_protocol must be 'native' or 'text'")
        if not 0 < self.context.compact_at <= 1:
            raise ValueError("context.compact_at must be in (0, 1]")
        if self.shell.timeout <= 0 or self.shell.timeout > self.shell.max_timeout:
            raise ValueError("shell.timeout must be positive and at most shell.max_timeout")


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "boilerplate" / "agent.yaml"


def default_sessions_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "boilerplate" / "agent" / "sessions"


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    if path.stat().st_size > 1 << 20:
        raise ValueError(f"{path}: config larger than 1 MiB")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _set_fields(obj, data: dict, where: str) -> None:
    for k, v in data.items():
        key = k.replace("-", "_")
        if not hasattr(obj, key):
            raise ValueError(f"{where}: unknown key {k!r}")
        cur = getattr(obj, key)
        if isinstance(cur, Path) or key == "sessions_dir":
            v = Path(os.path.expanduser(str(v)))
        setattr(obj, key, v)


def _apply_user(cfg: AgentConfig, data: dict, where: str) -> None:
    for section, obj in (("backend", cfg.backend), ("shell", cfg.shell), ("limits", cfg.limits), ("context", cfg.context)):
        if section in data:
            _set_fields(obj, data.pop(section) or {}, f"{where}:{section}")
    perms = data.pop("permissions", None) or {}
    for k in ("allow", "ask", "deny"):
        getattr(cfg, k).extend(perms.pop(k, []) or [])
    if perms:
        raise ValueError(f"{where}:permissions: unknown keys {sorted(perms)}")
    data.pop("workspace", None)  # the workspace is chosen on the command line, not by a file
    _set_fields(cfg, data, where)


def _apply_untrusted_project(cfg: AgentConfig, data: dict, where: str) -> None:
    """Only the tightening subset of the project file."""
    perms = data.get("permissions") or {}
    cfg.deny.extend(perms.get("deny", []) or [])
    cfg.ask.extend(perms.get("ask", []) or [])
    shell = data.get("shell") or {}
    cfg.shell.denylist.extend(shell.get("denylist", []) or [])
    if "timeout" in shell:
        cfg.shell.timeout = min(cfg.shell.timeout, float(shell["timeout"]))
    for k, v in (data.get("limits") or {}).items():
        if hasattr(cfg.limits, k) and isinstance(v, int):
            setattr(cfg.limits, k, min(getattr(cfg.limits, k), v))
    if data.get("system_prompt_extra"):
        cfg.system_prompt_extra = str(data["system_prompt_extra"])[:4000]
    ignored = sorted(set(data) - {"permissions", "shell", "limits", "system_prompt_extra"})
    ignored += [f"permissions.{k}" for k in perms if k not in ("deny", "ask")]
    ignored += [f"shell.{k}" for k in shell if k not in ("denylist", "timeout")]
    if ignored:
        cfg.notes.append(f"{where}: workspace is not trusted, ignored {', '.join(ignored)} "
                         "(add it to trusted_workspaces in your user config to honour them)")


def load_config(workspace: str | Path, config_path: str | Path | None = None) -> AgentConfig:
    ws = Path(workspace).expanduser().resolve()
    if not ws.is_dir():
        raise ValueError(f"workspace {ws} is not a directory")
    cfg = AgentConfig(workspace=ws)
    upath = Path(config_path).expanduser() if config_path else user_config_path()
    if config_path and not upath.is_file():
        raise ValueError(f"config file {upath} not found")
    _apply_user(cfg, _read_yaml(upath), str(upath))
    trusted = {str(Path(p).expanduser().resolve()) for p in cfg.trusted_workspaces}
    cfg.project_config_trusted = str(ws) in trusted
    ppath = ws / ".boilerplate" / "agent.yaml"
    if ppath.is_file() and not ppath.is_symlink():
        pdata = _read_yaml(ppath)
        if cfg.project_config_trusted:
            _apply_user(cfg, pdata, str(ppath))
        else:
            _apply_untrusted_project(cfg, pdata, str(ppath))
    if cfg.sessions_dir is None:
        cfg.sessions_dir = default_sessions_dir()
    cfg.validate()
    return cfg
