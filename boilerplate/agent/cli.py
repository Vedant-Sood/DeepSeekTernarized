"""``boilerplate agent``: interactive REPL and one-shot mode."""

from __future__ import annotations

import argparse
import sys
import time

from .backends import Backend, BackendError, make_backend
from .config import MODES, AgentConfig, load_config
from .loop import UI, Agent
from .permissions import Decision, PermissionEngine, parse_rule
from .security import Redactor
from .session import Session

HELP = """Commands:
  /help                 this help
  /exit                 quit (also Ctrl-D)
  /clear                start a fresh conversation (new session)
  /compact              summarise older messages now
  /mode [MODE]          show or set the permission mode (plan, default, accept-edits, auto)
  /permissions          show rules and mode
  /allow RULE           allow e.g. shell(pytest *) for this session
  /deny RULE            deny e.g. write_file(*.lock) for this session
  /tools                list tools
  /sessions             list saved sessions
  /resume ID            load a saved session
  /cost                 token usage this session
Ctrl-C interrupts the current step without leaving."""


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


class TerminalUI(UI):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._mid_line = False

    def text(self, chunk):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        self._mid_line = not chunk.endswith("\n")

    def _nl(self):
        if self._mid_line:
            sys.stdout.write("\n")
            self._mid_line = False

    def tool_start(self, name, summary):
        self._nl()
        print(_c("36", f"• {name} {summary[:200]}"))

    def tool_end(self, name, status, output):
        if status != "ok":
            print(_c("33", f"  {status}: {output[:300]}" if output else f"  {status}"))
        elif self.verbose and output:
            print(_c("2", "  " + output[:1500].replace("\n", "\n  ")))

    def notice(self, msg):
        self._nl()
        print(_c("33", f"! {msg}"), file=sys.stderr)


class TerminalPrompter:
    def approve(self, tool: str, preview: str, decision: Decision, rule: str):
        print()
        print(_c("1;35", f"Approve {tool}?") + f"  ({decision.reason})")
        print(preview if len(preview) < 6000 else preview[:6000] + "\n[preview truncated]")
        while True:
            try:
                ans = input(_c("35", "[y]es once / [a]lways this session / [n]o (optionally: n <reason>) > ")).strip()
            except EOFError:
                return "no", None
            if ans.lower() in ("y", "yes"):
                return "yes", None
            if ans.lower() in ("a", "always"):
                print(_c("2", f"  allowed for this session: {rule}"))
                return "always", None
            if ans.lower().startswith("n"):
                reason = ans[1:].strip() if not ans.lower().startswith("no") else ans[2:].strip()
                return "no", reason or None


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="boilerplate agent")
    p.add_argument("--workspace", default=".", help="repository the agent may read and change (default: cwd)")
    p.add_argument("--config", help="user config file (default: ~/.config/boilerplate/agent.yaml)")
    g = p.add_argument_group("model")
    g.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    g.add_argument("--model", help="model name sent to the endpoint")
    g.add_argument("--api-key-env", help="environment variable holding the API key")
    g.add_argument("--checkpoint", help="run a Boilerplate checkpoint in-process instead of an endpoint")
    g.add_argument("--tool-protocol", choices=["auto", "native", "text"],
                   help="native tool calls or <tool_call> text (auto: text for boilerplate serve)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--max-tokens", type=int)
    s = p.add_argument_group("permissions")
    s.add_argument("--mode", choices=MODES)
    s.add_argument("--allow", action="append", default=[], metavar="RULE", help="e.g. 'shell(pytest *)'")
    s.add_argument("--deny", action="append", default=[], metavar="RULE")
    s.add_argument("--yes", action="store_true", help="one-shot mode: approve what would otherwise be asked "
                   "(never overrides deny rules, sensitive paths, taint or hard limits)")
    r = p.add_argument_group("run")
    r.add_argument("-p", "--prompt", help="run one task non-interactively and exit")
    r.add_argument("--resume", metavar="ID", help="continue a saved session")
    r.add_argument("--max-steps", type=int)
    r.add_argument("--no-save", action="store_true", help="do not write a transcript")
    r.add_argument("--verbose", action="store_true", help="show tool output")
    return p


def config_from_args(a) -> AgentConfig:
    cfg = load_config(a.workspace, a.config)
    b = cfg.backend
    if a.checkpoint:
        b.kind, b.checkpoint, b.tool_protocol = "local", a.checkpoint, "text"
    if a.endpoint:
        b.kind, b.endpoint = "openai", a.endpoint
    for attr in ("model", "api_key_env", "tool_protocol", "temperature", "max_tokens"):
        v = getattr(a, attr)
        if v is not None:
            setattr(b, attr, v)
    if a.mode:
        cfg.mode = a.mode
    for rule in a.allow + a.deny:
        parse_rule(rule)
    cfg.allow.extend(a.allow)
    cfg.deny.extend(a.deny)
    if a.max_steps:
        cfg.limits.max_steps = a.max_steps
    cfg.validate()
    return cfg


class Repl:
    def __init__(self, cfg: AgentConfig, backend: Backend, *, interactive: bool, yes: bool, save: bool,
                 verbose: bool, resume: str | None = None):
        self.cfg, self.backend, self.save = cfg, backend, save
        self.redactor = Redactor()
        if cfg.backend.api_key_env:
            import os

            self.redactor.add_value(os.environ.get(cfg.backend.api_key_env))
        self.ui = TerminalUI(verbose)
        self.perms = PermissionEngine(cfg, interactive=interactive, yes=yes)
        session = Session.load(cfg.sessions_dir, resume, self.redactor) if resume else self._new_session()
        self.agent = self._agent(session)

    def _new_session(self) -> Session:
        return Session(self.cfg.sessions_dir, self.redactor, persist=self.save)

    def _agent(self, session: Session) -> Agent:
        return Agent(self.cfg, self.backend, session, perms=self.perms, prompter=TerminalPrompter(),
                     ui=self.ui, redactor=self.redactor)

    def command(self, line: str) -> bool:
        """Handle a slash command; return False to exit."""
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("exit", "quit"):
            return False
        if cmd == "help":
            print(HELP)
        elif cmd == "clear":
            self.agent = self._agent(self._new_session())
            print("new session", self.agent.session.id)
        elif cmd == "compact":
            print("compacted" if self.agent.maybe_compact(force=True) else "nothing to compact")
        elif cmd == "mode":
            if arg:
                try:
                    self.perms.set_mode(arg)
                except ValueError as e:
                    print(e)
            print("mode:", self.perms.mode)
        elif cmd == "permissions":
            print(self.perms.describe())
        elif cmd in ("allow", "deny"):
            try:
                parse_rule(arg)
            except ValueError as e:
                print(e)
                return True
            (self.perms.grant_session(arg) if cmd == "allow" else self.cfg.deny.append(arg))
            print(f"{cmd}: {arg}")
        elif cmd == "tools":
            for t in self.agent.tools.values():
                print(f"  {t.name:11s} [{t.risk}] {t.description.splitlines()[0][:90]}")
        elif cmd == "sessions":
            for sid, mtime, first in Session.list(self.cfg.sessions_dir):
                print(f"  {sid}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}  {first}")
        elif cmd == "resume":
            try:
                self.agent = self._agent(Session.load(self.cfg.sessions_dir, arg, self.redactor))
                print(f"resumed {arg} ({len(self.agent.session.messages)} messages)")
            except (ValueError, FileNotFoundError) as e:
                print(e)
        elif cmd == "cost":
            u = self.agent.usage
            print(f"prompt tokens {u['prompt_tokens']}, completion tokens {u['completion_tokens']}, "
                  f"context now ~{self.agent.context_tokens()} tokens")
        else:
            print(f"unknown command /{cmd}; /help lists commands")
        return True

    def loop(self) -> int:
        try:
            import readline  # noqa: F401  (line editing and history)
        except ImportError:
            pass
        print(f"boilerplate agent · workspace {self.cfg.workspace} · mode {self.cfg.mode} · "
              f"session {self.agent.session.id}\n/help for commands")
        for n in self.cfg.notes:
            self.ui.notice(n)
        while True:
            try:
                line = input(_c("1", "\n> ")).strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                continue
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    return 0
                continue
            res = self.agent.run(line)
            self.ui._nl()
            if res.stopped == "interrupted":
                print(_c("33", "[interrupted]"))


def main(argv: list[str] | None = None, args=None) -> int:
    a = args if args is not None else build_parser().parse_args(argv)
    try:
        cfg = config_from_args(a)
        backend = make_backend(cfg.backend)
    except (ValueError, BackendError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.prompt is not None:
        repl = Repl(cfg, backend, interactive=False, yes=a.yes, save=not a.no_save, verbose=a.verbose,
                    resume=a.resume)
        for n in cfg.notes:
            repl.ui.notice(n)
        res = repl.agent.run(a.prompt)
        repl.ui._nl()
        return 0 if res.stopped == "done" else 1
    if not sys.stdin.isatty():
        print("error: interactive mode needs a terminal; use -p for one-shot runs", file=sys.stderr)
        return 2
    return Repl(cfg, backend, interactive=True, yes=False, save=not a.no_save, verbose=a.verbose,
                resume=a.resume).loop()
