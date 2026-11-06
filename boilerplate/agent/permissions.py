"""Allow / ask / deny decisions for tool calls.

The order is fixed and documented in docs/agent_prompt_coding_cli.md section 5. Nothing the model
writes can change a decision: rules come from config files, the command line, or the user's
answers at an approval prompt.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

from .config import MODES, AgentConfig

_RULE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*$", re.S)

MODE_DEFAULTS = {
    "plan": {"read": "allow", "write": "deny", "execute": "deny"},
    "default": {"read": "allow", "write": "ask", "execute": "ask"},
    "accept-edits": {"read": "allow", "write": "allow", "execute": "ask"},
    "auto": {"read": "allow", "write": "allow", "execute": "allow"},
}


def parse_rule(rule: str) -> tuple[str, str | None]:
    m = _RULE.match(rule)
    if not m:
        raise ValueError(f"bad permission rule {rule!r}; expected tool or tool(pattern)")
    return m.group(1), m.group(2)


def rule_matches(rule: str, tool: str, subject: str) -> bool:
    name, pat = parse_rule(rule)
    if name != tool and name != "*":
        return False
    return pat is None or fnmatch.fnmatchcase(subject, pat)


@dataclass
class Request:
    tool: str
    risk: str  # read | write | execute
    subject: str  # workspace-relative path or the command line
    sensitive: bool = False
    hard_deny: str | None = None
    allowlisted: str | None = None  # shell allowlist pattern that matched, if any
    runs_workspace_code: bool = False


@dataclass
class Decision:
    action: str  # allow | ask | deny
    reason: str
    source: str  # hard | deny-rule | plan | taint | sensitive | allow-rule | session | allowlist | ask-rule | mode | user


@dataclass
class PermissionEngine:
    cfg: AgentConfig
    interactive: bool = True
    yes: bool = False
    session_allow: list[str] = field(default_factory=list)
    tainted: bool = False
    unreviewed_writes: bool = False

    def __post_init__(self):
        for r in self.cfg.allow + self.cfg.ask + self.cfg.deny:
            parse_rule(r)

    @property
    def mode(self) -> str:
        return self.cfg.mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg.mode = mode

    def _first(self, rules, req) -> str | None:
        return next((r for r in rules if rule_matches(r, req.tool, req.subject)), None)

    def decide(self, req: Request) -> Decision:
        d = self._decide(req)
        if d.action == "ask" and not self.interactive:
            if self.yes and d.source in ("ask-rule", "mode", "allowlist-escalated"):
                return Decision("allow", d.reason + " (approved by --yes)", "yes")
            return Decision("deny", d.reason + "; no one is available to approve in non-interactive mode", d.source)
        return d

    def _decide(self, req: Request) -> Decision:
        if req.hard_deny:
            return Decision("deny", req.hard_deny, "hard")
        r = self._first(self.cfg.deny, req)
        if r:
            return Decision("deny", f"denied by rule {r}", "deny-rule")
        if self.mode == "plan" and req.risk != "read":
            return Decision("deny", "plan mode is read-only", "plan")
        if self.tainted and req.risk != "read":
            return Decision("ask", "recent tool output looked like a prompt injection", "taint")
        if req.sensitive:
            return Decision("ask", f"{req.subject} looks like a credential or key file", "sensitive")
        r = self._first(self.cfg.allow, req)
        if r:
            return Decision("allow", f"allowed by rule {r}", "allow-rule")
        r = self._first(self.session_allow, req)
        if r:
            return Decision("allow", f"allowed for this session ({r})", "session")
        if req.allowlisted and self.mode != "plan":
            if self.unreviewed_writes and req.runs_workspace_code:
                return Decision("ask", "files were changed without review since the last approved command, "
                                "and this command runs workspace code", "allowlist-escalated")
            return Decision("allow", f"on the shell allowlist ({req.allowlisted})", "allowlist")
        r = self._first(self.cfg.ask, req)
        if r:
            return Decision("ask", f"rule {r} requires approval", "ask-rule")
        action = MODE_DEFAULTS[self.mode][req.risk]
        return Decision(action, f"{self.mode} mode: {req.risk} actions are {action}", "mode")

    def record(self, req: Request, decision: Decision, user_approved: bool) -> None:
        """Update taint and review state after a call has been decided."""
        if req.risk == "read":
            return
        if user_approved:
            self.tainted = False
            if req.risk == "execute":
                self.unreviewed_writes = False
        elif decision.action == "allow" and req.risk == "write":
            self.unreviewed_writes = True

    def grant_session(self, rule: str) -> None:
        parse_rule(rule)
        if rule not in self.session_allow:
            self.session_allow.append(rule)

    def describe(self) -> str:
        lines = [f"mode: {self.mode}"]
        for label, rules in (("allow", self.cfg.allow), ("session allow", self.session_allow),
                             ("ask", self.cfg.ask), ("deny", self.cfg.deny)):
            lines.append(f"{label}: {', '.join(rules) if rules else '(none)'}")
        lines.append(f"tainted: {self.tainted}")
        return "\n".join(lines)
