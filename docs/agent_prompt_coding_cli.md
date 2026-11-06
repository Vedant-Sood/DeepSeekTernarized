# Agent prompt: a terminal coding agent on the Boilerplate engine

This is a self-contained brief for an engineer or coding agent. It specifies `boilerplate agent`,
an interactive command-line coding agent that drives a language model served by the Boilerplate
engine (or any OpenAI-compatible endpoint) and lets it read, search and change code in one
workspace through a small set of tools. The first implementation lives in `boilerplate/agent/`;
this document is the contract it is tested against and the list of what remains.

The design is original. It borrows only the general shape every tool-using agent shares: a model,
a loop, tools, and a human who approves risky actions.

## 1. Goals and non-goals

Goals:

1. A usable REPL and a one-shot mode (`-p "task"`) for coding tasks in a local repository.
2. The same agent runs against three backends: the engine in-process, the engine behind
   `boilerplate serve`, and any OpenAI-compatible HTTP endpoint (vLLM, llama.cpp server, a hosted
   API).
3. Safe by default. Every action that writes, executes or leaves the workspace is either allowed by
   an explicit rule or approved by the user. The model's output is never a source of permission.
4. Tests run on CPU without a model, a GPU or network access, using a scripted mock model.

Non-goals for v1: a full-screen TUI, a plugin marketplace, web browsing or fetching, multi-agent
orchestration, an OS-level container sandbox (see section 6.7 for the hook), and training the
ternary model to be good at tool use (that is M5 and later evaluation work).

## 2. Architecture

```
boilerplate/agent/
  config.py       layered YAML config, permission modes, rules
  backends.py     Backend protocol; OpenAI-compatible HTTP client (SSE); in-process engine; mock
  chat_format.py  chat template rendering and text tool-call parsing for raw-completion models
  tools.py        tool registry, JSON-schema argument validation, the built-in tools
  sandbox.py      workspace path confinement, sensitive-path rules, subprocess runner
  permissions.py  allow / ask / deny decisions, session grants, taint escalation
  security.py     secret redaction, prompt-injection heuristics, untrusted-output framing
  session.py      JSONL transcripts (0600), resume, token estimate, context compaction
  loop.py         the agent loop
  cli.py          REPL, slash commands, one-shot mode, approval prompts
boilerplate/serve.py  minimal OpenAI-compatible HTTP server over Engine
```

Data flow for one user turn:

1. Append the user message to the session.
2. If the estimated context exceeds the compaction threshold, compact (section 7).
3. Call the backend with the messages and the tool schemas; stream text to the terminal.
4. For each tool call, in order: parse and validate the arguments against the tool's JSON
   schema, decide the permission (section 5), ask the user if needed, run the tool in the
   sandbox, redact and frame the output, and append it as a `tool` message.
5. Repeat from step 3 until the model answers without a tool call, the step limit is hit, or the
   user interrupts.

## 3. Backends

All backends implement `chat(messages, tools, on_text) -> AssistantTurn(text, tool_calls, usage)`.

- **OpenAI-compatible HTTP** (`--endpoint URL --model NAME`). `POST {base}/chat/completions` with
  `stream: true`, parsing server-sent events and accumulating `delta.tool_calls` by index. Native
  tool calling is used when the server supports it; with `tool_protocol: text` the tool list is
  put into the system prompt and calls are parsed from text instead. Uses only the standard
  library (`urllib`), honours `HTTPS_PROXY`, verifies TLS, has a request timeout and a response
  size cap. The API key is read from the environment variable named by `api_key_env`; it is never
  written to config files, transcripts or logs. Plain `http://` is refused for non-loopback hosts
  unless `allow_insecure_http: true`.
- **In-process engine** (`--checkpoint DIR`). Loads `boilerplate.engine.Engine`, renders the
  conversation with the checkpoint's chat template (`tokenizer_config.json` `chat_template` via
  Jinja2 in a sandboxed environment, falling back to a plain role-tagged template), generates
  with streaming, and parses tool calls from text.
- **Mock** (tests only). Returns scripted turns and records what it was sent.

Text tool-call protocol (for models without native tool calling): the system prompt describes the
tools and asks for

```
<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>
```

The parser accepts several calls per turn, tolerates surrounding prose and code fences, and
reports malformed JSON back to the model as a tool error rather than crashing.

## 4. Tools

Each tool declares a name, description, JSON schema and a risk class: `read`, `write` or
`execute`. Arguments are validated (types, required keys, no unknown keys, length caps) before
any permission check.

| Tool | Risk | Behaviour |
|---|---|---|
| `read_file` | read | Text read with optional line offset and limit; numbered lines; size cap; binary files refused. |
| `list_dir` | read | One directory level, sorted, capped entry count. |
| `glob` | read | Workspace-relative glob, capped results, ignores `.git/`. |
| `grep` | read | Python regex over text files with optional glob filter, capped matches and line length, regex compile timeout guarded by a pattern length cap. |
| `write_file` | write | Create or overwrite; parent directories created inside the workspace only; shows a diff for approval. |
| `edit_file` | write | Replace an exact string that must occur exactly once (or `replace_all`); shows a unified diff. |
| `shell` | execute | Runs a command (see section 6.4), returns exit code, stdout and stderr, truncated. |
| `git` | read or execute | Subcommand allowlist. `status`, `diff`, `log`, `show`, `branch` (list) are read; `add`, `commit`, `checkout`, `switch`, `restore`, `stash` are execute; `push`, `reset --hard`, `clean`, `config`, `remote` and anything else are refused unless explicitly allowed by rule. |

git hardening: global options before the subcommand are refused, as are options that run programs,
write elsewhere or need a terminal (`--output`, `--ext-diff`, `--no-index`, `-O`, `--exec`, `-i`...,
including git's abbreviated long options). Every run adds `-c core.fsmonitor=false -c core.pager=cat
-c diff.external= -c credential.helper= -c protocol.allow=never` and `--no-ext-diff --no-textconv`
for diff/log/show. If the repository's `.git/config` sets keys that can run programs (`core.fsmonitor`,
`filter.*`, `diff.*.textconv`, aliases, helpers...), every git call asks and the prompt says why.

Tool output is capped (default 30 000 characters) with a note saying how much was cut.

## 5. Permissions

Decisions are `allow`, `ask` or `deny`. The decision for a call is computed in this order, and the
first match wins:

1. **Hard denies** that no setting overrides: paths outside the workspace, writes into `.git/`
   internals, and the built-in destructive git operations unless a rule allows them.
2. **Deny rules** from any config layer.
3. **Taint escalation**: after a tool returned output flagged as a likely prompt injection, the
   next write or execute call is `ask` regardless of mode or allow rules (cleared once the user
   answers).
4. **Sensitive paths** (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `.ssh/`, `.aws/`, `.netrc`,
   `*credentials*`, `*.p12`...): reads and writes are `ask`.
5. **Allow rules** from the user config and the command line, and grants made during the session
   ("always allow this for the session").
   A shell-allowlist match also counts as allow here, except that a command that runs workspace
   code (pytest, make, npm, python...) asks if files were written without user review since the
   last command the user approved. Otherwise `accept-edits` would let a model write a
   `conftest.py` and run it with no human in the loop.
6. **Ask rules**.
7. **Mode default**:

| Mode | read | write | execute |
|---|---|---|---|
| `plan` | allow | deny | deny |
| `default` | allow | ask | ask |
| `accept-edits` | allow | allow | ask |
| `auto` | allow | allow | allow |

In non-interactive mode (`-p`) an `ask` becomes `deny` and the model is told so, unless `--yes`
was given, which upgrades `ask` from ask rules, mode defaults and the allowlist escalation to
allow (never hard denies, deny rules, plan mode, taint or sensitive paths).

Rules are strings `tool` or `tool(pattern)`, with `fnmatch` patterns over the command line for
`shell`/`git` and over the workspace-relative path for file tools, for example
`shell(pytest *)`, `git(commit *)`, `write_file(docs/*)`.

**Config trust:** the project file `.boilerplate/agent.yaml` lives in the repository the agent is
working on, which may be untrusted. It can add deny and ask rules and lower limits, but its allow
rules, mode, endpoint, API-key variable and shell settings are ignored unless the workspace is
listed in the user config's `trusted_workspaces`.

## 6. Sandbox

### 6.1 Path confinement

Every path argument is resolved against the workspace root with `os.path.realpath`, after which it
must be the root or inside it. This rejects `..`, absolute paths elsewhere, and symlinks (at any
component) that point outside. Writes refuse to follow a final-component symlink.

### 6.2 File limits

Reads are capped (default 2 MB per file); writes are capped (default 1 MB). Binary files (a NUL
byte in the first 8 KB) are not returned to the model.

### 6.3 Environment scrubbing

Subprocesses get a copy of the environment without variables whose names look like credentials
(`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSW*`, `*CREDENTIAL*`, `*AUTH*`, `AWS_*`, `GH_*`, `GITHUB_*`,
`HF_*`, `OPENAI_*`, `ANTHROPIC_*`) and all `GIT_CONFIG*` variables (they inject git config such as
auth headers). Variables in `env_passthrough` are re-added; only a trusted config can set it.
`SSH_AUTH_SOCK` is removed so commands cannot use the user's SSH agent. `GIT_TERMINAL_PROMPT=0` and `PAGER=cat` are set.

### 6.4 Command execution

- By default commands are tokenised with `shlex` and run without a shell (`shell=False`), so
  unquoted pipes, redirects, `;`, `&&`, backticks and `$(...)` are refused with a clear error.
  Quoted text (`python -c "a; b"`) reaches the program literally, since no shell interprets it. Setting
  `shell.allow_shell_syntax: true` (user config only) runs them through `/bin/sh -c` instead.
- Commands are matched against `shell.allowlist` (default: common read-only and build/test
  programs) and `shell.denylist` (`sudo`, `su`, `rm -rf`, `curl`, `wget`, `ssh`, `scp`, `nc`, `dd`,
  `mkfs`, `chmod -R 777`, `pip install`, `git push`...). The denylist wins and is a hard deny. Both
  look through prefixes such as `env`, `nice`, `timeout` and `xargs`, and the denylist reduces
  `/usr/bin/curl` to `curl`. A wrapped command never matches the allowlist, nor does `find` with
  `-exec`/`-delete`, `sh -c`, or an allowlisted program whose path arguments leave the workspace
  or name a credential file. A command on neither list falls through to the mode default.
- cwd is the workspace root; stdin is `/dev/null`; the process runs in its own process group; on
  timeout (default 120 s, max 600 s) the whole group is killed.
- stdout and stderr are read with a byte cap; on POSIX, `RLIMIT_FSIZE`, `RLIMIT_CPU` and
  `RLIMIT_NOFILE` are applied in the child.

### 6.5 Network

The agent itself makes no network calls except to its configured model endpoint. Commands that
reach the network are denylisted by default; this is a speed bump, not isolation (any allowed
interpreter can open a socket). Section 6.7 is the real fix.

### 6.6 Secrets

All text written to transcripts and all tool output given to the model (local or remote) passes
through redaction: private key blocks, `ghp_`/`gho_`/`github_pat_`, `sk-`, `hf_`, AWS access keys,
`xox[bp]-` Slack tokens, `Bearer` headers, `password=` style assignments, and the literal values of
credential-looking environment variables present at start-up.

### 6.7 OS isolation (recommended, not yet implemented)

For untrusted repositories, run the agent inside a container or `bwrap`/`firejail` with the
workspace bind-mounted and no network. The config has `sandbox.wrapper` (a command prefix such as
`["bwrap", "--ro-bind", "/", "/", ...]`) that is prepended to every shell command when set.

## 7. Sessions and context

- Transcripts are JSONL under `$XDG_STATE_HOME/boilerplate/agent/sessions/` (default
  `~/.local/state/...`), directory mode 0700, files 0600, redacted. `--resume ID` and `/resume`
  reload one; `/sessions` lists them.
- Token use is estimated with the engine tokenizer when local, else characters / 3.5. When it
  passes `compact_at` (default 75 % of `context_window`), older messages are replaced by a summary
  the model writes of them (falling back to a mechanical digest if that fails), keeping the system
  prompt, the summary and the last `keep_recent` messages. A tool call and its result are never
  separated.

## 8. CLI

```
boilerplate agent [--workspace DIR] [--endpoint URL --model NAME | --checkpoint DIR]
                  [--mode plan|default|accept-edits|auto] [--allow RULE]... [--deny RULE]...
                  [-p PROMPT] [--yes] [--resume ID] [--config FILE] [--max-steps N]
boilerplate serve CHECKPOINT [--host 127.0.0.1] [--port 8000] [--api-key-env VAR]
```

Slash commands: `/help`, `/exit`, `/clear`, `/compact`, `/mode MODE`, `/permissions`,
`/allow RULE`, `/deny RULE`, `/sessions`, `/resume ID`, `/tools`, `/cost`.

Approval prompt for a risky call shows the tool, its arguments (a diff for edits) and offers
`y` (once), `a` (always this session, for this exact rule pattern), `n` (deny, with optional
reason given to the model). Ctrl-C during generation or a tool run cancels the current step, not
the session.

## 9. `boilerplate serve`

Minimal OpenAI-compatible server over `Engine`: `GET /v1/models`, `POST /v1/chat/completions`
(streaming and not), using the same chat template and text tool-call parser so `tools` work.
Binds to 127.0.0.1 by default and refuses a non-loopback bind unless an API key is configured;
compares keys in constant time; caps request body size, `max_tokens` and message count;
serialises generation behind a lock. It is a development server, not a multi-tenant service.

## 10. Tests (mock model, CPU, no network)

- Path confinement: `..`, absolute paths, symlink escapes, final-component symlink writes.
- Edit semantics: missing string, non-unique string, `replace_all`.
- Permission order: hard deny beats allow; deny beats allow; project config cannot add allows;
  taint escalation; non-interactive ask becomes deny; `--yes` does not override hard denies.
- Shell: metacharacters refused without `allow_shell_syntax`; denylist; timeout kills the process
  group; output truncation; credential variables are not in the child environment.
- Git: read subcommands allowed, `push` refused by default.
- Redaction of each secret family; injection heuristics flag known phrasings.
- Text tool-call parser: multiple calls, fences, malformed JSON.
- OpenAI backend against a local `http.server` stub: streaming text and streamed tool calls.
- Agent loop end to end with a scripted model: read, edit, run tests, finish; step limit.
- Compaction keeps tool-call pairs together. Transcripts are created with mode 0600.
- `serve`: request validation and a round trip with a stub engine.

## 11. Acceptance criteria

1. `pytest tests/test_agent*.py tests/test_serve.py` passes on CPU without network.
2. `boilerplate agent --endpoint http://127.0.0.1:8000/v1 --model x -p "list the files"` works
   against `boilerplate serve` on the tiny checkpoint (output will be nonsense; the plumbing is
   what is tested).
3. No code path executes a write or execute tool without either an allow decision or a user
   approval, which is checked by a test that runs every tool under `plan` and `default` mode with
   an always-deny prompter.

Status 2026-11-06: all three hold on CPU (`tests/test_agent.py`, `tests/test_serve.py`; criterion 2
run by hand against `boilerplate make-tiny`, both through `serve` and with `--checkpoint`).

## 12. Follow-ups

- OS-level sandbox wrapper presets (bwrap, Docker) and a `--sandbox` flag.
- Measure tool-use quality of Mode A against Mode C/D checkpoints on a small agentic benchmark
  once a chat-tuned checkpoint runs on the engine (fits the M5 report).
- MCP client support, so external tool servers can be attached under the same permission system.
- A richer TUI (diff viewer, interruptible streaming display).
- Regex search runs Python `re` on workspace files; a pathological pattern can still backtrack for
  a long time on one large file. A per-search time budget (or the `regex` module's timeout) would
  close that.
- Include `boilerplate/agent/` and `boilerplate/serve.py` in the next whole-codebase security review.
