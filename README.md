# Jtech-CLI

A full-screen terminal chat client for `llama-server` (OpenAI-compatible API),
built on Textual. The UI appears immediately: prior conversation renders as chat
bubbles, replies stream live as Markdown, and the input is pinned at the bottom.
The theme follows your terminal's light/dark setting.

## Run

```bash
./run.sh
# or directly:
uv run python -m jtech_cli
# legacy entry shim (still works):
uv run python main.py
```

The TUI needs an interactive terminal; it prints a message and exits otherwise
(e.g. piped output or a non-TTY context).

Extra args pass through: `./run.sh --no-discover`.
UI theme: `./run.sh --theme light` (or `dark`/`auto`; default `auto`).

## First-run setup

On first launch (no `~/.mycli/config.toml` yet) a setup wizard runs:

```
Welcome to JTech CLI! Let's set up your LLM.
Provider: OpenAI-compatible API (e.g. llama-server) — the only supported option.
The URL should look like: http://127.0.0.1:8080/v1
Configuring profile: default
Environment variable holding the API key (blank for a local server with no auth):
Enter server URL: <your url>
Testing connection...
Success! Model found: <real model>
Configuration saved to ~/.mycli/config.toml
```

The three fields are the API-key environment variable name, the base URL, and
the model. The wizard tests the endpoint by querying `/v1/models`; a credential
failure returns to the API-key field and a connection failure returns to the URL
field, and nothing is written until one complete, reachable profile has been
collected. It edits the active profile, or creates `default` on first run.
Re-run it anytime with `./run.sh --setup`.

The API **key value is never requested, stored, displayed, or logged** — only
the name of the environment variable that supplies it.

## Profiles

A profile is one OpenAI-compatible endpoint identity:

```text
name + base_url + model + api_key_env
```

One installation can keep several and switch between them while running, with
`/profile NAME` or the `/profiles` manager. The selection persists for the next
launch. Everything else — theme, temperature, reasoning display, prompt, shell
policy — stays global.

Switching is only allowed while idle. One user turn (its first reply, every
command result, and every nudge) always runs against a single endpoint, model,
and credential, so a mid-turn switch is refused rather than half-applied.

## Configuration

Defaults come from `~/.mycli/config.toml`, overridden by CLI flags:

```toml
[server]
active_profile = "local"
temperature = 0.7
theme = "auto"                  # auto / light / dark; optional
reasoning = "transient"         # hide / transient / tail / always; optional
system_prompt = "You are a helpful assistant."   # optional

[profiles.local]
base_url = "http://127.0.0.1:8080/v1"
model = "qwen3"                 # optional; auto-discovered if unset

[profiles.cloud]
base_url = "https://api.example.com/v1"
model = "cloud-model"
api_key_env = "CLOUD_API_KEY"   # names the variable, never the key itself
```

`api_key_env` names an environment variable that must be set and non-empty when
that profile is used; requests then carry `Authorization: Bearer <value>`. A
profile with no `api_key_env` is treated as a local server needing no
authentication and sends no authorization header. A missing or empty variable is
reported before any request is made — it is never silently skipped.

If `model` is not set, the CLI queries the endpoint at startup and uses the
actual served model — no hardcoded/mock names. If the server serves more than
one model and none is configured, the turn stops with an explicit error instead
of guessing. If the server is unreachable it shows no model info rather than a
fake one.

`--base-url` and `--model` override the selected profile **for that run only**;
they are never written back to the config file, and the status line marks the
run as `(override)`. Activating a profile with `/profile NAME` clears the
override.

### Migrating an older config

A pre-profile config — `base_url` and `model` directly under `[server]` — still
loads: it becomes one profile named `default`, in memory. The file is not
rewritten until the next intentional settings or profile save, which writes the
new format. Mixing the two layouts in one file is refused with an explicit
error rather than resolved by guessing which endpoint was meant.

## The TUI

- **Agent workspace**: the activity pane (85% of the width) on the left, an
  agent sidebar (15%) on the right, with the composer and status line spanning
  the full width below them. The split follows the terminal continuously —
  there is no divider to drag, no collapse button, and no persisted width;
  sidebar text is ellipsized rather than widening. The conversation you type
  into is **Primary**, the coordinator, and it is selected on startup. Every
  agent it dispatches gets its own row, with one line per task indented two
  columns beneath it, and a status glyph carried by shape rather than color so
  a monochrome terminal keeps the signal: `○` idle, `●` running, `◌` waiting,
  `✓` completed, `!` failed.
- **Selecting an agent**: click a row, or `Tab` into the sidebar and press
  `Enter` on the highlighted one. Arrowing through the list only moves the
  highlight — `Enter` or a click commits it — and clicking a task line selects
  its owning agent, since tasks are text inside the agent's row rather than
  targets of their own. Selection shows that agent's own live stream: its
  reasoning, replies, commands, and output. Each stream keeps its own content,
  live tail, and scroll position while another is displayed, and a new agent,
  a status change, or a finished task never steals the selection or scrolls
  something else into view. The status line stays Primary's throughout — it
  describes your profile, model, context, and queue, never the selected
  agent's.
- **Read-only agent view**: with an agent selected the composer, the command
  menu, and the multi-line editor are hidden and replaced by
  `Read only — subagents communicate with their dispatcher.` Nothing you had
  typed is touched: return to Primary and the same draft, selection,
  multi-line text, and queue are exactly where you left them. Text selection
  and copy stay available in the agent's transcript. While an agent is shown,
  `Ctrl+L` reports `Subagent activity is read only; switch to Primary to clear
  chat.` instead of clearing anything, `Esc` does nothing, and `Ctrl+C` copies
  a selection if there is one and otherwise opens the quit confirmation —
  it never clears the Primary draft you cannot see.
- **Automatic dispatch**: there is no `/agent` command and nothing to configure.
  When a request has separable work, the coordinator emits standalone
  `jtech_agent("key", "Label", "profile", "Task label", "the task")` calls of
  its own. It picks one of *your* configured API profiles per agent, so a local
  model can investigate while a cloud model reviews. Several agents in one
  reply start together and run concurrently; each one's final answer returns to
  the coordinator automatically, in the order it dispatched them, and it keeps
  working until your goal is done. Reusing a key sends a follow-up task into
  the same agent's conversation and adds a task row; a new key creates a new
  agent. A key's label and profile are fixed for the session.
  All agents share this one working directory: the coordinator is told to
  parallelize only independent work, and nothing locks or merges concurrent
  edits.
- **Agent commands and approvals**: agents inherit the live `cmd_mode`,
  allowlist, and hard blacklist — there are no per-agent permissions. Approvals
  are serialized, so exactly one prompt is on screen at a time whichever stream
  you are viewing, and its title names the requester (`Run command for
  Coder?`). Choosing **always allow** saves the rule once and any agent already
  waiting behind it re-reads the policy instead of asking you again. `Esc`
  stops Primary's own work only; it never cancels an agent. Quitting stops
  every runtime and kills every command they are running.
- **Ephemeral agent history**: an agent's conversation lives in memory for the
  run. The results the coordinator received are part of *its* history and are
  restored on the next launch, but the agents themselves are not: after a
  restart, the first new task for a key starts a fresh worker. Durable
  per-agent history is separate, unshipped work.
- **Chat bubbles**: your prompts appear as gray "USER" bubbles on a shaded
  surface that spans the full chat width; the model's reply streams live into
  an unshaded "AI" Markdown bubble, so code blocks are syntax-highlighted in
  place. While streaming, the AI label shows a spinner, elapsed time, and a
  live character count — all ticked by a 1s timer, so they keep moving even
  when the stream is silent.
- **Command entries**: an authorized command appears immediately as a dim
  "SYSTEM" entry showing the command and `running…`, and that same entry
  becomes its captured output once the process exits. Shell commands have no
  elapsed-time deadline — a build, a test suite, or a migration runs as long
  as it needs. Stop Primary's command with `Esc`, or exit the app to stop
  every runtime's command at once.
- **Reasoning bubbles**: thinking models' reasoning tokens stream into a
  separate dim "REASONING" bubble, never mixed into the answer bubble.
  `reasoning` (or `/set reasoning`) selects the display mode:
  - `hide` — never show reasoning
  - `transient` (default) — show it while reasoning, then hide it once the
    answer starts
  - `tail` — show only the last 500 characters of the reasoning
  - `always` — keep the full reasoning visible after the reply
- **Pinned input**: a single bordered input line at the bottom, above the
  status line. `Enter` submits and `Ctrl+J` inserts a newline; `Shift+Enter`
  does the same only when your terminal reports it distinctly. The first
  newline opens the multi-line editor and creates the line in the same
  keypress, carrying the current text across unchanged; in the editor `Enter`
  submits, `Ctrl+J` adds another line, and `Esc` cancels. The editor border
  grows with the draft and scrolls once it reaches its maximum height.
- **Multi-line paste**: pasting text that contains a line break — from the
  terminal or from `Ctrl+V` — opens the multi-line editor with every line
  intact, replacing the selection if there is one. A paste never submits by
  itself; review or edit it, then press `Enter`.
- **Esc to stop**: pressing `Esc` while a reply is streaming closes the
  provider response instead of waiting for another token. The partial answer
  stays in the conversation under an `AI · stopped` label, followed by
  `[Response interrupted by user.]`; reasoning is discarded. Future requests
  send only that marker for the stopped turn, never the incomplete answer, so
  half-finished prose or a truncated tool call cannot steer the next reply. It
  applies to Primary's own work, and only while Primary is selected; a selected
  agent's stream is read only and is never cancelled by it.
- **Message queue**: pressing `Enter` while a reply is streaming queues the
  message — it shows as a dim "Queued" line and a count in the status bar — and
  queued messages send in order once the reply finishes or is stopped. A queued
  message starts only after the stopped reply's provider stream has closed and
  its worker has exited, so two requests are never in flight at once. Press
  `Up` (with an empty input) to pull the next queued message back into the
  input for editing, or clear it to cancel it.
- **Status bar**: the bottom row of the app (below the input) shows the active
  profile name, its base URL (no `base_url=` prefix), the configured or
  uniquely discovered model, and context length. A `--base-url`/`--model` run
  is marked `profile: NAME (override)`. It re-renders immediately after
  settings or profile changes.
- **Themes**: calm-blue custom themes (`jtech-dark` / `jtech-light`) are
  auto-detected from your terminal's light/dark background (the `COLORFGBG`
  env var), with a single calm blue for primary highlights (input, dialogs,
  links), neutral grays for bubbles and system text, and red only for errors.
  Override with `--theme light|dark`, `/theme`, or the settings dialog.
- **Profile manager**: `/profiles` opens a modal to list, add, edit, rename,
  activate, and delete profiles. Arrow keys move, Enter selects, Esc backs out
  one step. Deleting the active profile is refused — activate another first, so
  no endpoint is ever chosen for you. Nothing is probed here: a local server may
  legitimately be stopped while its profile is edited.
- **Contextual `Ctrl+C`**: one key, resolved in priority order — copy a
  non-empty selection; otherwise clear the chat composer (single-line or
  multi-line, whitespace included) without leaving; otherwise, with an empty
  composer, open a quit confirmation. **Stay** is the default, `Esc` returns to
  the previous screen, arrows/`Tab` plus `Enter` choose, and pressing `Ctrl+C`
  again in that prompt exits immediately. Opened above `/settings`, `/profiles`,
  or a command prompt it leaves that screen and any unsaved field untouched.
  `Ctrl+Q` and `/exit` still quit immediately.
- **Clipboard**: `Ctrl+V` pastes Textual's local clipboard into the focused
  single-line or multi-line editor, so copy-then-paste inside the app works
  everywhere with no clipboard integration at all. Copying *out* to the system
  clipboard is a best-effort OSC 52 request that your terminal — and, under
  tmux, its configuration — may or may not honour; the app is never told either
  way, so it never claims success. Pasting *in* from the system clipboard stays
  with your terminal's own shortcut and bracketed paste (`Ctrl+Shift+V`,
  `Shift+Insert`, or `Cmd+V` are common examples, not guaranteed bindings). For
  environment-specific setup, see
  [tmux's clipboard documentation](https://github.com/tmux/tmux/wiki/Clipboard).
- **Connection errors**: a missing profile or an unreachable endpoint renders a
  clear notice in the chat (and a startup hint to open `/profiles`) instead of a
  raw stack trace. An invalid config file stops startup with one actionable
  line on stderr rather than launching against defaults.

## Keys

| Key | Action |
| --- | --- |
| `Enter` | Submit input · submit the multi-line editor |
| `Ctrl+J` | Insert a newline (the first one opens and expands the multi-line editor) |
| `Shift+Enter` | Same as `Ctrl+J` when the terminal reports it distinctly |
| `Esc` | Cancel multi-line editor · stop a reply while it streams |
| `Ctrl+S` | Open settings dialog |
| `Ctrl+L` | Clear the chat (refused while a subagent is selected) |
| `Ctrl+Q` | Quit immediately |
| `Ctrl+C` | Copy selection · otherwise clear composer · when empty, confirm quit |
| `Ctrl+V` | Paste the local clipboard into the focused editor |

### Newline keys and your terminal

`Ctrl+J` sends a literal LF, which terminals preserve, so it is the supported
newline key. If your terminal reports `Shift+Enter` distinctly — or you map it
to a single LF — it invokes exactly the same action. If your terminal sends a
plain `Enter` for both keys, the application cannot tell them apart, because
the modifier was discarded before it arrived: use `Ctrl+J`, or map
`Shift+Enter` to one LF. Jtech-CLI has no terminal-specific code path and never
edits your terminal, tmux, or editor configuration.

- **Konsole** (optional): in *Settings → Edit Current Profile → Keyboard*, copy
  the current keyboard scheme rather than editing
  `/usr/share/konsole/default.keytab` in place, replace the `Return+Shift`
  output with `key Return+Shift : "\n"`, then select the copied scheme. That
  exact rule is untested on the measured Konsole + tmux stack — no shipped
  keytab uses a bare `"\n"` output.
- **VS Code**: a `sendSequence` keybinding must send one `"\u000a"`. The
  common `"\\\r\n"` recipe is not a newline — it sends a backslash, then
  Enter (which submits), then LF.
- **tmux**: tmux decides how modified keys are reported. On the measured tmux
  3.4 with `extended-keys off` — the default — `Shift+Enter` is unavailable but
  `Ctrl+J` works. Turning `extended-keys` on is not the fix here: it reports
  `shift+\r`, which is not `shift+enter`.

## Commands

| Command | Description |
| --- | --- |
| `/exit` | Quit |
| `/help` | Show help |
| `/clear` | Clear history |
| `/read PATH[:LINE]` | Print a file with line numbers (`main.py:10-40`) |
| `/write PATH` | Write content to a file (`Ctrl+J` newline, `Shift+Enter` when supported, `Enter` writes) |
| `/diff PATH` | Create a temp copy of a file to diff against |
| `/set KEY VALUE` | Change a global setting: temperature / theme / reasoning / cmd_mode / debug_level |
| `/settings` | Open the settings dialog (also Ctrl+S) |
| `/profiles` | Manage API profiles: list, add, edit, rename, activate, delete |
| `/profile NAME` | Activate a named API profile (persists for the next launch) |
| `/theme [MODE]` | Switch theme: auto / light / dark |
| `/system` | Show current system prompt |
| `/prompt FILE` | Load a system prompt from a file |
| `/models` | List models served by the active profile's endpoint |
| `/stats` | Show history size, tokens, and context usage |
| `/render` | Re-render the last reply as Markdown |

Multi-line input: press `Ctrl+J` to add a line, then `Enter` to send.
`Shift+Enter` performs the same action only when the terminal reports it
distinctly. The editor border grows with the draft and scrolls after reaching
its existing maximum height. No closing line is needed.

## Project structure

The code lives in a `jtech_cli` package. Modules are grouped by responsibility,
with dependencies injected through the composition root rather than reached for
as globals.

```
jtech_cli/
  cli.py          # composition root: arg parsing + dependency wiring + entry point
  configuration/  # settings schema, profiles, paths, and TOML persistence
    paths.py
    profiles.py   # profile identity, catalog rules, credential resolution
    settings.py
    storage.py
  resources/      # static-only package data; no Python modules
    config/defaults.toml
    prompts/*.md
    themes/dark.toml
    themes/light.toml
    styles/tui.css
  resource_loader.py # validated access to bundled resources
  config.py       # stable configuration API facade
  prompts.py      # Markdown prompt API
  theme.py        # TOML theme API and terminal detection
  tui.py          # stable public TUI import/facade
  tui_app.py      # app lifecycle, agent catalog, profiles, approvals, dispatch
  tui_runtime.py  # one conversation's autonomous model/command loop (all agents)
  tui_screens.py  # settings and command-approval modals
  tui_widgets.py  # reusable inputs, events, and output sink
  commands.py     # slash-command registry/handlers + CommandContext (DI container)
  cmd_tools.py    # shell policy, parsing, and execution result helpers
  session.py      # JSONL conversation history
  llm_client.py   # streaming OpenAI-compatible client
  server_info.py  # server introspection (/models, /tokenize)
  file_tools.py   # /read, /write, /diff helpers
  wizard.py       # first-run setup wizard (line-based, runs before the TUI)
```

## Test

```bash
uv run pytest -q
uv run --with ruff ruff check .
```
