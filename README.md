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
Provider: OpenAI-compatible API (OpenAPI) — the only supported option.
Enter server URL: <your url>
Testing connection...
Success! Model found: <real model>
Configuration saved to ~/.mycli/config.toml
```

The wizard tests the endpoint by querying `/v1/models`; if it fails it loops back
to the URL prompt. On success it saves the config. Re-run it anytime with
`./run.sh --setup`.

## Configuration

Defaults come from `~/.mycli/config.toml`, overridden by CLI flags:

```toml
[server]
base_url = "http://your-host:port/v1"
model = "the-real-model-name"   # optional; auto-discovered if unset
temperature = 0.7
theme = "auto"                  # auto / light / dark; optional
reasoning = "transient"         # hide / transient / tail / always; optional
system_prompt = "You are a helpful assistant."   # optional
```

If `model` is not set, the CLI queries the endpoint at startup and uses the
actual served model — no hardcoded/mock names. If the server is unreachable it
shows no model info rather than a fake one.

## The TUI

- **Chat bubbles**: your prompts appear as gray "USER" bubbles on a shaded
  surface that spans the full chat width; the model's reply streams live into
  an unshaded "AI" Markdown bubble, so code blocks are syntax-highlighted in
  place. While streaming, the AI label shows a spinner, elapsed time, and a
  live character count — all ticked by a 1s timer, so they keep moving even
  when the stream is silent. Command output appears as dim "SYSTEM" messages.
- **Reasoning bubbles**: thinking models' reasoning tokens stream into a
  separate dim "REASONING" bubble, never mixed into the answer bubble.
  `reasoning` (or `/set reasoning`) selects the display mode:
  - `hide` — never show reasoning
  - `transient` (default) — show it while reasoning, then hide it once the
    answer starts
  - `tail` — show only the last 500 characters of the reasoning
  - `always` — keep the full reasoning visible after the reply
- **Pinned input**: a single bordered input line at the bottom, above the
  status line. Press `Enter`/`Ctrl+Enter` to submit, or `Shift+Enter` to open
  the multi-line editor pre-filled with the current text (typing `'''` is an
  alias); in multi-line mode `Ctrl+Enter` submits and `Esc` cancels.
- **Esc to stop**: pressing `Esc` while a reply is streaming aborts the
  generation and discards the partial bubble (with a dim "Generation stopped."
  note).
- **Message queue**: pressing `Enter` while a reply is streaming queues the
  message — it shows as a dim "Queued" line and a count in the status bar — and
  queued messages send in order once the reply finishes or is stopped. Press
  `Up` (with an empty input) to pull the next queued message back into the
  input for editing, or clear it to cancel it.
- **Status bar**: the bottom row of the app (below the input) shows the base
  URL (no `base_url=` prefix), the active model, context length, and the
  history file path. It re-renders immediately after settings change.
- **Themes**: calm-blue custom themes (`jtech-dark` / `jtech-light`) are
  auto-detected from your terminal's light/dark background (the `COLORFGBG`
  env var), with a single calm blue for primary highlights (input, dialogs,
  links), neutral grays for bubbles and system text, and red only for errors.
  Override with `--theme light|dark`, `/theme`, or the settings dialog.
- **Connection errors**: an empty or unreachable `base_url` renders a clear
  notice in the chat (and a startup hint to open `/settings`) instead of a raw
  stack trace.

## Keys

| Key | Action |
| --- | --- |
| `Enter` / `Ctrl+Enter` | Submit input |
| `Shift+Enter` | Open multi-line editor (current text pre-filled) |
| `'''` | Begin / end multi-line input |
| `Esc` | Cancel multi-line editor · stop a reply while it streams |
| `Ctrl+S` | Open settings dialog |
| `Ctrl+L` | Clear the chat |
| `Ctrl+Q` / `Ctrl+C` | Quit |

## Commands

| Command | Description |
| --- | --- |
| `/exit` | Quit |
| `/help` | Show help |
| `/clear` | Clear history |
| `/read PATH[:LINE]` | Print a file with line numbers (`main.py:10-40`) |
| `/write PATH` | Write content to a file |
| `/diff PATH` | Create a temp copy of a file to diff against |
| `/set KEY VALUE` | Change model / base_url / temperature / theme / reasoning |
| `/settings` | Open the settings dialog (also Ctrl+S) |
| `/theme [MODE]` | Switch theme: auto / light / dark |
| `/system` | Show current system prompt |
| `/prompt FILE` | Load a system prompt from a file |
| `/models` | List models served by the endpoint |
| `/stats` | Show history size, tokens, and context usage |
| `/render` | Re-render the last reply as Markdown |

Multi-line input: press `Shift+Enter` (or type `'''`), edit, then `Ctrl+Enter`
with `'''` on its own line to finish.

## Project structure

The code lives in a `jtech_cli` package. Modules are grouped by responsibility,
with dependencies injected through the composition root rather than reached for
as globals.

```
jtech_cli/
  cli.py          # composition root: arg parsing + dependency wiring + entry point
  tui.py          # the Textual TUI (ChatApp) + settings dialog + output sink
  commands.py     # slash-command registry/handlers + CommandContext (DI container)
  config.py       # Settings + TOML config persistence
  session.py      # JSONL conversation history
  llm_client.py   # streaming OpenAI-compatible client
  server_info.py  # server introspection (/models, /tokenize)
  file_tools.py   # /read, /write, /diff helpers
  wizard.py       # first-run setup wizard (line-based, runs before the TUI)
  prompts.py      # default system prompt + /help text
```

## Test

```bash
uv run pytest -q
uv run --with ruff ruff check .
```
