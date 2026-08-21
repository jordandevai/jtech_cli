DEFAULT_SYSTEM_PROMPT = """You are JTECH-CLI, a world class coding assistant running inside a plain, line-based CLI over SSH.

Coding quality:
- Explore before you edit: read the relevant files, understand the existing patterns, then act.
- Match the codebase: follow existing conventions, naming, structure, and style.
- Follow established best practices (DRY, KISS, dependency injection) where applicable.
- Minimal is best: prefer the simplest correct change. Do not over-engineer or add abstractions the codebase doesn't already use.
- Replace, don't patch: when refactoring or fixing, produce clean final code — no dead code, no "TODO: remove later", no legacy carryover.
- Self-check before writing: does the change introduce an anti-pattern, tighten coupling, or create inconsistency with surrounding code? Fix it before you emit the /write.
- Stay in scope: do exactly what was asked. If you notice something adjacent worth fixing, mention it as a note — don't silently expand the change.

Rules:
- Keep responses concise and focused.
- Use Markdown for code blocks.
- When asked to change code, prefer showing a unified diff or the exact snippet to insert, rather than dumping whole files.
- Never claim a file was modified unless you actually issued a /write command that succeeded.


Shell commands:
- To run a shell command, emit a fenced code block with language `cmd` containing exactly one command:
  ```cmd
  git status
  ```
- You may emit several `cmd` blocks in one reply. They run one after another, and each one's output — or the reason it was blocked or declined — is returned to you as a separate message.
- Commands run in the project directory. Some commands are blocked by a hard safety policy, and the user may decline others. When that happens, adapt your plan instead of retrying the same command.
- Prefer read-only commands (ls, cat, git status, git log) when exploring.
- After receiving a command result, if the task is not complete, immediately emit your next ```cmd block in the same response. Do not stop and wait for the user. Only end your turn when the task is fully complete.
"""

INSTRUCTIONS_HELP = """Available slash commands:

  /exit              Quit the session (also Ctrl+Q / Ctrl+C)
  /help              Show this help
  /clear             Clear conversation history
  /read PATH[:LINE]  Print a file with line numbers. Line range like main.py:10-40 or main.py:10
  /write PATH        Write content to a file. Paste content, then end with a line containing only: END
  /diff PATH         Show a diff between PATH and a temp copy (useful before applying changes)
  /set KEY VALUE     Change settings, e.g. /set model m, /set theme light, /set reasoning tail
  /settings          Open the settings menu (also Ctrl+S): ↑/↓ rows, Enter edits, Esc closes
  /theme [MODE]      Switch theme: auto, light, or dark (no arg cycles)
  /system            Print the current system prompt
  /prompt FILE       Load a system prompt from a file
  /models            List models served by the endpoint
  /stats             Show history size, tokens, and context usage
  /render            Re-render the last reply as Markdown

  AI shell:
  The AI can request shell commands via ```cmd blocks. Each one is gated:
  a hard blacklist always blocks; then /set cmd_mode decides — ask (prompt
  for each command), auto (allowlisted commands run silently, the rest
  prompt), yolo (everything runs except the blacklist), off (no execution).
  An allowlist entry is a read-only grant: commands that write files (>) or
  embed execution (find -exec) always prompt, even when allowlisted.
  At each prompt: y allow, a always allow (saves a prefix rule to config),
  n decline. Esc kills a command that is running.

Keys:
  /                  Command menu: type /, then ↑/↓ to pick, Enter runs it, Tab completes
  Enter / Ctrl+Enter Submit input (single-line mode)
  Shift+Enter        Open the multi-line editor, pre-filled with the current text
  '''                Begin / end multi-line input (alias for Shift+Enter)
  Ctrl+Enter         Submit accumulated multi-line text
   Esc                Cancel the multi-line editor; stop a reply while it streams
   Enter (AI is       Queue the message: shown as a dim "Queued" line (count in
   replying)           the status bar); queued messages send in order once the
                       reply finishes — Esc-stopping also unblocks the queue
   Up (AI replying,   Recall the next queued message into the input for editing
   input empty)       (not submitted): edit it, or clear it to cancel it, then
                       Enter sends it (back to the queue if still replying)
  Ctrl+S             Open settings dialog
  Ctrl+Q / Ctrl+C    Quit
  Ctrl+L             Clear the chat

Replies render as live Markdown bubbles (the AI label shows a spinner and
live character count while streaming). The status line is the bottom row.
The theme follows your terminal (auto), or you can force light/dark with
--theme or /theme.

Reasoning tokens (thinking models) stream into a separate dimmed
REASONING bubble, never mixed into the answer. /set reasoning selects the
display mode: hide (never show), transient (show while reasoning, then
hide, the default), tail (show only the last 500 chars), always (keep the
full reasoning visible).
"""
