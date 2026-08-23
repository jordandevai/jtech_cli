Available slash commands:

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
