Available slash commands:

  /exit              Quit the session (also Ctrl+Q)
  /help              Show this help
  /clear             Clear conversation history
  /read PATH[:LINE]  Print a file with line numbers. Line range like main.py:10-40 or main.py:10
  /write PATH        Write content to a file. Paste content, then end with a line containing only: END
  /diff PATH         Show a diff between PATH and a temp copy (useful before applying changes)
  /set KEY VALUE     Change a global setting: temperature, theme, reasoning, cmd_mode,
                     debug_level. The endpoint and model belong to a profile, not here.
  /settings          Open the settings menu (also Ctrl+S): ↑/↓ rows, Enter edits, Esc closes
  /profiles          Manage API profiles: list, add, edit, rename, activate, delete
  /profile NAME      Activate a named API profile; the choice persists for the next launch
  /theme [MODE]      Switch theme: auto, light, or dark (no arg cycles)
  /system            Print the effective prompt and its source
  /prompt FILE       Load and persist an additional prompt file
  /prompt reload     Reload the selected prompt file
  /prompt reset      Return to the bundled runtime prompt
  /models            List models served by the active profile's endpoint
  /stats             Show history size, tokens, and context usage
  /render            Re-render the last reply as Markdown

  AI shell:
  The AI can request shell commands via standalone `jtech_cmd(...)` calls. Each one is gated:
  Calls may be interleaved with commentary, but each call must begin its own
  line and occupy that line. Do not put calls inline in prose or Markdown fences.
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
  Ctrl+Q             Quit immediately
  Ctrl+C             Copy the selection if there is one; otherwise clear the
                     chat composer (single- or multi-line, whitespace included)
                     without leaving; otherwise, with an empty composer, open
                     the quit confirmation. Over settings, profiles, or a
                     command prompt it opens that confirmation and leaves the
                     screen and its unsaved fields untouched.
  Ctrl+C / Esc       In the quit confirmation: Ctrl+C exits immediately, Esc
  (quit prompt)      returns to the previous screen. Stay is the default;
                     arrows/Tab move and Enter confirms.
  Ctrl+V             Paste the application's local clipboard into the focused
                     single-line or multi-line editor. Pasting from the system
                     clipboard uses your terminal's own shortcut and bracketed
                     paste, not an application key.
  Ctrl+L             Clear the chat

API profiles:
  A profile is one OpenAI-compatible endpoint identity: name, base URL, model,
  and the name of the environment variable holding its API key. The key value
  itself is never stored in the config file, shown, or logged; a profile with no
  variable set is treated as a local server needing no authentication. Leave the
  model blank to use the single model the server reports. The status line shows
  the active profile, its URL, and its model. Switching profiles is only allowed
  while idle: one user turn — its first reply, every command result, and every
  nudge — always runs against one endpoint, model, and credential.

Replies render as live Markdown bubbles (the AI label shows a spinner and
live character count while streaming). The status line is the bottom row.
The theme follows your terminal (auto), or you can force light/dark with
--theme or /theme.

Reasoning tokens (thinking models) stream into a separate dimmed
REASONING bubble, never mixed into the answer. /set reasoning selects the
display mode: hide (never show), transient (show while reasoning, then
hide, the default), tail (show only the last 500 chars), always (keep the
full reasoning visible).
