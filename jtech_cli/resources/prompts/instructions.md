Available slash commands:

  /exit              Quit the session (also Ctrl+Q)
  /help              Show this help
  /clear             Clear conversation history
  /read PATH[:LINE]  Print a file with line numbers. Line range like main.py:10-40 or main.py:10
  /write PATH        Write content to a file. Ctrl+J adds a newline (Shift+Enter too,
                     when your terminal reports it), Enter writes it
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

  Agents:
  The AI you talk to is the coordinator. When a request has separable work, it
  dispatches subagents itself with standalone `[[[jtech_agent]]]` blocks — you
  never type a command for it, and there is no /agent or /agents. Each agent
  gets a name you see in the sidebar, an API profile the coordinator chose from
  your configured ones, and its own private conversation and context. Several
  agents run at once when their work is independent; they share this one
  working directory, so nothing merges or locks their edits.
  The sidebar on the right lists Primary first, then each agent with its tasks
  indented two columns beneath it. Status is a glyph, not a color: ○ idle,
  ● running, ◌ waiting for you, ✓ completed, ! failed.
  Click an agent, or Tab into the sidebar and press Enter on the highlighted
  row, to watch its live stream: its reasoning, replies, commands, and output.
  Arrow keys only move the highlight; Enter or a click commits the choice.
  Clicking a task line selects its agent. A new agent or a status change never
  changes what you are looking at. The status line always describes Primary.
  An agent view is read only: the composer, command menu, and multi-line editor
  are hidden and replaced by "Read only — subagents communicate with their
  dispatcher." Your draft, selection, multi-line text, and queue are untouched
  and come back when you return to Primary. Selecting and copying text in the
  agent's transcript still works. Ctrl+L reports that the view is read only
  instead of clearing, Esc does nothing, and Ctrl+C copies a selection or opens
  the quit confirmation — it never clears the Primary draft you cannot see.
  Results return to the coordinator automatically; you do not relay them.
  Reusing an agent's key sends it a follow-up task in the same conversation and
  adds a task row beneath it; a new key creates a new agent. An agent's label
  and profile are fixed once it exists.
  Agents inherit the same command policy and hard blacklist you set. An
  approval prompt names the agent asking for it ("Run command for Coder?") and
  only one appears at a time, whichever stream you are looking at. Quitting
  stops every agent and kills every command they are running.
  Subagent transcripts live in memory for this run only: they are not restored
  after a restart, although the results the coordinator received are, because
  they are part of its own history.

  AI shell:
  The AI can request shell commands via standalone `[[[jtech_cmd]]]` blocks:
  a `[[[jtech_cmd]]]` line, the raw command on the lines that follow, and a
  `[[[/jtech_cmd]]]` line. Nothing in the command is quoted or escaped, so
  multiline scripts, heredocs, and quotes pass through untouched. Blocks may be
  interleaved with commentary, but each delimiter must sit alone at the first
  column of its own line; wrapped in prose or a Markdown fence, nothing runs.
  Each command is gated:
  a hard blacklist always blocks; then /set cmd_mode decides — ask (prompt
  for each command), auto (allowlisted commands run silently, the rest
  prompt), yolo (everything runs except the blacklist), off (no execution).
  An allowlist entry is a read-only grant: commands that write files (>) or
  embed execution (find -exec) always prompt, even when allowlisted.
  At each prompt: y allow, a always allow (saves a prefix rule to config),
  n decline. Esc kills a command that is running.

Keys:
  /                  Command menu: type /, then ↑/↓ to pick, Enter runs it, Tab completes
  Enter              Submit the input, or the multi-line editor
  Ctrl+J             Insert a newline; the first one opens the multi-line editor
                     and carries the current text across unchanged. The input
                     grows a row per line and scrolls once it reaches its
                     maximum height
  Shift+Enter        The same action as Ctrl+J, but only when your terminal
                     reports it distinctly; otherwise it arrives as Enter and
                     submits, so use Ctrl+J
  Paste              Pasted text containing a line break opens the multi-line
                     editor with every line intact; it never submits by itself
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
