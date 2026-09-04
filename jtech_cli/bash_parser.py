"""Bash parsing behind a version-pinned ``bashlex`` compatibility adapter.

``bashlex 0.18`` matches a here-document's terminator line against the raw
delimiter word, skipping the quote-removal step Bash performs first. Every
quoted spelling — ``<<'EOF'``, ``<<"EOF"``, ``<<\\EOF``, ``<<E'O'F``,
``<<$'EOF'``, ``<<$"EOF"``, and the empty ``<<''`` — therefore looks
unterminated to it, and Jtech's fail-closed command gate blocks a command Bash
itself accepts.

The repair lives here rather than in :mod:`jtech_cli.cmd_tools` because it is
dependency compatibility, not command policy: the security boundary keeps
deciding what may run, and this module only makes the parser agree with Bash
about where a heredoc ends. Nothing here rewrites the command, so the AST
positions and source slices policy reads still describe the model's original
text, and an unterminated heredoc still raises.

Agreement with Bash 5.2 is verified rather than assumed: every quoting form and
every byte value was compared against the delimiter Bash names in its own
``here-document ... (wanted `X')`` diagnostic. The only delimiters refused are
those Bash itself mangles while building them — see ``_BASH_RESERVED_BYTES`` —
where refusing keeps the disagreement on the fail-closed side.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import threading
from collections.abc import Callable

import bashlex
import bashlex.heredoc

#: The one ``bashlex`` release whose internals this adapter has been read
#: against. The adapter replaces a private function, so a ranged dependency
#: would let an unreviewed release change that function while Jtech kept
#: applying compatibility behaviour written for the old one.
_SUPPORTED_BASHLEX_VERSION = "0.18"

#: Upstream's parameter list for ``makeheredoc``. The adapter forwards all four
#: positionally, so a rename or reorder upstream must fail loudly rather than
#: silently mis-binding ``killleading`` and breaking ``<<-``.
_EXPECTED_PARAMETERS = ("tokenizer", "redirnode", "lineno", "killleading")

#: Attribute stamped on the installed wrapper. Seeing it means some import of
#: this module already patched the process, which is an idempotent success.
_ADAPTER_MARKER = "__jtech_heredoc_adapter__"

#: Attribute on the installed wrapper holding the callable it wraps.
#: ``importlib.reload`` re-executes this module *into its existing namespace*,
#: so the already-installed wrapper's ``__globals__`` is reset along with it and
#: ``_UPSTREAM_MAKEHEREDOC`` goes back to ``None`` underneath a live wrapper.
#: Recording the upstream on the function object itself survives that, and lets
#: the reload path put the reference back.
_UPSTREAM_ATTRIBUTE = "__jtech_heredoc_upstream__"

#: Characters whose backslash escape is honoured inside double quotes. Bash
#: leaves a backslash before anything else in that state literal.
_DOUBLE_QUOTE_ESCAPES = frozenset({"$", "`", '"', "\\"})

#: Bash's non-numeric ANSI-C escapes for ``$'...'``. The numeric forms
#: (``\\nnn``, ``\\xHH``, ``\\uHHHH``, ``\\UHHHHHHHH``) and ``\\cX`` are scanned
#: separately because they consume a variable number of following characters.
_ANSI_C_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}

_OCTAL_DIGITS = "01234567"
_HEX_DIGITS = "0123456789abcdefABCDEF"

#: Bytes Bash mangles while building a here-document delimiter, found by
#: sweeping every byte value against Bash 5.2's own ``(wanted `X')`` diagnostic:
#:
#: * NUL truncates the rest of the ``$'...'`` run it appears in — ``<<$'A\\x00B'``
#:   wants ``A``, while text outside that run survives;
#: * CTLESC (0x01) and CTLNUL (0x7f) are Bash's internal quoting bytes and get a
#:   second CTLESC prefixed wherever they appear — ``<<$'A\\x01B'`` wants
#:   ``A\\x01\\x01B``, which no ordinary line can contain.
#:
#: An escape producing one of these is refused rather than expanded. Expanding
#: it would make the analyzer disagree with Bash about where the body ends, and
#: refusing keeps that disagreement on the fail-closed side.
_BASH_RESERVED_BYTES = frozenset({"\x00", "\x01", "\x7f"})

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_UPSTREAM_MAKEHEREDOC: Callable[..., str] | None = None


class BashParserCompatibilityError(RuntimeError):
    """The installed bashlex internals do not match Jtech's adapter."""


def _take_digits(word: str, index: int, alphabet: str, limit: int) -> tuple[str, int]:
    """Consume up to ``limit`` characters of ``alphabet``; return them and the index."""
    end = index
    while end < len(word) and end - index < limit and word[end] in alphabet:
        end += 1
    return word[index:end], end


def _read_single_quoted(word: str, index: int) -> tuple[str, int]:
    """Consume a ``'...'`` run starting just after its opening quote.

    Raises:
        ValueError: if the run is never closed.
    """
    end = word.find("'", index)
    if end == -1:
        raise ValueError(f"{word!r} leaves a single-quoted section open")
    return word[index:end], end + 1


def _read_double_quoted(word: str, index: int) -> tuple[str, int]:
    """Consume a ``"..."`` run starting just after its opening quote.

    Also serves ``$"..."``: with no message catalogue loaded Bash's locale
    translation returns the string unchanged, so the two differ only in the
    ``$`` that the caller has already consumed.

    Raises:
        ValueError: if the run is never closed, or ends on a bare backslash.
    """
    text: list[str] = []
    while index < len(word):
        char = word[index]
        index += 1
        if char == '"':
            return "".join(text), index
        if char != "\\":
            text.append(char)
            continue
        if index == len(word):
            raise ValueError(f"{word!r} ends on a dangling backslash")
        following = word[index]
        index += 1
        if following == "\n":
            continue  # Line continuation: neither character survives.
        if following in _DOUBLE_QUOTE_ESCAPES:
            text.append(following)
        else:
            text.append(char)
            text.append(following)
    raise ValueError(f"{word!r} leaves a double-quoted section open")


def _escaped_char(word: str, produced: str) -> str:
    """Return an escape's character, refusing the two Bash re-quotes internally.

    Raises:
        ValueError: if ``produced`` is a byte Bash mangles when it builds the
            delimiter, so no line could match what Bash really compares.
    """
    if produced in _BASH_RESERVED_BYTES:
        raise ValueError(
            f"{word!r} expands to a character Bash re-quotes inside a "
            "here-document delimiter, leaving the heredoc unterminable"
        )
    return produced


def _read_ansi_c_quoted(word: str, index: int) -> tuple[str, int]:
    """Consume a ``$'...'`` run starting just after its opening quote.

    Implements Bash's ANSI-C expansion: the named escapes, octal ``\\nnn``, hex
    ``\\xHH``, Unicode ``\\uHHHH``/``\\UHHHHHHHH``, and control ``\\cX``. A
    backslash before anything else stays literal along with the character it
    precedes, matching Bash.

    Raises:
        ValueError: if the run is never closed, ends on a bare backslash, or
            expands to a character Bash re-quotes in a delimiter.
    """
    text: list[str] = []
    while index < len(word):
        char = word[index]
        index += 1
        if char == "'":
            return "".join(text), index
        if char != "\\":
            text.append(char)
            continue
        if index == len(word):
            raise ValueError(f"{word!r} ends on a dangling backslash")
        following = word[index]
        index += 1
        if following in _ANSI_C_ESCAPES:
            text.append(_escaped_char(word, _ANSI_C_ESCAPES[following]))
        elif following in _OCTAL_DIGITS:
            digits, index = _take_digits(word, index - 1, _OCTAL_DIGITS, 3)
            text.append(_escaped_char(word, chr(int(digits, 8) & 0xFF)))
        elif following in ("x", "u", "U"):
            width = {"x": 2, "u": 4, "U": 8}[following]
            digits, index = _take_digits(word, index, _HEX_DIGITS, width)
            if not digits:
                # No hex digits followed, so Bash leaves the escape literal.
                text.append(char)
                text.append(following)
            else:
                text.append(_escaped_char(word, chr(int(digits, 16))))
        elif following == "c":
            if index == len(word):
                raise ValueError(f"{word!r} ends on a dangling backslash")
            controlled = word[index]
            index += 1
            # Bash accepts ``\c\\`` as an escaped backslash before the control
            # conversion, so the second backslash is consumed here too.
            if controlled == "\\" and index < len(word) and word[index] == "\\":
                index += 1
            text.append(
                _escaped_char(
                    word,
                    "\x7f" if controlled == "?" else chr(ord(controlled) & 0x1F),
                )
            )
        else:
            text.append(char)
            text.append(following)
    raise ValueError(f"{word!r} leaves a single-quoted section open")


def _remove_heredoc_delimiter_quotes(word: str) -> str:
    """Apply Bash quote removal to one heredoc delimiter word.

    Bash strips quoting from the delimiter before comparing it with each line
    of the body, which is why ``<<'EOF'`` ends at a line reading ``EOF``. Every
    quoting construct Bash applies to a delimiter is handled: single quotes,
    double quotes, backslash escapes, ANSI-C ``$'...'``, and locale ``$"..."``.
    No *expansion* is performed — the delimiter is never parameter-expanded,
    command-substituted, arithmetically expanded, filename-expanded, or split.

    An empty result is returned rather than refused: ``<<''`` is valid Bash and
    ends at the first empty line.

    Args:
        word: The raw delimiter exactly as written after ``<<`` or ``<<-``.

    Returns:
        The word Bash would match the terminator line against.

    Raises:
        ValueError: if the word leaves a quote unclosed, ends on a dangling
            backslash, or expands to one of the bytes Bash mangles while
            building a delimiter. Malformed quoting is reported, never guessed
            at, and a delimiter Bash could not use as written is refused rather
            than approximated.
    """
    removed: list[str] = []
    index = 0
    while index < len(word):
        char = word[index]
        following = word[index + 1 : index + 2]
        if char == "'":
            text, index = _read_single_quoted(word, index + 1)
        elif char == '"':
            text, index = _read_double_quoted(word, index + 1)
        elif char == "$" and following == "'":
            text, index = _read_ansi_c_quoted(word, index + 2)
        elif char == "$" and following == '"':
            text, index = _read_double_quoted(word, index + 2)
        elif char == "\\":
            if not following:
                raise ValueError(f"{word!r} ends on a dangling backslash")
            text, index = following, index + 2
        else:
            text, index = char, index + 1
        removed.append(text)
    return "".join(removed)


def _quote_aware_makeheredoc(
    tokenizer: object,
    redirnode: object,
    lineno: int,
    killleading: bool,
) -> str:
    """Run bashlex's matcher with a quote-removed delimiter word.

    The substitution is confined to the redirect node upstream is parsing right
    now and is undone before returning, so the AST keeps the model's original
    quoting and every node position still indexes the original command. Nodes
    are not shared between concurrent parses, so this needs no lock; swapping
    the module-level function per call would instead race across threads.

    Args:
        tokenizer: Upstream's tokenizer, forwarded untouched.
        redirnode: The redirect node whose heredoc body is being gathered.
        lineno: Line number used in upstream's diagnostics.
        killleading: True for ``<<-``, which strips leading tabs from the body
            and from the terminator it matches.

    Returns:
        The heredoc body, exactly as upstream computed it.

    Raises:
        BashParserCompatibilityError: if the delimiter's quoting is malformed,
            or the adapter was somehow invoked before installation completed.
        bashlex.errors.ParsingError: if the heredoc is never terminated.
    """
    if _UPSTREAM_MAKEHEREDOC is None:
        raise BashParserCompatibilityError(
            "the bashlex heredoc adapter ran before its upstream was captured"
        )
    raw_word = redirnode.output.word
    try:
        delimiter = _remove_heredoc_delimiter_quotes(raw_word)
    except ValueError as error:
        raise BashParserCompatibilityError(
            f"heredoc delimiter {raw_word!r} is not valid Bash quoting: {error}"
        ) from error

    redirnode.output.word = delimiter
    try:
        return _UPSTREAM_MAKEHEREDOC(tokenizer, redirnode, lineno, killleading)
    finally:
        redirnode.output.word = raw_word


setattr(_quote_aware_makeheredoc, _ADAPTER_MARKER, True)


def _install_adapter() -> None:
    """Validate bashlex internals and install the adapter exactly once.

    Runs on first parse rather than at import so a version or signature
    mismatch surfaces through :func:`parse_bash`'s documented failure — and
    from there through ``cmd_tools``' existing ``ShellParseError`` boundary —
    instead of making :mod:`jtech_cli.cmd_tools` unimportable.

    Raises:
        BashParserCompatibilityError: if ``bashlex`` is not the pinned version,
            is not installed as a distribution at all, no longer exposes
            ``makeheredoc`` with the signature this adapter forwards to, or has
            an adapter installed that records no upstream to fall back on.
    """
    global _INSTALLED, _UPSTREAM_MAKEHEREDOC
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        try:
            installed_version = importlib.metadata.version("bashlex")
        except importlib.metadata.PackageNotFoundError as error:
            raise BashParserCompatibilityError(
                "bashlex is not installed as a distribution, so Jtech cannot "
                f"confirm it is version {_SUPPORTED_BASHLEX_VERSION}"
            ) from error
        if installed_version != _SUPPORTED_BASHLEX_VERSION:
            raise BashParserCompatibilityError(
                "Jtech's heredoc adapter is written against bashlex "
                f"{_SUPPORTED_BASHLEX_VERSION}, but {installed_version} is "
                "installed; re-pin bashlex or review the adapter against the "
                "new release"
            )

        upstream = getattr(bashlex.heredoc, "makeheredoc", None)
        if not callable(upstream):
            raise BashParserCompatibilityError(
                "bashlex.heredoc.makeheredoc is missing or is not callable"
            )
        if getattr(upstream, _ADAPTER_MARKER, False):
            # An adapter is already installed — a reload of this module, whose
            # namespace that live wrapper reads its upstream from. Recover the
            # reference it recorded rather than capturing the wrapper itself,
            # which would make it call itself on the next heredoc.
            recovered = getattr(upstream, _UPSTREAM_ATTRIBUTE, None)
            if not callable(recovered):
                raise BashParserCompatibilityError(
                    "an adapter is installed on bashlex.heredoc.makeheredoc "
                    "but records no upstream callable to delegate to"
                )
            _UPSTREAM_MAKEHEREDOC = recovered
            _INSTALLED = True
            return

        parameters = tuple(inspect.signature(upstream).parameters)
        if parameters != _EXPECTED_PARAMETERS:
            raise BashParserCompatibilityError(
                f"bashlex.heredoc.makeheredoc takes {parameters}, but Jtech's "
                f"adapter forwards {_EXPECTED_PARAMETERS}"
            )

        _UPSTREAM_MAKEHEREDOC = upstream
        setattr(_quote_aware_makeheredoc, _UPSTREAM_ATTRIBUTE, upstream)
        bashlex.heredoc.makeheredoc = _quote_aware_makeheredoc
        _INSTALLED = True


def parse_bash(command: str) -> list[object]:
    """Parse Bash using Jtech's version-pinned heredoc compatibility layer.

    Args:
        command: The exact command string, never a normalized rewrite of it.

    Returns:
        The ``bashlex`` AST roots, with heredoc delimiters still carrying the
        quoting the model wrote.

    Raises:
        BashParserCompatibilityError: if the installed ``bashlex`` no longer
            matches the internals this adapter patches, or a heredoc delimiter
            carries malformed quoting.
        bashlex.errors.ParsingError: if the command is not parseable Bash,
            which includes a heredoc that is never terminated.
        NotImplementedError: for Bash constructs ``bashlex`` cannot represent.
    """
    _install_adapter()
    return bashlex.parse(command)
