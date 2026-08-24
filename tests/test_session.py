"""Unit tests for JSONL session persistence."""


from jtech_cli.session import Session


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "session.jsonl"
    s = Session(path)
    s.add("user", "hi")
    s.add("assistant", "hello")
    s.save()

    loaded = Session(path)
    loaded.load()
    assert loaded.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_load_ignores_garbage_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('{"role":"user","content":"ok"}\nnot json\n')
    s = Session(path)
    s.load()
    assert s.messages == [{"role": "user", "content": "ok"}]


def test_load_missing_file_is_empty(tmp_path):
    s = Session(tmp_path / "nope.jsonl")
    s.load()
    assert s.messages == []


def test_clear_removes_file(tmp_path):
    path = tmp_path / "session.jsonl"
    s = Session(path)
    s.add("user", "x")
    s.save()
    s.clear()
    assert not path.exists()
    assert s.messages == []


def test_persist_false_does_not_write(tmp_path):
    path = tmp_path / "session.jsonl"
    s = Session(path, persist=False)
    s.load()
    s.add("user", "x")
    s.save()
    assert not path.exists()
    assert s.messages == [{"role": "user", "content": "x"}]


def test_messages_with_system():
    s = Session()
    s.add("user", "hi")
    with_system = s.messages_with_system("be brief")
    assert with_system == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    # no system prompt -> passthrough, original list unmutated
    assert s.messages_with_system("") == [{"role": "user", "content": "hi"}]
    assert s.messages == [{"role": "user", "content": "hi"}]


def test_runtime_event_can_use_model_observation_role_without_changing_display_role():
    s = Session()
    s.add(
        "system",
        "$ pwd\nexit 0\n/project",
        model_role="user",
        model_content="[JTECH runtime event]\n$ pwd\nexit 0\n/project",
    )

    assert s.messages == [
        {
            "role": "system",
            "content": "$ pwd\nexit 0\n/project",
            "_model_role": "user",
            "_model_content": "[JTECH runtime event]\n$ pwd\nexit 0\n/project",
        }
    ]
    assert s.messages_with_system("") == [
        {
            "role": "user",
            "content": "[JTECH runtime event]\n$ pwd\nexit 0\n/project",
        }
    ]


def test_context_excluded_message_is_persisted_but_not_sent_to_model(tmp_path):
    path = tmp_path / "session.jsonl"
    s = Session(path)
    s.add("system", "Continue your task", include_in_context=False, debug_only=True)
    s.save()

    loaded = Session(path)
    loaded.load()
    assert loaded.messages == [
        {
            "role": "system",
            "content": "Continue your task",
            "_include_in_context": False,
            "_debug_only": True,
        }
    ]
    assert loaded.messages_with_system("") == []


def test_ephemeral_adds_then_strips():
    s = Session(persist=False)
    s.add("user", "hi")
    with s.ephemeral("system", "keep going"):
        assert s.messages[-1] == {"role": "system", "content": "keep going"}
    assert s.messages == [{"role": "user", "content": "hi"}]


def test_ephemeral_strips_when_the_body_raises():
    """A leaked ephemeral would be permanent — the strip has to run on failure."""
    s = Session(persist=False)
    try:
        with s.ephemeral("system", "keep going"):
            raise RuntimeError("stream died")
    except RuntimeError:
        pass
    assert s.messages == []


def test_ephemeral_strips_its_own_copy_not_an_equal_one():
    """Identity, not equality: an equal message from a crashed run must survive."""
    s = Session(persist=False)
    s.add("system", "keep going")  # stale twin, e.g. left by an earlier crash
    with s.ephemeral("system", "keep going"):
        assert len(s.messages) == 2
    assert s.messages == [{"role": "system", "content": "keep going"}]


def test_ephemeral_tolerates_history_cleared_underneath_it():
    """/clear can empty the list mid-block; the strip must not raise."""
    s = Session(persist=False)
    with s.ephemeral("system", "keep going"):
        s.messages.clear()
    assert s.messages == []


def test_clear_leaves_the_file_alone_when_not_persisting(tmp_path):
    """--no-persist promises not to touch stored history, including on /clear."""
    path = tmp_path / "session.jsonl"
    path.write_text('{"role": "user", "content": "precious"}\n')

    s = Session(path, persist=False)
    s.load()
    s.add("user", "throwaway")
    s.clear()

    assert s.messages == []
    assert path.exists()
    assert "precious" in path.read_text()
