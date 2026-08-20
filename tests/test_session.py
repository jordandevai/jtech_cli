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
