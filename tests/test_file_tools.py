"""Unit tests for the file tools."""

from jtech_cli.file_tools import cmd_diff, cmd_read, cmd_write


def test_write_then_read(tmp_path):
    path = tmp_path / "a.txt"
    out = cmd_write(str(path), "one\ntwo\n")
    assert "Wrote" in out
    assert path.read_text() == "one\ntwo\n"

    shown = cmd_read(str(path))
    assert "1 | one" in shown
    assert "2 | two" in shown


def test_read_missing_file(tmp_path):
    assert "No such file" in cmd_read(str(tmp_path / "nope.txt"))


def test_read_range(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\n".join(f"line{i}" for i in range(1, 11)))
    shown = cmd_read(f"{path}:3-5")
    assert "3 | line3" in shown
    assert "5 | line5" in shown
    assert "line1" not in shown


def test_read_single_line(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n")
    shown = cmd_read(f"{path}:2")
    assert "2 | b" in shown
    assert "1 | a" not in shown


def test_read_range_out_of_bounds(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("only\n")
    assert "out of bounds" in cmd_read(f"{path}:5-9")


def test_diff_creates_temp_copy(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("orig\n")
    out = cmd_diff(str(path))
    assert "Temp copy" in out
    assert "diff" in out
