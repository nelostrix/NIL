"""Portable immutable datasets need only the Python standard library."""
import pytest

from nil.data import snapshot, load_snapshot


def test_snapshot_is_immutable_independent_and_integrity_checked(tmp_path):
    rows = [{"x": 1, "label": "a"}, {"x": 2, "label": "b"}]
    path = snapshot(iter(rows), tmp_path / "v1")
    loaded = load_snapshot(path)
    assert list(loaded) == rows
    loaded[0]["x"] = 99
    assert load_snapshot(path)[0]["x"] == 1
    with pytest.raises(FileExistsError):
        snapshot(rows, path)
    with (path / "records.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="integrity"):
        load_snapshot(path)


@pytest.mark.parametrize('options', [{"max_records": 1}, {"max_bytes": 1}])
def test_snapshot_limits_fail_before_publishing(tmp_path, options):
    with pytest.raises(ValueError, match="exceeds"):
        snapshot([{"x": 1}, {"x": 2}], tmp_path / "data", **options)
    assert not list(tmp_path.iterdir())
