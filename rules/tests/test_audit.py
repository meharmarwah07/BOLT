import json

import pytest

from rules.audit import GENESIS_PREV_HASH, append_event, read_events, verify_chain


def test_verify_chain_true_for_missing_file(tmp_path):
    assert verify_chain(tmp_path / "does_not_exist.jsonl") is True


def test_verify_chain_true_for_empty_log(audit_log_path):
    audit_log_path.write_text("", encoding="utf-8")
    assert verify_chain(audit_log_path) is True


def test_append_and_read_round_trip(audit_log_path):
    append_event(audit_log_path, "PROPOSED", "r1", "solver", {"x": 1})
    append_event(audit_log_path, "CHECK1_PASSED", "r1", "sim", {"y": 2})
    events = read_events(audit_log_path)
    assert len(events) == 2
    assert events[0].event == "PROPOSED"
    assert events[1].event == "CHECK1_PASSED"
    assert events[1].prev_hash == events[0].hash


def test_first_event_uses_genesis_prev_hash(audit_log_path):
    append_event(audit_log_path, "PROPOSED", "r1", "solver")
    events = read_events(audit_log_path)
    assert events[0].prev_hash == GENESIS_PREV_HASH


def test_verify_chain_true_for_clean_log(audit_log_path):
    for i in range(5):
        append_event(audit_log_path, "EVENT", f"r{i}", "actor", {"i": i})
    assert verify_chain(audit_log_path) is True


@pytest.mark.parametrize("target_line", [1, 2, 3])
def test_tampering_any_line_breaks_chain_from_that_point(audit_log_path, target_line):
    append_event(audit_log_path, "PROPOSED", "r1", "solver", {"n": "one"})
    append_event(audit_log_path, "CHECK1_PASSED", "r1", "sim", {"n": "two"})
    append_event(audit_log_path, "APPROVED", "r1", "alice", {"n": "three"})

    lines = audit_log_path.read_text(encoding="utf-8").splitlines()
    d = json.loads(lines[target_line - 1])
    d["actor"] = d["actor"] + "_TAMPERED"
    lines[target_line - 1] = json.dumps(d)
    audit_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_chain(audit_log_path)
    assert result == target_line


def test_tampering_hash_itself_is_detected(audit_log_path):
    append_event(audit_log_path, "PROPOSED", "r1", "solver")
    append_event(audit_log_path, "APPROVED", "r1", "alice")

    lines = audit_log_path.read_text(encoding="utf-8").splitlines()
    d = json.loads(lines[0])
    d["hash"] = "0" * 64
    lines[0] = json.dumps(d)
    audit_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert verify_chain(audit_log_path) == 1


def test_no_edit_or_delete_api_exposed():
    import rules.audit as audit_module

    public_names = [n for n in dir(audit_module) if not n.startswith("_")]
    assert "edit_event" not in public_names
    assert "delete_event" not in public_names


def test_only_append_mode_is_used(audit_log_path, monkeypatch):
    original_open = open
    modes_used = []

    def spy_open(path, mode="r", *args, **kwargs):
        if str(path) == str(audit_log_path):
            modes_used.append(mode)
        return original_open(path, mode, *args, **kwargs)

    import rules.audit as audit_module

    monkeypatch.setattr(audit_module, "open", spy_open, raising=False)
    append_event(audit_log_path, "PROPOSED", "r1", "solver")
    assert all(m in ("a", "r") for m in modes_used)
    assert "w" not in modes_used
