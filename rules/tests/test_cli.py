import pytest

from rules.lifecycle import Check1Record, LifecycleError, State, current_state, record_check1, transition
from rules.review_cli import load_rule, main, store_rule


@pytest.fixture
def cli_env(tmp_path, rule_factory, capsys):
    log = tmp_path / "audit.jsonl"
    store = tmp_path / "store"
    rule = rule_factory(rule_id="r_demo")
    store_rule(store, rule)
    return log, store, rule


def _run(log, store, *args):
    return main(["--log", str(log), "--store", str(store), *args])


def test_list_empty_when_no_rules(cli_env, capsys):
    log, store, rule = cli_env
    code = _run(log, store, "list")
    out = capsys.readouterr().out
    assert code == 0
    assert "no rules pending review" in out


def test_list_shows_only_check1_passed_rules(cli_env, capsys):
    log, store, rule = cli_env
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    code = _run(log, store, "list")
    out = capsys.readouterr().out
    assert code == 0
    assert rule.rule_id not in out  # not pending yet, Check 1 hasn't run

    record_check1(log, rule.rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_0001"]))
    code = _run(log, store, "list")
    out = capsys.readouterr().out
    assert rule.rule_id in out


def test_show_prints_explain_sentence_and_check1_numbers(cli_env, capsys):
    log, store, rule = cli_env
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    record_check1(log, rule.rule_id, "sim", True, Check1Record("glass_jar", 5, 1, True, 0, ["fr_0001"]))

    code = _run(log, store, "show", rule.rule_id)
    out = capsys.readouterr().out
    assert code == 0
    assert rule.rule_id in out
    assert "glass_jar" in out
    assert "drops_before=5" in out
    assert "drops_after=1" in out
    assert "crush_count=0" in out
    # never uses the rationale text in the explanation path's template source
    assert rule.rationale not in out.split("Check 1:")[0] or "slip before drop" not in out


def test_approve_requires_reviewer_and_note(cli_env, capsys):
    log, store, rule = cli_env
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    record_check1(log, rule.rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_0001"]))

    with pytest.raises(SystemExit):
        _run(log, store, "approve", rule.rule_id)  # missing --reviewer/--note


def test_approve_then_reject_is_illegal(cli_env, capsys):
    log, store, rule = cli_env
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    record_check1(log, rule.rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_0001"]))

    code = _run(log, store, "approve", rule.rule_id, "--reviewer", "alice", "--note", "ok")
    assert code == 0
    assert current_state(log, rule.rule_id) == State.APPROVED

    code = _run(log, store, "reject", rule.rule_id, "--reviewer", "bob", "--note", "too late")
    err = capsys.readouterr().err
    assert code == 1
    assert "error" in err


def test_reject_logs_event(cli_env, capsys):
    log, store, rule = cli_env
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    record_check1(log, rule.rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_0001"]))

    code = _run(log, store, "reject", rule.rule_id, "--reviewer", "alice", "--note", "unsafe")
    assert code == 0
    assert current_state(log, rule.rule_id) == State.REJECTED


def test_show_unknown_rule_raises(cli_env, capsys):
    log, store, rule = cli_env
    with pytest.raises(FileNotFoundError):
        _run(log, store, "show", "does_not_exist")
