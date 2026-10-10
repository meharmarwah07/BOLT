"""Append-only, hash-chained JSON Lines audit log.

Every event in the system (lifecycle transitions, share decisions, etc.) is
one line: {ts, event, rule_id, actor, details, prev_hash, hash}. hash is the
sha256 of the canonical JSON (sorted keys, no whitespace) of every other
field including prev_hash. The first line's prev_hash is a fixed genesis
value. verify_chain() walks the file and reports the first broken line, if
any.

Only append_event() writes to the file, and only in mode "a". There is no
edit or delete API -- that is the point of an audit log.
"""
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

GENESIS_PREV_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEvent:
    ts: str
    event: str
    rule_id: str
    actor: str
    details: dict
    prev_hash: str
    hash: str


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _compute_hash(ts: str, event: str, rule_id: str, actor: str, details: dict, prev_hash: str) -> str:
    payload = _canonical_json(
        {"ts": ts, "event": event, "rule_id": rule_id, "actor": actor, "details": details, "prev_hash": prev_hash}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_hash(path: Union[str, Path]) -> str:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_PREV_HASH
    last_line = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return GENESIS_PREV_HASH
    return json.loads(last_line)["hash"]


def append_event(path: Union[str, Path], event: str, rule_id: str, actor: str, details: dict | None = None) -> AuditEvent:
    """Append one event to the log at `path`. Creates the file if needed."""
    details = details or {}
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    prev_hash = _last_hash(path)
    hash_ = _compute_hash(ts, event, rule_id, actor, details, prev_hash)
    record = AuditEvent(ts=ts, event=event, rule_id=rule_id, actor=actor, details=details, prev_hash=prev_hash, hash=hash_)

    line = json.dumps(
        {
            "ts": record.ts,
            "event": record.event,
            "rule_id": record.rule_id,
            "actor": record.actor,
            "details": record.details,
            "prev_hash": record.prev_hash,
            "hash": record.hash,
        }
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return record


def read_events(path: Union[str, Path]) -> list[AuditEvent]:
    """Read every event from the log, in order. Does not verify the chain."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events.append(AuditEvent(**d))
    return events


def verify_chain(path: Union[str, Path]):
    """Walk the log and verify the hash chain.

    Returns True if the whole file is intact (or the file is empty/missing).
    Returns the 1-indexed line number of the first broken line otherwise.
    """
    path = Path(path)
    if not path.exists():
        return True

    prev_hash = GENESIS_PREV_HASH
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                return lineno

            try:
                expected_hash = _compute_hash(d["ts"], d["event"], d["rule_id"], d["actor"], d["details"], d["prev_hash"])
            except KeyError:
                return lineno

            if d["prev_hash"] != prev_hash or d["hash"] != expected_hash:
                return lineno

            prev_hash = d["hash"]

    return True
