# Part B — LLM solver

Turns failure reports into a candidate `Rule` (or `NoSafeRule`), via an
OpenAI-compatible chat endpoint. See `../CONTEXT.md` for the project-wide
contract and ownership.

`rule_check.py` imports `rules.schema` and `rules.validate` from Part C
(`fleet/rules/`), so this package only works with the repo root on
`PYTHONPATH` (or with `rules` installed, e.g. `pip install -e ../rules`,
once its `requires-python` is compatible with the venv in use — see
`../rules/INTEGRATION.md`). Run it as:

```bash
cd ~/fleet/llm && PYTHONPATH=$HOME/fleet python solver.py
```

Without `PYTHONPATH` (or an installed `rules` package), `rule_check.py`
fails with `ModuleNotFoundError: No module named 'rules'`.

## Files

- `rule_check.py` — `Rule` / `NoSafeRule` pydantic models and `parse_reply()`,
  the local v0 copy of Part C's contract (placeholder limits).
- `try_replies.py` — runs `parse_reply()` against 6 hand-written fake LLM
  replies to show what gets accepted/rejected and why.
- `solver.py` — `solve(failure_reports) -> Rule | NoSafeRule`. Calls an
  OpenAI-compatible `/v1/chat/completions` endpoint, validates the reply with
  `parse_reply()`, and retries transient failures up to 2 times (3 attempts
  total), feeding the rejection error back to the model. Raises
  `RuntimeError` if all attempts fail, or immediately on a permanent client
  error — it never fabricates a `NoSafeRule` the model didn't actually say.
- `fake_server.py` — stdlib-only stub server mimicking the one endpoint vLLM
  exposes, so `solver.py` can be tested without a GPU. Supports env vars to
  force specific failure modes (below).
- `test_solver.py` — regression suite; spawns `fake_server.py` in each
  failure mode as a subprocess, runs `solver.py` against it, checks the
  result. Run with one command (below).

## Running the stub + solver by hand

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
python fake_server.py          # terminal 1, leave running
```

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
python solver.py                # terminal 2
```

## Running the test suite

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
python test_solver.py
```

Starts and tears down its own `fake_server.py` subprocess per case; port 8000
must be free before you run it.

## Env vars

`solver.py`:

| Var | Default | Meaning |
|---|---|---|
| `FLEET_LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint base URL |
| `FLEET_LLM_MODEL` | `stub-model` | model name sent in the request |
| `FLEET_LLM_TIMEOUT_S` | `60` | per-request timeout (seconds) |
| `FLEET_LLM_MAX_TOKENS` | `512` | `max_tokens` sent with each request; doubled (ceiling 2048) after a `finish_reason == "length"` reply |
| `FLEET_LLM_RETRY_SLEEP_S` | `1` | sleep before a retry after a transient failure (set to `0` in tests) |

### Retry policy

| Failure | Behaviour |
|---|---|
| Timeout, connection error | retry (counts as an attempt), sleep `FLEET_LLM_RETRY_SLEEP_S` first |
| HTTP 429, HTTP 5xx | retry (counts as an attempt), same sleep |
| HTTP 400/401/403/404/409/422 (other 4xx) | fail fast, no retry — error includes the status code and first 200 chars of the response body |
| `finish_reason == "length"` | do not feed the truncated text back; retry at most once with `max_tokens` doubled; a second `length` raises immediately naming `finish_reason=length` and the limit used |
| Missing/empty `choices`, or missing `message` | raises `"malformed response: no choices"` internally and retries (counts as an attempt) |
| Empty/missing message content | retries, feeding the error back to the model |
| Reply fails `parse_reply()` validation | retries, feeding the error back to the model |
| `source_report_ids` on the parsed `Rule` not a subset of the input reports' `report_id`s | treated as a validation failure; retries, feeding back which ids were unknown and which are valid |

`fake_server.py` (testing only, all default off):

| Var | Effect |
|---|---|
| `FAKE_ALWAYS_GOOD=1` | every reply is valid (no retry needed) |
| `FAKE_ALWAYS_BAD=1` | every reply is invalid (extra field) |
| `FAKE_HTTP_STATUS=500` | respond with this HTTP status instead of 200 |
| `FAKE_EMPTY_CONTENT=1` | `message.content` is `null` instead of real text |
| `FAKE_SLEEP_S=10` | sleep this many seconds before replying (for timeout testing) |
| `FAKE_SCRIPT=<json>` | a JSON list of per-request step dicts (status, finish_reason, content, no_choices); the last step repeats once the list is exhausted. Takes priority over the flags above. |
| (none set) | request 1 is bad, request 2+ is good — tests retry-recovers |

`fake_server.py` also exposes `GET /__log__`, returning
`{"count": N, "max_tokens": [...]}` for the requests it has received so
far, so tests can assert exactly how many requests were made and with
what `max_tokens`.

## Known limits

- Everything has only been tested against `fake_server.py`, a hand-written
  stub. A real vLLM server has never been hit by this code.
- Real-model behaviour is still untested: markdown-fenced JSON,
  partial/truncated JSON that isn't a clean `finish_reason == "length"` cutoff,
  tool-call responses instead of plain `content`, multi-candidate sampling.
- `fake_server.py` is single-threaded (`http.server.HTTPServer`); under
  `FAKE_SLEEP_S`, retries queue at the OS level instead of being served
  concurrently. This only affects the stub's own behavior, not `solver.py`.
- `requirements.txt` pins `openai`'s and `pydantic`'s runtime dependency
  closure, traced by hand with `pip show`. It has not been installed into a
  fresh empty venv to confirm it's sufficient and self-contained.
- The AMD GPU / vLLM step has not started: no GPU instance exists, ROCm and
  vLLM are both untested.
