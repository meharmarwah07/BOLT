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

- `prompts/solver_system.txt` — the system prompt template, loaded by
  `solver.py` at import time (path resolved relative to `solver.py`, not the
  working directory). Its hard-limit and crush-limit numbers are
  placeholders like `{grip_min}` / `{crush_limits}`, filled in from
  `rules/limits.py` so the numbers in the prompt can never drift from the
  numbers the real validator enforces. Edit the wording freely; edit the
  placeholder names only if you also update `_load_system_prompt()` in
  `solver.py` to pass a matching keyword.
- `prompts/examples.json` — 3 synthetic few-shot examples (two valid Rules,
  one `no_safe_rule`), sent to the model as user/assistant message pairs
  before the real request. Each Rule example is checked by the test suite
  against the real validator (`rule_check.parse_reply`), so if you edit an
  example's numbers, make sure it still respects `rules/limits.py`'s ranges
  and crush limits. The object classes and numbers here are deliberately
  different from the `glass_jar` case in `CONTEXT.md`, so the examples
  teach the output format without anchoring the model on that answer.
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
- `fixtures/synthetic_reports.json` — 18 hand-made FailureReports (6 each
  for `steel`, `cardboard`, `glass_jar`), used by `smoke_real.py`. Synthetic,
  not simulator output; distinct from the few-shot examples in `prompts/`.
- `fixtures/truth.json` — hidden ground truth for the fixtures above (a
  `hidden_mu` per report plus a derived `feasible` / `min_grip_at_accel_floor`
  verdict). **Never sent to the model.** `smoke_real.py` reads it only after
  the fact, to score a reply.
- `smoke_real.py` — calls `solver.solve()` once per fixture report (see
  below) and scores the result against `fixtures/truth.json`. Prints a
  per-class and overall table with Wilson confidence intervals, and writes
  every per-call result to `runs/<timestamp>.json`.
- `stats.py` — `wilson(k, n, z=1.96) -> (low, high)`, a from-scratch Wilson
  score interval, used by `smoke_real.py` and nothing else yet.
- `test_smoke.py` / `test_stats.py` — regression suites for the two files
  above, same PASS/FAIL style as `test_solver.py`.

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

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
python test_smoke.py
python test_stats.py
```

## Running the smoke script (`smoke_real.py`)

Against the local stub — a plumbing check only, costs nothing, proves
`solve()` runs end to end and `runs/` gets written:

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
python fake_server.py          # terminal 1, leave running
```

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
python smoke_real.py --repeats 1   # terminal 2
```

Against a real model — this costs GPU time, so only run it after a clean
stub run above. Point it at the real server with the same env vars
`solver.py` reads, then run the same script unchanged:

```bash
cd ~/fleet/llm && source ~/fleet/.venv/bin/activate
FLEET_LLM_BASE_URL=http://<host>:<port>/v1 FLEET_LLM_MODEL=<model-name> \
    python smoke_real.py --repeats 3
```

`--repeats N` calls `solve()` N times per fixture report (default 1), useful
against a real, non-deterministic model; it's a no-op against the stub,
which always answers the same way.

`smoke_real.py` calls `solve()` sequentially (concurrency 1) and enforces a
minimum gap between request starts, via `--min-interval-s` or
`FLEET_MIN_INTERVAL_S` (default `4.0`, i.e. ≤15 requests/minute). Set it to
`0` when running against the local stub to skip the wait.

Each `runs/<timestamp>.json` also records `model`, `base_url` (never the
key), `utc_date`, `reasoning_effort`, `min_interval_s`, `http_requests_made`
(actual HTTP requests across the whole run, including retries),
`http_429_count`, and `finish_reason_counts`.

**What the numbers mean.** `smoke_real.py` prints, per object class and
overall:
- **valid-JSON rate** — fraction of calls that didn't end in `error` (i.e.
  `solve()` returned *something* parseable, a `Rule` or a `no_safe_rule`).
- **effective-rule rate among feasible cases** — of the fixture reports
  where a safe rule actually exists (`truth.json`'s `feasible: true`), the
  fraction where `solve()` both returned a `Rule` and that `Rule`'s numbers
  would prevent the drop under the fixture's **hidden** `mu`.
- **correct-refusal rate among infeasible cases** — of the reports where no
  safe rule exists, the fraction where `solve()` correctly said `no_safe_rule`.
- **wrong-refusal rate among feasible cases** — of the reports where a safe
  rule *does* exist, the fraction where `solve()` wrongly gave up anyway.

Each rate comes with a 95% Wilson interval from `stats.py`.

**These are all an ANALYTIC stand-in, not a simulator measurement.**
"Effective" and "feasible" come from the formula `N >= m*(g+a)/(2*mu)`
against a hand-picked hidden `mu` in `fixtures/truth.json`, not from a
MuJoCo replay. Treat every number here as "does the model's answer agree
with the formula", not as a measured drop rate — the real measurement is
Part A's held-out-seed replay (CONTEXT.md section 8).

## Env vars

`solver.py`:

| Var | Default | Meaning |
|---|---|---|
| `FLEET_LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint base URL |
| `FLEET_LLM_MODEL` | `stub-model` | model name sent in the request |
| `FLEET_LLM_API_KEY` | (falls back to `RADEON_API_KEY`, then `"not-needed"`) | API key passed to the OpenAI client; never printed, logged, or written to `runs/*.json` |
| `FLEET_LLM_REASONING_EFFORT` | unset | if set, sent as `reasoning_effort` on every request; omitted entirely otherwise (some endpoints silently drop it anyway) |
| `FLEET_LLM_TIMEOUT_S` | `60` | per-request timeout (seconds) |
| `FLEET_LLM_MAX_TOKENS` | `512` | `max_tokens` sent with each request; doubled (ceiling 2048) after a `finish_reason == "length"` reply |
| `FLEET_LLM_RETRY_SLEEP_S` | `1` | sleep before a retry after a non-429 transient failure (set to `0` in tests) |

The OpenAI client is constructed with `max_retries=0` so the SDK never adds
hidden retries on top of `solve()`'s own retry loop.

### Retry policy

| Failure | Behaviour |
|---|---|
| Timeout, connection error | retry (counts as an attempt), sleep `FLEET_LLM_RETRY_SLEEP_S` first |
| HTTP 429 | retry (counts as an attempt), sleep the response's `Retry-After` header (capped at 90s, default 60s if missing/unparseable) plus up to 0.5s jitter |
| HTTP 5xx | retry (counts as an attempt), sleep `FLEET_LLM_RETRY_SLEEP_S` first |
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
| `FAKE_SCRIPT=<json>` | a JSON list of per-request step dicts (status, retry_after, finish_reason, content, no_choices); the last step repeats once the list is exhausted. Takes priority over the flags above. |
| (none set) | request 1 is bad, request 2+ is good — tests retry-recovers |

A `FAKE_SCRIPT` step with `"status": 429` may also set `"retry_after": N` to
send a `Retry-After: N` header on that response (also sent, with a fixed
default of 60, when `FAKE_HTTP_STATUS=429` is used instead).

`fake_server.py` also exposes `GET /__log__`, returning
`{"count": N, "max_tokens": [...], "messages": [...], "reasoning_effort": [...]}`
for the requests it has received so far, so tests can assert exactly how
many requests were made, with what `max_tokens`, what message content
(e.g. that the system prompt and few-shot examples actually reached the
server), and whether `reasoning_effort` was sent.

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
