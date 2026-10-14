# CSE Decision Support — dashboard shell

Explainable machine-learning decision-support dashboard for the Colombo Stock
Exchange (S&P SL20 constituents). This is the **presentation layer only** (module
M8 in Chapter 4): it reads precomputed results and runs one lightweight
computation (Monte Carlo stress tests from stored GJR-GARCH parameters).
Training, SHAP, sentiment scoring and VaR estimation happen offline.

> While `DATA_SOURCE = "mock"` every page carries a striped amber
> **DEMONSTRATION DATA — NOT SYSTEM OUTPUT** banner. All numbers are synthetic.

No accounts, no login beyond the shared passcode, no personal data collected or
stored. The only per-visitor state is the stock/date/horizon selection in
Streamlit's in-memory session, which disappears when the tab closes.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit PASSCODE
streamlit run app.py
```

Without a `secrets.toml` the app still runs in mock mode with the development
passcode `cse-demo`. Never rely on that outside your laptop.

Before an evaluation session, run through [`TEST_CHECKLIST.md`](TEST_CHECKLIST.md).

## Project layout

```
app.py                     entrypoint: page config → CSS → passcode gate → warm-up → guarded dispatch
config.py                  the ONLY place settings are resolved (secrets → env → default); cache/retry constants
views/                     one file per page; pages call get_repository() and nothing else
  market_overview.py
  stock_prediction.py
  explainability.py
  risk_stress_test.py
  news_sentiment.py
data/
  repository.py            abstract Repository (every data method) + get_repository() factory
  resilient.py             wrapper the factory returns: retry with backoff, short-TTL cache, typed errors
  errors.py                DataUnavailable (unreachable after retries) and NoData (valid query, no rows)
  mock_repository.py       seeded synthetic data — reproducible across reruns; ?scenario= failure hooks
  postgres_repository.py   stub: same methods, raise NotImplementedError
services/
  stress.py                vectorised Monte Carlo from stored parameters (online-safe)
  perf.py                  page-render and stress-test timers → `cse.perf` log
ui/
  auth.py                  passcode gate
  layout.py                design tokens, CSS (incl. phone breakpoints), banner, disclaimer, sidebar, cards,
                           chart captions, responsive tables
  charts.py                Plotly template (validated colour-blind-safe colorway) + gauge
  states.py                waking / empty / unreachable / error panels, try_fetch(), run_guarded()
.streamlit/
  config.toml              theme; client.showErrorDetails = false
  secrets.toml.example     copy → secrets.toml (git-ignored)
TEST_CHECKLIST.md          manual checks: phone layout, every error state, three-interaction path, targets
```

Pages live in `views/` rather than `pages/` because Streamlit auto-discovers a
`pages/` folder and would bypass the single passcode gate in `app.py`.

## Shared state

The sidebar writes `ticker`, `as_of` and `horizon` to `st.session_state`;
pages read them through `ui.layout.selection()`. An empty-state action can
request a date reset by setting `_reset_as_of`; the sidebar applies it on the
next run (widget state cannot be changed after the widget is drawn).

## Cold start, retries and caching

### What sleeps

| Layer | Sleeps after | Wake time | What the participant sees |
|---|---|---|---|
| Streamlit Community Cloud app | ~12 h without traffic | 30–90 s | Streamlit's own *"Zzz, this app has gone to sleep"* page with a wake-up button. This happens **before** our code runs and cannot be styled from inside the app. Visit the link yourself shortly before a session to avoid it. |
| Neon Postgres compute | 5 min idle | typically 1–5 s, occasionally 10–15 s | Our **"Waking things up…"** card (below). Only in `DATA_SOURCE = "postgres"`; mock mode never sleeps. |

### What the app does about it

1. **Explicit waking state.** `ui.states.warm_up()` runs once per session, right
   after the passcode. It draws a card — *"Our database goes to sleep when nobody
   is using the app. Waking it usually takes 5 to 15 seconds, and you only wait
   once per visit."* — **before** the first query, so a slow wake-up is explained
   rather than silently endured. The card updates on every retry
   (*"attempt 2 of 5, trying again in 2 seconds"*).
2. **Retry with exponential backoff.** Every repository read goes through
   `data.resilient.call_with_retry`: five attempts with waits of 1, 2, 4 and 8
   seconds (`config.RETRY_DELAYS_SECONDS`), ~15 s in total. Connection-type
   errors are retried; `NoData`, `NotImplementedError`, `LookupError`,
   `ValueError` and `TypeError` are not, because retrying a wrong request only
   delays the same answer.
3. **Typed failure, friendly panel.** After the last attempt the wrapper raises
   `DataUnavailable`. `app.py` runs the whole page inside `run_guarded()`, which
   turns that into *"We can't reach the data right now"* with a **Try again**
   button, turns `NoData` into an empty-state panel, and turns anything else
   into *"Something went wrong on our side"*. Tracebacks go to the server log
   only; `client.showErrorDetails = false` in `config.toml` is the safety net
   behind that for anything Streamlit itself raises.

### What is cached, and for how long

| What | Mechanism | Lifetime | Why |
|---|---|---|---|
| The repository object (mock or Postgres engine) | `st.cache_resource` on `get_repository()` | process lifetime | one connection pool per app, not per session |
| Every repository read (`list_stocks`, `get_prediction`, `get_articles`, …) | `st.cache_data` in `data/resilient.py`, keyed on method name + arguments (+ mock scenario) | **10 minutes** (`config.CACHE_TTL_SECONDS`) | results are precomputed offline and refreshed at most daily, so a 10-minute window is safe; with N participants on the same share and date the database is hit once, not N times, and a brief outage is bridged by cached values |
| Monte Carlo stress test | **not cached** | — | 2 × 10,000 paths take ~5 ms; caching would make the measured stress-test time meaningless as evidence |
| Plotly figures, HTML | not cached | — | rebuilt from cached data on each run, well under the 5 s target |

Exceptions are never cached. `st.cache_data` returns copies, so pages may
mutate the frames they receive. To force a refresh press **C** in the app
("Clear cache") or restart the process.

## Error and empty states

Every page degrades to a plain-English panel (`ui.states.empty_state`) instead
of a crash or a blank area. Pages use `try_fetch(repo.method, …)`, which
returns `None` on `NoData`; the page then decides what still makes sense to
show (for example, Explainability keeps the global-importance chart when only
the local explanation is missing).

| Situation | Where | Panel |
|---|---|---|
| No news for the stock in the period | News sentiment | *"No news for {name} in this period"* + *"No articles about … match these filters"* |
| No prediction for the date (weekend, holiday, outside history) | Stock prediction, Explainability | *"No prediction for this date"* with a **Use the latest date** button |
| Risk engine not yet run for the stock | Risk & stress test | *"Risk figures aren't ready for {name}"* |
| Stored VaR missing but parameters present | Risk & stress test | *"No stored risk estimate for this date"* — live estimate still available |
| Database unreachable | any | *"We can't reach the data right now"* + **Try again** |
| Anything unexpected | any | *"Something went wrong on our side"* + **Try again** |

A repository implementation signals "valid query, nothing stored" by raising
`data.errors.NoData` (or returning an empty frame for list-type methods). Any
other exception is treated as transient and retried.

### Rehearsing the failure states without a database (mock mode only)

Append `?scenario=<name>` to any page URL. The parameter is ignored unless
`DATA_SOURCE = "mock"`, and it is part of the cache key so it cannot leak into
normal results.

| `scenario=` | Effect |
|---|---|
| `no_news` | sentiment series and articles come back empty |
| `no_prediction` | `get_prediction`, `get_shap_explanation`, `get_narrative`, `get_block_accuracy` raise `NoData` |
| `no_risk` | `get_risk_parameters`, `get_var_estimate`, `get_kupiec_results` raise `NoData` |
| `db_down` | every call raises a connection error → retries → unreachable panel |
| `cold_start` | the first two calls in any 60 s window fail, the third succeeds → waking card with retry messages |

Sidebar navigation drops query parameters, so type the scenario into the URL of
the page you want to test (e.g. `/risk_stress_test?scenario=no_risk`).

## Accessibility

* **Colour-blind-safe polarity.** Up/positive is teal `#0FA890`, down/negative
  is orange `#D46A12`. The pair was checked with the OKLab-based validator from
  the data-viz skill: ΔE 13.6 under deuteranopia simulation (target ≥ 8), 24.4
  normal vision. The previous teal/rose pair scored 6.6 and was replaced. The
  full chart colorway `[teal, indigo, orange, violet]` passes every adjacent-pair
  check on the card surface `#10141D`.
* **Never colour alone.** Every polarity mark carries a symbol or a word:
  prediction cards say *▲ Up* / *▼ Down*; SHAP bars are labelled *▲ +0.37* /
  *▼ −0.31* with a legend that spells out *pushes towards UP* / *pulls towards
  DOWN*; the sentiment gauge prints *▲ Positive* / *▼ Negative* / *● Neutral*;
  metric tiles say "positive tone" / "negative tone" in text.
* **Captions.** Every chart has an *"About this chart"* caption beneath it that
  describes the shape and quotes the key numbers, so the content is available to
  screen readers and to anyone who finds the chart hard to read.
* **Contrast.** Body text `#EEF2F7` ≥ 15:1, secondary `#B4BDCC` ≥ 9:1, captions
  `#7C8698` ≥ 4.7:1 on every surface (WCAG AA for normal text is 4.5:1). Teal
  and orange text ≥ 4.7:1. Indigo is used for chart lines only (≥ 3:1, the
  non-text threshold). Smallest text is 0.72 rem.
* **"How to read this page"** on Explainability explains the two charts and
  what a SHAP value is without any maths.

## Mobile

Checked at 375 px. The sidebar starts collapsed on phones
(`initial_sidebar_state="auto"`); the questionnaire link is duplicated into the
page header at ≤ 640 px so it stays visible without scrolling. Headline cards
form a 2 × 2 grid, Plotly figures are `autosize` + `responsive`, and every
data grid has a card-list twin (`ui.layout.responsive_table`) that CSS swaps in
at ≤ 640 px, because the grid needs horizontal scrolling on a phone.

## Three-interaction rule

From a fresh tab: ① passcode → Market overview; ② **Stock prediction** in the
sidebar → prediction on screen; ③ **Why this prediction?** link under the price
chart (or **Explainability** in the sidebar) → SHAP explanation and narrative for
the same share and date. Prediction in two interactions, explanation in three.
The navigation structure is unchanged; the in-page link was added so the path
does not depend on the sidebar, which is collapsed on phones.

## Performance instrumentation

`services/perf.py` measures two things and writes them to the `cse.perf`
logger (terminal locally; *Manage app → Logs* on Community Cloud) and to a
per-session ring buffer:

| Measurement | How | Target | Where shown |
|---|---|---|---|
| Page render | from navigation dispatch (`start_page` in `app.py`) to `footer()` — server-side script time, excluding the one-off wake-up and the browser's paint | < 5 s once awake | footer line *"Page built in 310 ms · target under 5 s"* |
| Stress test | `timed("stress")` around the 2 × 10,000-path simulation | < 10 s | caption under the histogram |
| Wake-up | first data touch of a session | informational | log only |

```
2026-09-18 10:21:07 cse.perf kind=page name="Stock prediction" seconds=0.312 target=5.0 status=ok
2026-09-18 10:21:09 cse.perf kind=stress name="JKH 2×10,000 paths" seconds=0.006 target=10.0 status=ok
```

`status=SLOW` marks a target miss. Server-side time is a lower bound on what a
participant experiences; for the report, pair it with a browser measurement
(DevTools → Network → "Load") on the evaluation device.

## Switching to Postgres later

1. Implement each method in `data/postgres_repository.py` with a read-only SQL
   query. Keep the return shapes exactly as documented in `data/repository.py`
   — the pages depend on those column names. Raise `data.errors.NoData` when a
   query is valid but returns no rows; let connection errors propagate (the
   wrapper retries them).
2. Add `sqlalchemy` and `psycopg[binary]` to `requirements.txt`. Create the
   engine with `pool_pre_ping=True` so a connection that died while Neon slept
   is replaced instead of failing the first query.
3. In Streamlit secrets set
   ```toml
   DATA_SOURCE = "postgres"
   DATABASE_URL = "postgresql://readonly_user:...@host/db?sslmode=require"
   ```
4. Redeploy. The demo banner disappears automatically because it is driven by
   the same `DATA_SOURCE` flag as the repository factory. No page file changes.

## Deploying to Streamlit Community Cloud

Push to GitHub (secrets are git-ignored), create the app pointing at `app.py`,
and paste the contents of your `secrets.toml` into *App settings → Secrets*.
Imports are deliberately light (Streamlit, Plotly, pandas, NumPy) to stay
within the free tier's memory; no torch / transformers / xgboost / shap.

## Placeholders to replace

- `QUESTIONNAIRE_URL` in `config.py`
- `PASSCODE` in secrets
- Company names in `mock_repository.py` are labels only; every figure is synthetic.
