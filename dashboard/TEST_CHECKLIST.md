# Manual test checklist

Run through this before every evaluation session. Everything below works in
mock mode with no database; the failure states are switched on with a
`?scenario=` query parameter (mock mode only, ignored with real data).

Local URL is `http://localhost:8501` unless you changed the port. Passcode is
whatever `PASSCODE` is in your secrets (`cse-demo` if none is set).

Tick each line only when you have **seen** the behaviour, not assumed it.

---

## 1. Phone layout (≈ 380 px wide)

Use Chrome DevTools device toolbar (iPhone SE / 375 px) or a real phone.

| # | Step | Expect | ✓ |
|---|------|--------|---|
| 1.1 | Open the app on a phone | Passcode card fills the width, nothing cut off, button reachable without zooming | ☐ |
| 1.2 | Enter the passcode | Page content is visible immediately. The sidebar does **not** cover the screen | ☐ |
| 1.3 | Look at the top of any page | Striped **DEMONSTRATION DATA** banner is the first thing on screen; **Questionnaire ↗** pill is in the header, visible without scrolling | ☐ |
| 1.4 | Market overview | Four headline cards show as a 2 × 2 grid; both charts fit the width with no sideways scroll; gainers/losers show as card lists, not a scrolling grid | ☐ |
| 1.5 | Tap the **»** control (top-left) | Sidebar slides in; stock, date and horizon controls are usable; **Open evaluation questionnaire** and **Sign out** reachable; tap **‹** or outside to close | ☐ |
| 1.6 | Stock prediction | Cards in a 2 × 2 grid; candlestick chart fits; "Why this prediction?" link visible; accuracy blocks show as card list | ☐ |
| 1.7 | Explainability | "How to read this page" expander opens; both bar charts fit the width; bar labels and ▲/▼ legend readable | ☐ |
| 1.8 | Risk & stress test | Inputs stack vertically; the crisis button is tappable; four VaR cards in 2 × 2; histogram fits; Kupiec rows show as cards | ☐ |
| 1.9 | News sentiment | Chart fits; three metric tiles readable; article list shows as cards with an "Open article ↗" link; up to 20 shown with a note | ☐ |
| 1.10 | Scroll to the bottom of every page | Disclaimer present, followed by "Page built in x.xx s" | ☐ |
| 1.11 | Rotate the phone / resize the window | Charts re-flow to the new width (no fixed-width overflow) | ☐ |

## 2. Cold start

| # | Step | Expect | ✓ |
|---|------|--------|---|
| 2.1 | Open `/?scenario=cold_start` in a **new** browser tab and enter the passcode | A card **"Waking things up…"** with a spinner appears immediately, then changes to *"Still waking up. We will try again in 1 seconds (attempt 2 of 5)…"*, then *"…in 2 seconds (attempt 3 of 5)"*, then the page loads. Total wait ≈ 3 s | ☐ |
| 2.2 | Navigate to another page in the same tab | No waking card again (once per visit) | ☐ |
| 2.3 | Press **C** (clear cache), wait 60 s, reload with the same URL | The waking sequence repeats (the scenario resets every 60 s) | ☐ |
| 2.4 | Real deployment: leave the app idle > 5 min, then open it | Waking card shows during the Neon wake-up; page loads without any red error box | ☐ |
| 2.5 | Real deployment: leave the app idle > 12 h, then open it | Streamlit Community Cloud's own *"Zzz — this app has gone to sleep"* page appears first (outside our control); after clicking to wake it, our waking card handles the database | ☐ |

## 3. Error and empty states

Each row: open the URL, enter the passcode if asked, and read the panel as a
non-technical person would. There must be **no red box, no traceback, no blank
panel**, and the disclaimer must still be at the bottom.

| # | URL | Page | Expect | ✓ |
|---|-----|------|--------|---|
| 3.1 | `/news_sentiment?scenario=no_news` | News sentiment | *"No news for John Keells Holdings in this period"* with advice to widen the window; article panel says no articles match | ☐ |
| 3.2 | `/news_sentiment` → untick all three languages | News sentiment | *"No articles about … match these filters"*, no crash | ☐ |
| 3.3 | `/stock_prediction?scenario=no_prediction` | Stock prediction | *"No prediction for this date"* with a **Use the latest date** button; pressing it clears the panel (remove `?scenario` first, or it stays) | ☐ |
| 3.4 | `/explainability?scenario=no_prediction` | Explainability | Left chart still renders; right side shows *"No prediction to explain for this date"*; no narrative; page footer intact | ☐ |
| 3.5 | `/risk_stress_test?scenario=no_risk` | Risk & stress test | *"Risk figures aren't ready for John Keells Holdings"*; no inputs, no VaR cards, footer intact | ☐ |
| 3.6 | `/?scenario=db_down` in a new tab, passcode | Any | After the waking card retries for ≈ 15 s: *"We can't reach the data right now"* with a **Try again** button. Nothing else on the page | ☐ |
| 3.7 | Press **Try again** (with `?scenario=db_down` still in the URL) | Any | Same panel returns after the retries — no error box | ☐ |
| 3.8 | Remove `?scenario=db_down`, press **Try again** | Any | Page loads normally | ☐ |
| 3.9 | Pick a Saturday or Sunday as the as-of date (real data only) | Stock prediction | *"No prediction for this date"* panel, not an error | ☐ |
| 3.10 | Deliberately break something (e.g. temporarily rename a column in a view) | Any | *"Something went wrong on our side"* panel with **Try again**; the traceback is only in the server log | ☐ |

## 4. Accessibility

| # | Step | Expect | ✓ |
|---|------|--------|---|
| 4.1 | Explainability, right chart | Every bar carries a **▲ +0.xx** or **▼ −0.xx** label; a legend spells out *▲ pushes towards UP* and *▼ pulls towards DOWN* | ☐ |
| 4.2 | Turn on a colour-blindness simulator (Chrome DevTools → Rendering → Emulate vision deficiency → Deuteranopia) on Explainability and Market overview | Teal and orange bars remain distinguishable; nothing depends on colour alone | ☐ |
| 4.3 | Market overview gauge | Number is followed by *▲ Positive* / *▼ Negative* / *● Neutral* in words | ☐ |
| 4.4 | Every chart | An *"About this chart · …"* caption sits under it describing the shape and the key numbers | ☐ |
| 4.5 | Prediction cards | Direction is written as *▲ Up* / *▼ Down*, not colour only | ☐ |
| 4.6 | Zoom the browser to 200 % | Text reflows, no clipped labels in cards or section headers | ☐ |

## 5. Three-interaction path (first-time user)

Start logged out on a fresh tab. Count every tap/click after the page loads.

| # | Interaction | Result | ✓ |
|---|-------------|--------|---|
| 5.1 | ① Type the passcode and press **Enter workspace** | Market overview loads | ☐ |
| 5.2 | ② Click **Stock prediction** in the sidebar | A prediction (▲ Up / ▼ Down, P(up)) for the default share is on screen | ☐ |
| 5.3 | ③ Click **Why this prediction?** under the chart (or **Explainability** in the sidebar) | The SHAP explanation and plain-language narrative for the same share and date are on screen | ☐ |
| 5.4 | Open **How to read this page** | Text is understandable without knowing what SHAP is | ☐ |

Result: prediction in 2 interactions, explanation in 3. On a phone the sidebar
needs one extra tap (**»**) unless the user takes the in-page link in 5.3,
which keeps the count at 3.

## 6. Non-negotiables (check after any layout change)

| # | Check | ✓ |
|---|-------|---|
| 6.1 | Mock mode: the striped **DEMONSTRATION DATA — Not system output** banner is at the top of **all five** pages, desktop and phone | ☐ |
| 6.2 | The disclaimer *"Research prototype for decision support only…"* is at the bottom of all five pages **and** on every empty/error panel page | ☐ |
| 6.3 | Desktop: **Open evaluation questionnaire** is visible in the sidebar without scrolling at 1366 × 768 | ☐ |
| 6.4 | Phone: **Questionnaire ↗** is visible in the page header without scrolling | ☐ |
| 6.5 | The passcode gate is still the only access control; no other login, account or personal-data field exists | ☐ |

## 7. Performance evidence

| # | Step | Expect | ✓ |
|---|------|--------|---|
| 7.1 | Visit every page once; read the footer | *"Page built in x.xx s · target under 5 s"* well under 5 s (typically < 0.5 s in mock mode) | ☐ |
| 7.2 | Risk & stress test: read the caption under the histogram | *"simulation ran in x.xx s (target under 10 s)"* — expect well under 1 s | ☐ |
| 7.3 | Check the server log (terminal, or Community Cloud → Manage app → Logs) | Lines like `cse.perf kind=page name="Stock prediction" seconds=0.312 target=5.0 status=ok` and `kind=stress …`; any `status=SLOW` line is a target miss to report | ☐ |
| 7.4 | Change the stock in the sidebar and revisit a page | Second visit is faster (cached reads); numbers identical | ☐ |
