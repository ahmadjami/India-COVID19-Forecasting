# Multi-Market Equity Short-Horizon Signal — Solution Design Document

**Owner:** Data Science — Equity Execution
**Status:** Draft v0 (initial framework)
**Scope:** Tick-level quote & trade data across Hong Kong, India, Shanghai, Shenzhen
**Goal:** A symbol-agnostic regression model predicting forward short-horizon returns (5 / 30 / 60 min) for execution signal generation.

---

## 0. How to read this document

This is a design doc, not a final spec. Each section states **what we do**, **why**, the **challenges**, and **proposed ideas / future upgrades**. Decisions already locked for v0 are marked **[v0 LOCKED]**. Open decisions are marked **[OPEN]**. The single biggest theme throughout is **leakage safety** — most of the "challenge" notes are really "here's where you can accidentally see the future."

---

## 1. Business context & objectives

### 1.1 What the business wants
The execution desk needs a **signal that says whether price is about to move enough to matter** over a short horizon, so execution decisions (timing, aggressiveness, whether to cross the spread) can be informed by a forward-looking estimate rather than reacting after the fact.

### 1.2 Objectives (concrete)
- **This is a regression problem** — predict a continuous **forward return** of an instrument over three horizons: **5 min, 30 min, 60 min**.
- Predict a **change**, never a price level. (Predicting the level is trivially autocorrelated and useless.)
- Be **symbol-agnostic and market-agnostic**: one pooled model that works across all instruments and all four markets, without learning "this is the 563-HKD stock."
- Be **leakage-safe**: every statistic causal (uses only past-of-now), so backtest performance is honest.
- Be **scalable**: adding markets or stocks should require *no code change*, only data.

### 1.3 What "success" looks like
- A working end-to-end framework (clean → bar → feature → label → model → evaluate).
- Evaluation on **rank/sign quality** (information coefficient, directional accuracy), not just error magnitude — because an execution signal is about *ordering and direction*, not exact basis points.

### 1.4 Non-goals (for v0)
- Live latency simulation / production trading loop.
- Multi-day regime validation (we currently have ~1 day of data — see §3.4).
- Order-book depth modeling beyond L1 (data is top-of-book only).

---

## 2. Data & variables

### 2.1 Sources
Two tables, each merged across all four markets:

| Table | Granularity | Approx size | Key columns |
|---|---|---|---|
| `df_trade` | one row per trade print | ~hundreds of k rows | date, sym, time, localtime, price, size, cond, source_file |

> **`cond`** is the **continuous-trade indicator** — it flags whether a trade belongs to the continuous session (vs auction/off-book). It arrives empty/NaN here because **the business already pre-filtered the feed to continuous trades only** (see §2.5), so the filtering `cond` would have done is already applied upstream. Its NaN state is therefore *expected and harmless*, not a data-quality gap.
| `df_quote` | one row per quote update | ~hundreds of k rows | date, sym, time, markettime, bid, ask, bsize, asize, source_file |

After cleaning, both carry: `market`, tz-aware `*_local` and `*_utc` timestamps, and (next step) a `session` label.

### 2.2 Market microstructure — *the markets are NOT the same*
This is the most important fact in the whole project. The four feeds have different data-generating processes:

| Market | Quotes | Trades | Lunch break | Timezone |
|---|---|---|---|---|
| Hong Kong | continuous | continuous | 12:00–13:00 | UTC+8 |
| Shanghai | 3-sec snapshots | continuous | 11:30–13:00 | UTC+8 |
| Shenzhen | 3-sec snapshots | **conflated** | 11:30–13:00 | UTC+8 |
| India | 1-sec snapshots | **conflated** | none (single session) | UTC+5:30 |

**Conflation / snapshot** means the feed already aggregated events upstream. Consequence: a "tick" in Hong Kong ≠ a "tick" in Shenzhen. This single fact drives the bar-type choice (§7).

### 2.3 Timestamp semantics
- **Event time** = when it happened at the exchange: `markettime` (quotes), `localtime` (trades). **Use these.**
- **Capture time** = when our system recorded it: `time` (both tables). Ignore for research; only relevant for live-latency simulation.
- In the current dataset, event and capture times are **identical** — no measurable latency at this resolution — so the choice is moot; we standardize on event time for correctness.

### 2.4 Known data-quality issues
- **HK timestamps were mangled** (hour dropped, showed `30:00.2`). **Fixed** in cleaning step.
- **`date` mixed format** (`3/17/2026` vs `2026-03-17`). **Fixed.**
- **`cond` is empty/NaN — by design, not a defect.** `cond` is the continuous-trade flag; it's blank because the business already delivered **continuous-trade-only** data, so no further filtering by `cond` is needed. We do **not** treat this as a quality issue.
- **Quotes are L1 only** — single bid/ask/bsize/asize. No depth features possible.

### 2.5 Pre-cleaning already applied by business
- **Trades:** delivered as **continuous-session trades only** — opening/closing auctions and lunch excluded. The `cond` (continuous-trade) flag's job is therefore already done upstream; we don't filter trades ourselves.
- **Quotes:** auctions and lunch excluded.
- Implication: no auction/condition filtering needed on our side, **but the lunch *time gap* still exists** and must be respected in bars and labels (§6, §9).

---

## 3. Solution design (high level)

### 3.1 Two-level principle (the backbone of the whole design)
Everything in the pipeline lives at one of two levels. Keeping them straight is what makes the solution both correct and scalable.

| Level | What lives here | Treatment | Model sees it? |
|---|---|---|---|
| **Market-aware** | timezone, session hours, lunch, tick rules, conflation | kept *different* per market, on purpose | No |
| **Symbol/market-agnostic** | features & labels (dimensionless, normalized) | made *comparable* across all instruments | Yes (only this) |

Misconception to avoid: making things "agnostic" does **not** mean flattening timestamps. Timestamps stay market-aware. Agnosticism applies only to the features and labels the model trains on.

### 3.2 Pipeline flow
```
RAW (all markets merged)
  │
  ├─ derive `market` from sym suffix (.HK/.SZ/.SS/India)
  ├─ clean timestamps (tz-aware, fix HK hour, unify date)
  ├─ assign `session` per market on LOCAL time   ◄── current step
  │
  ├─ per (symbol): compute volume-bar threshold
  ├─ build VOLUME bars per (symbol, date, session)   ◄── boundary-safe by construction
  ├─ as-of join quotes → bars (microprice at bar close)
  │
  ├─ FEATURES (causal, dimensionless / self-standardized)   ── symbol-agnostic
  ├─ LABELS (forward microprice return, bps, vol-normalized) ── per horizon, session-masked
  │
  ├─ concat all symbols → ONE pooled table
  └─ per horizon: walk-forward CV (purge+embargo) → CatBoost regressor → IC/DA eval
```

### 3.3 Why one pooled model (not per-symbol)
Dimensionless features make instruments comparable, so pooling gives the model far more data and better generalization than thousands of thin per-symbol models. Symbol/market identity is **excluded** as a feature so the model can't memorize instruments.

### 3.4 Critical constraint: data volume **[OPEN]**
We currently have ~**1 trading day**. This means:
- Causal *cross-day* statistics (trailing ADV, multi-day vol) **cannot** be computed — there's no prior day to look back on.
- Train/test split is *within* one session's evolution → weak test of generalization.
- **v0 stance:** build the framework, treat bar thresholds as in-sample (flagged non-causal), keep features/labels causal *within* the day. Do **not** read a good v0 score as proof of edge. **Action: acquire multiple days before trusting results.**

---

## 4. EDA (exploratory data analysis)

### 4.1 What to check, and why
| Check | Why it matters |
|---|---|
| `sym.value_counts()` per table | confirm intended symbol set; catch silent drops in merge |
| rows per `(market, session)` | confirm session assignment is sane; spot empty/oversized sessions |
| min/max timestamp per `(market, session)` | instantly catches timezone mistakes |
| trades/day per symbol | sets the bar-count target (§7.3) |
| spread distribution per symbol (bps) | sanity on quote quality; detect crossed/locked books |
| bid/ask sanity (ask ≥ bid, sizes > 0) | catch bad quote rows before they poison microprice |
| inter-trade time gaps | visualize the lunch gap; confirm no within-session holes |
| price/size distributions per symbol | confirm scale differences (motivates agnostic normalization) |

### 4.2 Challenges in EDA
- **Mixed scales** make naive cross-symbol plots meaningless — always plot per symbol or in dimensionless units.
- **Conflated markets** look "quieter" by tick count but aren't — don't infer activity from row counts across markets.
- **One day** limits distributional conclusions; treat EDA findings as provisional.

### 4.3 Proposed ideas
- A small automated **data-quality report** (per symbol: row count, spread stats, % crossed quotes, session coverage) that reruns whenever new data lands — turns EDA into a repeatable gate, not a one-off.

---

## 5. Preprocessing

### 5.1 Steps **[v0 LOCKED unless noted]**
1. Derive `market` from `sym` suffix (rule-based, so new markets need only a mapping entry).
2. Fix HK timestamps; unify `date`; build tz-aware `*_local` + `*_utc`.
3. Standardize on **event time** (`markettime`/`localtime`).
4. **Assign `session`** on **local** time, per-market schedule. India = single session (`session_1`); HK/China = `session_1` + `session_2`. *(Naming: avoid "morning/afternoon" — it misleads for India. Use `session_1/2` or `pre_lunch/post_lunch`.)*
5. Drop/repair bad quote rows (crossed books, non-positive sizes).

### 5.2 Challenges
- **Session boundary off-by-one:** does 11:30:00.000 belong to session_1 or the gap? Use inclusive comparisons on real `time` objects, not string compares (string compares are fragile and caused an India mislabel bug).
- **India mislabeling:** because India has no `session_2` rule, it can *never* be labeled session_2. If it is, a cross-market rule leaked — a correctness red flag.
- **Timezone errors** are silent and catastrophic — always verify min/max time per (market, session) against the schedule.

### 5.3 Proposed ideas
- Encode schedules in a **single config dict** (already done) so markets/holidays are data, not code.
- Add a `n_sessions` column (India=1, others=2) for self-documenting downstream logic.

---

## 6. Bar construction

### 6.1 What a bar is
A bar bundles consecutive raw events into one summary row (open/high/low/close/volume/vwap + timestamps). The only thing that differs between bar *types* is the rule for where one bar ends and the next begins.

### 6.2 Bar-type decision **[v0 LOCKED: volume bars]**
| Type | Closes when… | Verdict for us |
|---|---|---|
| Time | clock interval elapses | oversamples quiet periods; fat-tailed returns. Baseline only. |
| Tick | N events occur | **avoid across markets** — conflation makes a "tick" mean different things per venue |
| **Volume** | N shares trade | **v0 choice** — conflation preserves *size*, so robust to the snapshot/conflation gap |
| Dollar | N value (price×size) trades | best long-term (price-level & cross-symbol robust); 1-line upgrade from volume bars |

Rationale: volume bars neutralize the conflation problem (the central data quirk) with the least complexity. Dollar bars are the planned upgrade (swap `size` → `price*size` in the accumulator).

### 6.3 Threshold **[v0: in-sample, flagged non-causal]**
- Per **symbol**, set so each symbol yields ~N bars/session (comparable sampling rate across instruments).
- v0: `threshold = symbol_total_volume / TARGET_BARS` (in-sample — acceptable for *bar cutting*, which doesn't define the target; flag clearly).
- Production: trailing (causal) ADV estimate, fallback to market-median for cold-start/thin symbols.

### 6.4 Boundary safety **[v0 LOCKED]**
Build bars per `(symbol, date, session)` group. Because each session is a separate group, **a bar physically cannot span the lunch gap or the day boundary** — the reset is free. Drop the incomplete final bar of each group (partial volume scale would distort the bar population).

### 6.5 Challenges
- **Threshold sizing:** too few bars/session → a 5-min label barely covers one bar (no signal); too many → noise. Target so median bar duration ≪ 5 min. Verify empirically.
- **In-sample threshold leakage** is tolerable for cutting but must not creep into features/labels.
- **Thin symbols** can't form stable thresholds → need a fallback rule.

### 6.6 Proposed ideas
- Track `sum(price*size)` during accumulation → free VWAP *and* instant path to dollar bars.
- **Imbalance/run bars** (advanced): close bars on order-flow imbalance rather than raw volume — sample even more adaptively to information. Future experiment.

---

## 7. Feature engineering

### 7.1 Principle: dimensionless first, standardize second
The model must never see scale or identity. Two ways to kill scale:

1. **Structural ratios (preferred, zero leakage risk):** OBI `(bsize−asize)/(bsize+asize)`, relative spread `(ask−bid)/mid` in bps, microprice offset `/spread`, log returns. Already scale-free — a 0.3 OBI is 0.3 everywhere.
2. **Causal self-standardization (for non-ratio quantities like size/volume):** per-symbol *trailing* z-score, rolling **rank/percentile** (outlier-proof — favored for fat-tailed microstructure data), or divide by a domain scale (size ÷ rolling median size → "multiples of normal").

### 7.2 Candidate features (all L1-derivable)
- **Quote-based:** mid, microprice, microprice offset (spread-normalized), relative spread (bps), top-of-book OBI, depth, short rolling deltas.
- **Trade-based:** trade sign (tick rule / Lee-Ready — must be *inferred*, needs clean timestamps), signed volume / order-flow imbalance, trade intensity, realized vol, VWAP−mid.

### 7.3 The mechanism that guarantees correctness
`groupby('sym').transform(rolling(...))` — guarantees **per-symbol** (a stock is only normalized against its own history) **and causal** (trailing window, never sees its own future) in one expression. Fans out to thousands of symbols with no code change.

### 7.4 Challenges
- **Leakage via full-sample stats:** z-scoring with full-day mean/std leaks the future. Always trailing.
- **Double-normalizing:** don't z-score a feature that's already a ratio (OBI) — adds noise.
- **Cold start:** first `min_periods` bars per symbol are NaN → dropped. With one day, this costs a meaningful fraction of data.
- **Cross-market microstructure:** spread in *bps* (poolable) vs *ticks* (better microstructure signal) is an **[OPEN]** choice with downstream ripple effects (tick-size differs: China flat 0.01 vs HK price-banded).
- **Trade-sign inference** is approximate and timestamp-sensitive.

### 7.5 Proposed ideas
- **Liquidity feature** that is itself dimensionless (trailing turnover percentile) — lets one pooled model adapt to liquidity tiers without learning identity.
- **Cross-sectional normalization** (rank a feature across the universe at the same instant) — powerful for alpha, but needs simultaneous broad universe; awkward across 4 timezones. Park until universe grows; viable within a market during overlapping hours.
- **Quantile/rank-Gaussian mapping** if pooled marginals still misalign — heavy-handed, fit causally, use sparingly.

---

## 8. Target preparation

### 8.1 Principle: predict a normalized *change*
Target chain: **mid change → microprice change → in bps → ÷ trailing vol (sigma units)**. Each step buys cross-symbol comparability; each is optional but recommended.

- **Microprice** `(bid·asize + ask·bsize)/(bsize+asize)` beats raw mid for short horizons (uses the size imbalance you already have).
- **bps / log** makes the move scale-free (a 0.5 move means different things on a 563 vs a 21 stock).
- **÷ trailing realized vol** → "sigma units": the same label value means the same economic event on every instrument. Vol window should roughly match the horizon scale.

### 8.2 Problem type **[v0 LOCKED: regression]**
**This is a regression problem.** We predict the continuous normalized forward return directly.
- **Regression (chosen):** predict the continuous normalized forward return per horizon. This is the committed approach for v0 and beyond.
- **Classification (triple-barrier) — noted alternative only:** down/flat/up via ±k·σ barriers + vertical (time) barrier, spread-normalized. Recorded here for completeness as a possible future comparison; **not** the current problem framing.

### 8.3 Three horizons → three models **[v0 LOCKED]**
- One feature matrix `X`, three label columns `y_5/y_30/y_60`. Build `X` once; only the label changes.
- Each horizon is a different signal-to-noise problem → separate specialized models beat one shared multi-output model.

### 8.4 Session-boundary label masking **[v0 LOCKED — critical]**
Compute the forward label **within the same `(sym, date, session)` group**. If looking *h* minutes ahead runs past the session's last bar, the label is **NaN and dropped**. This single rule prevents a 30-min label at 11:15 from secretly becoming a 2-hour return across lunch. Grouped application makes it automatic — the search literally cannot see the next session.

### 8.5 Challenges
- **Overlapping labels:** consecutive bars' 60-min labels share ~59 min of future → heavy serial correlation → inflated apparent performance. Mitigate with **uniqueness/sample weighting** and per-horizon embargo (§9).
- **Sample scarcity at long horizons:** 60-min labels near close/lunch are invalid → dropped. Model C (60-min) trains on fewer rows and is structurally the weakest — expected, not a bug.
- **Vol-window/horizon mismatch:** normalizing all three horizons by one vol window makes "sigma units" inconsistent across models. Match window to horizon.

---

## 9. Modelling

### 9.1 Model **[v0 LOCKED: CatBoost regressor, pooled]**
- Gradient-boosted trees handle mixed-scale, non-linear microstructure features well, are robust, and (if ever needed) handle categoricals natively.
- **Pooled** across all symbols/markets. **Excluded features:** `sym`, `market`, `source_file`, raw price, raw size, and anything carrying identity or absolute scale.

### 9.2 Validation: walk-forward with purge + embargo **[v0 LOCKED — critical]**
- **Time-ordered** split (never random — random shuffles future into train).
- **Embargo sized to the horizon:** ≥5 / 30 / 60 min respectively. A label at *t* peeks *h* ahead, so any train row within *h* of a validation row shares information → leakage. The 60-min model needs a 60-min embargo; using one embargo for all three silently leaks on the long ones.
- **Purge** overlapping windows around the split.

### 9.3 Challenges
- **One-day data** → walk-forward is *within* one session's evolution; weak generalization test (§3.4).
- **Embargo eats data**, especially at 60 min — compounds long-horizon scarcity.
- **Label overlap** still inflates training signal even with embargo → sample weighting.
- **Regime dependence:** a model fit on one day may not transfer to another volatility regime.

### 9.4 Proposed ideas
- **Sample weights** by label uniqueness (down-weight overlapping forward windows) — from triple-barrier theory, applies to regression too.
- **Liquidity-tiered models** *only if* pooled performance is dragged by the illiquid tail (start pooled).
- **Multi-task / shared-trunk** models across horizons as a later experiment (v0 keeps them separate).

---

## 10. Model evaluation

### 10.1 Metrics **[v0 LOCKED: rank/sign first]**
- **Information Coefficient (IC):** Spearman correlation of prediction vs realized return — measures *ordering* quality (the thing that matters for a signal).
- **Directional accuracy:** hit-rate on sign.
- RMSE/MAE as secondary fitting diagnostics — but RMSE can look identical for a useless and a useful model if the useful one just gets ordering right.

### 10.2 How to evaluate honestly
- Always on the **embargoed out-of-sample** set.
- Report per-horizon and (diagnostically) per-market/per-liquidity-tier to see *where* signal lives — without letting identity into the model.
- Compare against a **naive baseline** (e.g. predict 0, or last-return persistence) — beating zero is the real bar.

### 10.3 Challenges
- **Leakage inflates every metric** — if IC looks "too good," suspect leakage first.
- **One-day evaluation** is not proof of edge; multi-day, multi-regime is required before any trust.
- **Overlapping labels** inflate IC; uniqueness weighting and proper embargo are the guardrails.

### 10.4 Proposed ideas
- A **purged, embargoed cross-validation** harness (reusable) so every experiment is evaluated identically.
- Track **IC stability over time** (rolling IC), not just a single number — a stable small IC beats a spiky large one for execution.

---

## 11. Scalability & engineering

### 11.1 Why this design scales to many markets & thousands of stocks
- **Per-level logic, not per-entity code:** market logic keys on a derived `market` column; scale/causal logic keys on `sym`. `groupby` fans both out automatically.
- Adding a market = one schedule entry + one suffix-mapping rule. Adding stocks = zero code change.
- Schedules/tick rules live in **config dicts** (data, not code).

### 11.2 Engineering challenges
- **Memory/throughput** at full universe — `groupby.apply` with Python loops (bar builder) is the bottleneck; vectorize or use chunked/parallel processing as volume grows.
- **Cold-start governance:** minimum-history rule before a symbol enters training; market-median fallback for thresholds/vol.
- **Reproducibility:** pin the cleaning → bar → feature → label config so results are re-runnable as data grows.

### 11.3 Proposed ideas
- Modular SDK structure (`x_prep` / `y_prep` + thin orchestrator; leakage-safe `(data, meta)` returns) so feature/label logic is reusable and testable.
- Checkpoint long stages (bar building) for restartable runs at scale.
- Automated data-quality gate (§4.3) before any model run.

---

## 12. Risk register (cross-cutting)

| Risk | Where it bites | Mitigation |
|---|---|---|
| **Future leakage** | features, labels, thresholds, CV split | trailing stats only; per-horizon embargo; flag any in-sample step |
| **Label crosses lunch/close** | target prep | session-grouped forward search; NaN-and-drop past session end |
| **Conflation distorts activity** | bar type, tick features | volume/dollar bars; avoid tick bars across markets |
| **Scale/identity leaks into model** | feature set | exclude sym/market/source/raw price&size; dimensionless features |
| **One-day overfit** | everything | acquire multi-day data; treat v0 scores as plumbing checks only |
| **India mislabeled as 2-session** | session assignment | India has only `session_1` defined; verify no `session_2` |
| **Overlapping labels inflate metrics** | long-horizon eval | uniqueness sample weights + embargo |

---

## 13. Status & next steps

**Done:** raw merge; `market` derivation; timestamp cleaning (HK hour fix, tz-aware local+UTC); event-vs-capture time understood (identical in this data); confirmed business delivered **continuous-trade-only** data (so `cond`/auction filtering is handled upstream — no action needed); problem framed as **regression**.

**Current step:** assign `session` on local time (India single, HK/China dual); verify counts and boundaries.

**Immediate next:**
1. Verify session assignment (India = 1 session, others = 2; min/max time per session matches schedule).
2. Per-symbol volume-bar threshold (v0 in-sample, flagged) + pick `TARGET_BARS` so median bar duration ≪ 5 min.
3. Build volume bars per `(sym, date, session)`; drop partial final bars; track VWAP.
4. As-of join quotes → bars (microprice at bar close).
5. Feature engineering (dimensionless + causal standardization).
6. Label construction (forward microprice return, bps, vol-normalized) + session masking, ×3 horizons.
7. Walk-forward CV (per-horizon embargo) → CatBoost ×3 → IC / directional accuracy.

**Before trusting any result:** acquire **multiple trading days**.

---

*This document captures the v0 framework and the reasoning behind each decision. Update [OPEN] items as data and experiments resolve them.*

---

# Appendix A — Methods Catalogue

A reference of options at each pipeline stage. Status legend:
**[CHOSEN]** in v0 · **[ALT]** viable alternative · **[FUTURE]** later experiment / needs more data · **[N/A]** not applicable to our data, recorded so the gap is explicit.

The body of the document above is the *decided path*; this appendix is the *full option space* so no method is silently forgotten.

---

## A.1 Data preparation & alignment

| Method | Status | Notes |
|---|---|---|
| Derive `market` from symbol suffix | [CHOSEN] | rule-based; new market = one mapping entry |
| tz-aware timestamps (local + UTC) | [CHOSEN] | local for sessions/bars, UTC for global ordering |
| Event-time vs capture-time selection | [CHOSEN] | use event time (`markettime`/`localtime`); identical to capture here |
| Session labelling on local time | [CHOSEN] | India 1 session, HK/China 2 |
| Continuous-trade filtering via `cond` | [N/A] | business pre-delivered continuous-only data |
| Crossed/locked quote handling (ask ≤ bid) | [ALT→do] | drop or flag; protects microprice. Recommended even in v0 |
| Non-positive / zero size handling | [ALT→do] | drop bad quote/trade rows |
| Duplicate-timestamp handling | [ALT] | keep last, aggregate, or jitter; matters for as-of joins |
| Fat-finger / outlier price detection | [ALT] | rolling-MAD spike filter; careful not to clip real jumps |
| **As-of join (quotes→trades/bars) with tolerance** | [CHOSEN] | snap latest quote ≤ event time; set a max staleness tolerance |
| **Snapshot forward-fill policy** | [OPEN] | India/China quotes are 1–3s snapshots — decide whether to ffill quote state between snapshots when valuing trades. Real design choice with leakage implications |
| Resampling/alignment of snapshot vs continuous feeds | [OPEN] | harmonize granularity across markets before pooling |
| Holiday / half-day calendar handling | [FUTURE] | needed once multi-day/multi-month data arrives |
| Corporate-action / split adjustment | [FUTURE] | not relevant intraday single-day; needed across days |

## A.2 Bar construction

| Method | Status | Notes |
|---|---|---|
| Time bars | [ALT/baseline] | simple, aligned; fat-tailed returns, oversamples quiet periods |
| Tick bars | [N/A across markets] | conflation makes a "tick" non-comparable across venues; OK within HK alone |
| **Volume bars** | [CHOSEN] | conflation preserves size → robust; v0 choice |
| Dollar bars | [FUTURE — primary upgrade] | price×size; robust across price levels & symbols; 1-line change |
| Tick imbalance bars (TIB) | [FUTURE] | close on signed-tick imbalance vs expectation |
| Volume imbalance bars (VIB) | [FUTURE] | imbalance in signed volume |
| Dollar imbalance bars (DIB) | [FUTURE] | imbalance in signed dollar flow |
| Tick/Volume/Dollar **run bars** | [FUTURE] | close on runs of same-sign flow; most information-adaptive |
| Range / Renko bars | [ALT] | close on fixed price range; useful for vol-targeting |
| **CUSUM event sampling** | [FUTURE] | sample only when cumulative move exceeds threshold; pairs well with event labels, reduces redundant bars |
| Per-symbol threshold (causal trailing) | [FUTURE — needs multi-day] | trailing ADV/ADT; v0 uses in-sample (flagged) |
| Per-symbol threshold (in-sample) | [CHOSEN v0] | acceptable for *cutting* (not labels); flagged non-causal |
| Cold-start / thin-symbol threshold fallback | [ALT→do] | market-median threshold until enough history |
| Track `sum(price*size)` for VWAP | [CHOSEN] | free VWAP + instant dollar-bar path |
| Drop incomplete final bar | [CHOSEN] | partial volume scale distorts the population |

## A.3 Feature engineering

Grouped by family. All must be **causal** (trailing, never future) and ideally **dimensionless** before the model sees them.

**Quote / book (L1) features**
| Method | Status | Notes |
|---|---|---|
| Mid price | [CHOSEN] | base; not a model feature directly (scale) |
| Microprice = (bid·asize+ask·bsize)/(bsize+asize) | [CHOSEN] | size-weighted; better short-horizon anchor |
| Microprice offset / spread | [CHOSEN] | dimensionless lean direction |
| Relative spread (bps) | [CHOSEN] | dimensionless cost proxy |
| Order-book imbalance (OBI) | [CHOSEN] | (bsize−asize)/(bsize+asize) |
| Queue/size dynamics (Δbsize, Δasize) | [ALT] | short rolling deltas of top-of-book sizes |
| Depth (bsize+asize) normalized | [ALT] | liquidity proxy; normalize per symbol |
| Multi-level book features | [N/A] | data is L1 only — recorded explicitly |
| Quoted-spread volatility | [FUTURE] | rolling std of spread |

**Trade / flow features**
| Method | Status | Notes |
|---|---|---|
| Trade sign (tick rule) | [CHOSEN] | inferred; needs clean timestamps |
| Trade sign (Lee-Ready) | [ALT] | uses quote midpoint; better with aligned quotes |
| Signed volume / Order-Flow Imbalance (OFI) | [CHOSEN] | core flow signal |
| Trade intensity / arrival rate | [ALT] | counts per unit time/volume |
| VWAP − mid | [ALT] | execution-pressure proxy |
| Realized volatility | [CHOSEN] | rolling; also feeds label normalization |
| **VPIN** (vol-synchronized prob. informed trading) | [FUTURE] | toxicity/adverse-selection proxy |
| **Kyle's λ** (price impact per unit flow) | [FUTURE] | regress Δprice on signed volume |
| **Amihud illiquidity** | [FUTURE] | |Δreturn|/dollar volume |
| Hasbrouck price-impact / information share | [FUTURE] | heavier; multi-venue |

**Statistical / temporal features**
| Method | Status | Notes |
|---|---|---|
| Past log returns (multi-lag) | [CHOSEN] | momentum/reversal |
| Rolling higher moments (skew, kurtosis) | [FUTURE] | tail/asymmetry signal |
| Autocorrelation / Hurst exponent | [FUTURE] | trending vs mean-reverting regime |
| Entropy / complexity features | [FUTURE] | microstructure regime |
| **Multi-window features** (same feature at several lookbacks) | [ALT→strongly consider] | gives model multi-scale view; cheap, high value |
| **Time-of-day / session-phase encoding** | [ALT→do] | dimensionless (e.g. fraction-through-session); captures open/close effects without leaking identity |
| Liquidity tier / turnover percentile (dimensionless) | [FUTURE] | lets pooled model adapt without learning identity |

## A.4 Normalization (symbol/market-agnostic)

| Method | Status | Notes |
|---|---|---|
| Structural ratios (build dimensionless) | [CHOSEN — preferred] | OBI, spread bps, offsets; zero leakage risk |
| Per-symbol rolling **z-score** | [CHOSEN] | trailing mean/std; fragile to outliers |
| **Robust z-score** (median / MAD) | [ALT→prefer over z] | outlier-resistant variant of z-score |
| Per-symbol rolling **rank / percentile** | [CHOSEN — favored for fat tails] | outlier-proof, bounded, tree-friendly |
| Divide by **domain scale** (÷ median size, ÷ vol, ÷ tick) | [CHOSEN] | stable denominator; great for sizes/volumes/returns |
| **EWMA** normalization | [ALT] | exponential window vs flat; smoother, recency-weighted |
| Winsorization / clipping (pre-step) | [ALT→do] | cap extremes before standardizing |
| Vol-of-vol scaling | [FUTURE] | normalize by volatility of volatility |
| **Cross-sectional** (rank/z across universe at instant t) | [FUTURE] | standard equity-alpha; needs simultaneous broad universe; awkward across 4 timezones |
| Quantile / rank-Gaussian mapping | [FUTURE] | strongest equalizer; heavy-handed; fit causally |
| Causal application via `groupby('sym').transform(rolling)` | [CHOSEN] | the mechanism guaranteeing per-symbol + causal |

## A.5 Target / label preparation

| Method | Status | Notes |
|---|---|---|
| Forward **mid** return | [ALT] | simplest; replaced by microprice |
| Forward **microprice** return | [CHOSEN] | better short-horizon target |
| Express in **bps / log** | [CHOSEN] | scale-free across price levels |
| ÷ **trailing realized vol** (sigma units) | [CHOSEN] | same label = same economic event across symbols; match vol window to horizon |
| Fixed-horizon (clock) label | [CHOSEN] | 5/30/60 min vertical barrier |
| **Session-boundary masking** (NaN past session end) | [CHOSEN — critical] | prevents labels crossing lunch/close |
| First-passage vs fixed-horizon distinction | [ALT] | does price *hit* a level vs *where it is* at h |
| **Triple-barrier** (±k·σ + time) classification | [ALT — noted only] | not the v0 framing (regression chosen); recorded for completeness |
| **Meta-labeling** (2nd model: act / don't act on signal) | [FUTURE — high value for execution] | sizes/filters the primary signal; pairs naturally with a regression primary |
| Trend-scanning labels | [FUTURE] | label by fitted local trend significance |
| Quantized / bucketed regression target | [FUTURE] | robustness to outliers in the target |
| Multi-output joint label (all horizons together) | [FUTURE] | v0 keeps horizons separate |
| **Uniqueness / sample weighting** for overlapping labels | [ALT→do at 30/60m] | down-weights serially-correlated forward windows |

## A.6 Modelling

| Method | Status | Notes |
|---|---|---|
| **CatBoost regressor** (pooled) | [CHOSEN] | robust, handles non-linear mixed features, native categoricals if ever needed |
| LightGBM / XGBoost | [ALT] | comparable GBDTs; benchmark candidates |
| Linear / ElasticNet baseline | [ALT→do] | essential sanity baseline; beat it or rethink |
| Quantile / probabilistic regression | [FUTURE] | predict intervals, not just point — useful for sizing |
| Simple NN / MLP | [FUTURE] | only with much more data |
| Sequence models (LSTM / Temporal CNN / Transformer) | [FUTURE] | needs large multi-day data; risk of overfit now |
| One model **per horizon** | [CHOSEN] | each horizon a distinct signal-to-noise problem |
| Multi-task / shared-trunk across horizons | [FUTURE] | parameter sharing experiment |
| Pooled across symbols/markets (no identity features) | [CHOSEN] | exclude sym/market/source/raw price&size |
| Liquidity-tiered models | [FUTURE] | only if pooled is dragged by illiquid tail |

## A.7 Validation & evaluation

| Method | Status | Notes |
|---|---|---|
| Time-ordered split (never random) | [CHOSEN] | random shuffles future into train |
| **Walk-forward CV** | [CHOSEN] | rolling/expanding train→test |
| **Purge + embargo (per-horizon)** | [CHOSEN — critical] | embargo ≥ horizon; prevents label-overlap leakage |
| **Combinatorial Purged CV (CPCV)** | [FUTURE] | more robust paths; needs multi-day data |
| Information Coefficient (Spearman) | [CHOSEN — primary] | ordering quality |
| Directional accuracy / hit-rate | [CHOSEN] | sign quality |
| Rolling IC stability | [ALT→do] | stable small IC > spiky large IC |
| RMSE / MAE | [CHOSEN — secondary] | fitting diagnostic only |
| Naive baseline comparison (zero / persistence) | [CHOSEN→do] | the real bar to beat |
| Per-market / per-liquidity diagnostic slicing | [ALT→do] | see *where* signal lives (diagnostic, not a feature) |
| **Deflated Sharpe / PBO** (overfitting detection) | [FUTURE] | guard against backtest overfit at scale |
| Feature importance / SHAP | [ALT→do] | interpretability; sanity-check leakage |

## A.8 Scalability & engineering

| Method | Status | Notes |
|---|---|---|
| Per-level logic (market keys / symbol keys) + `groupby` fan-out | [CHOSEN] | adding markets/stocks = no code change |
| Schedules/tick rules in config dicts | [CHOSEN] | data, not code |
| Vectorize bar builder | [FUTURE→needed at scale] | Python-loop `groupby.apply` is the bottleneck |
| Chunked / parallel processing | [FUTURE] | per (symbol, day) is embarrassingly parallel |
| Checkpointing long stages | [ALT→do] | restartable bar building |
| Modular SDK (`x_prep`/`y_prep` + orchestrator, `(data, meta)` returns) | [ALT→recommended] | reusable, testable, leakage-safe |
| Automated data-quality gate | [ALT→do] | per-symbol report reruns on new data |
| Cold-start / min-history governance | [ALT→do] | rule before a symbol enters training |

---

**How to use this appendix:** when revisiting any stage, scan its table — **[CHOSEN]** is the current build, **[ALT→do]** items are low-risk improvements worth pulling forward, **[FUTURE]** items mostly unlock once multi-day data arrives, and **[N/A]** items are deliberately excluded so they're never re-litigated by accident.
