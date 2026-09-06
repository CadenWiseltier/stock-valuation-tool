# Stock Valuation & Investment Analysis Tool

A full equity-research-style valuation engine hidden behind a one-ticker
interface. Type a ticker, click **Analyze Stock**, and get a complete DCF
valuation, comparable-company analysis, bull/base/bear scenarios,
sensitivity analysis, and a quantitative 0-1000 Investment Score --
automatically, with every assumption disclosed and editable.

> ## ⚠️ Disclaimer
>
> **This is an educational portfolio project. It is not financial advice.**
>
> The scores and ratings it produces are the mechanical output of a
> quantitative model built on free, delayed, third-party data that may be
> incomplete or wrong. A rating of "Strong Buy" or "Avoid" is a model output,
> not a recommendation. Nothing here should be used as the basis for an
> investment decision. Do your own research and consult a licensed financial
> adviser.
>
> The author is not a licensed financial adviser and accepts no liability for
> any decision made using this software.

## Overview

Most "stock analyzer" side projects either require the user to manually
enter a dozen modeling assumptions, or reduce the analysis to a single
naive metric. This project does neither: it automatically retrieves a
company's financial statements, generates defensible baseline assumptions
from that company's own historical data, runs a full corporate-finance
valuation stack (DCF, WACC/CAPM, comparable companies, scenario and
sensitivity analysis), and combines the results into a single transparent,
formula-driven score -- while keeping the default user experience to
**ticker -> Analyze -> report**.

**Design philosophy: simple on the outside, sophisticated on the inside.**

## Features

- **One-input UX** -- enter a ticker, click Analyze, done. No required
  manual inputs.
- **Automatic DCF valuation** -- 5-year (adjustable) FCFF projection with
  automatically generated, history-derived assumptions.
- **WACC via CAPM** -- cost of equity, cost of debt, and capital weights
  computed from live market and financial-statement data.
- **Comparable-company analysis** -- sector-matched peer group, P/E,
  EV/EBITDA, EV/Revenue, Price/Sales multiples, and implied valuation.
- **Bull / Base / Bear scenarios** -- systematic, documented shifts to
  growth, margin, WACC, and terminal growth.
- **Sensitivity analysis** -- WACC x Terminal Growth and Revenue Growth x
  Operating Margin grids.
- **Historical financial-statement analysis** -- revenue, margins, EPS,
  free cash flow, ROE, ROIC, leverage, and liquidity, with data-driven
  (never fabricated) trend commentary.
- **0-1000 quantitative, PRICE-SENSITIVE Fundamental Investment Score**
  across 9 weighted categories, with a full auditable breakdown, computed
  positive/negative factors, and a Price Sensitivity Test showing how the
  score changes at hypothetical +/-20%/10% prices.
- **0-1000 forward-looking Speculation Score** across 8 weighted categories
  (TAM/Future Market, Explosive Growth, Industry Optionality, Catalysts,
  5-10 Year Asymmetric Upside, Win Probability, Competitive Advantage,
  Market Mispricing), driven by an auditable industry knowledge base, with
  its own Confidence, Thesis, Key Risks, and a required five-part
  explanation for any score above 700 -- deliberately able to disagree with
  the Fundamental Score.
- **Advanced Assumptions panel** -- every automatically generated
  assumption is visible and editable, hidden by default.
- **Graceful error handling** -- invalid tickers, missing statements, and
  API issues degrade to a clear message or "N/A," never a crash.
- **Full unit test suite** covering every core financial formula.

## Financial Methodology

Full detail (every formula, every assumption rule, every scoring formula)
is in **[docs/methodology.md](docs/methodology.md)**. Summary:

- **DCF**: `FCFF = EBIT x (1 - Tax Rate) + D&A - CapEx - Change in NWC`,
  discounted at WACC, with a Gordon Growth terminal value
  (`TV = FCFF_(n+1) / (WACC - g)`) and an Exit Multiple cross-check.
- **WACC**: `E/(D+E) x Cost of Equity + D/(D+E) x After-Tax Cost of Debt`.
- **CAPM**: `Cost of Equity = Risk-Free Rate + Beta x Equity Risk Premium`.
- **Comparable Companies**: sector-matched peer multiples applied to the
  target's own EPS/EBITDA/Revenue to produce implied per-share values.
- **Sensitivity Analysis**: recomputes intrinsic value across a grid of
  WACC/terminal-growth and growth/margin combinations.
- **ROE**: `Net Income / Average Shareholders' Equity`.
- **ROIC**: `NOPAT / Average Invested Capital`, where
  `NOPAT = EBIT x (1 - Effective Tax Rate)` and
  `Invested Capital = Total Debt + Total Equity - Cash`.
- **Free Cash Flow**: `Operating Cash Flow - CapEx`.
- **Investment Score**: see below.

## Two Scores, Two Questions

The app shows **two separate 0-1000 scores**, deliberately designed to be
able to disagree:

- **Fundamental Investment Score** -- "Is this stock attractively priced
  enough to buy today, given its fundamentals, expected return, valuation,
  and risk?"
- **Speculation Score** -- "How much potential does this stock have to
  **explode in value** if its bullish thesis, catalysts, or
  emerging-industry opportunity plays out, and how credible is that
  scenario?"

**A high Speculation Score does NOT mean 'good investment', and a low one
does not mean 'bad company'.** The two are independent judgments:

| Profile | Fundamental | Speculation | Contradiction? |
|---|---|---|---|
| Unprofitable, expensive, huge emerging market | 350 | 900 | **No** |
| Mature, profitable, fairly priced, settled industry | 850 | 200 | **No** |

In validation across real companies, the largest and most profitable
companies score *lowest* on Speculation while scoring highest on the
Fundamental Score -- which is the clearest demonstration that the two are
genuinely measuring different things.

## Fundamental Investment Score

**This is a PRICE-sensitive score, not a company-quality score.** It
answers "if I bought this stock today at today's price, how attractive is
the risk-adjusted expected return?" -- not just "is this a good company?"
A wonderful business at an unreasonable price scores as a Hold/Weak; a
decent business at a genuine discount can score as a Strong Buy. Full
detail (every formula) is in
[docs/methodology.md](docs/methodology.md#7-investment-score-0-1000).

To make that true in the math, every category is one of two kinds:

| Category | Weight | Price-dependent? |
|---|---|---|
| Valuation & Margin of Safety | 250 | Yes |
| Expected Return at Current Price | 200 | Yes |
| Growth-Adjusted Valuation | 100 | Yes |
| Scenario-Weighted Risk/Reward | 100 | Yes |
| Growth | 70 | No |
| Profitability | 70 | No |
| Financial Health | 70 | No |
| Cash Flow Quality | 40 | No |
| Risk (lower risk scores higher) | 100 | No |
| **Total** | **1000** | |

**Business Quality** (Growth + Profitability + Financial Health + Cash
Flow Quality) is capped at 250 of 1000 points **by design**, so even a
perfect quality score cannot alone produce a "Strong Buy" (800+) -- the
remaining 750 points require the CURRENT price to actually be attractive
relative to the DCF, comparable companies, the company's own valuation
history, its peer group, and a probability-weighted scenario analysis.
This property is directly unit-tested in
`tests/test_scoring.py::test_quality_alone_cannot_reach_strong_buy`.

The app also displays four 0-100 headline readouts -- **Business
Quality**, **Valuation Attractiveness**, **Expected Return**, and
**Risk** -- and a **Price Sensitivity Test** that recomputes the whole
score at hypothetical prices 20%/10% above and below today's actual
price, to make visible exactly how much of the score is coming from
price vs. from company quality
(`tests/test_scoring.py::test_price_sensitivity_score_decreases_as_price_increases`
confirms the score moves the correct direction as price moves).

Scores map to ratings:

| Score | Rating |
|---|---|
| 900-1000 | Exceptional |
| 800-899 | Strong Buy |
| 700-799 | Buy |
| 600-699 | Moderate / Watch |
| 500-599 | Hold |
| 400-499 | Weak |
| 0-399 | Avoid |

The score is **business-model aware**: metrics that are meaningless for a
given kind of company (gross margin on a bank, a FCFF DCF on a lender,
industrial leverage limits on a REIT, P/E on a depreciation-heavy property
company) are excluded and their weight redistributed across the metrics that
do apply, rather than scored at a flat neutral. See
[methodology §7.11](docs/methodology.md) for the full scoring audit — nine
structural defects found by scoring 49 real companies and how each was fixed.

## Speculation Score

Answers a forward-looking, opportunity-focused question: **"How much
potential does this stock have to explode in value if its bullish thesis,
catalysts, or emerging-industry opportunity plays out -- and how credible is
that scenario?"** Full detail (every formula) is in
[docs/methodology.md](docs/methodology.md#9-speculation-score-0-1000-srcspeculativepy-srcthemespy).

| Category | Weight | Measures |
|---|---:|---|
| TAM / Future Market Opportunity | 150 | How large could the addressable market become? |
| Explosive Growth Potential | 175 | Could the business grow many times over? |
| Industry / Technology Optionality | 150 | Could a breakthrough transform its value? |
| Catalyst Potential | 125 | How likely is a discrete repricing event? |
| 5-10 Year Asymmetric Upside | 150 | How many times larger could it become? |
| Probability of Becoming a Major Winner | 100 | Is the bullish scenario actually reachable? |
| Competitive / First-Mover Advantage | 75 | Could it capture the industry's economics? |
| Market Mispricing / Underappreciated Potential | 75 | How much isn't priced in yet? |
| **Total** | **1000** | |

### Rating bands

| Score | Rating |
|---|---|
| 951-1000 | Once-in-a-Generation Speculative Opportunity |
| 851-950 | Exceptional / Explosive Speculation |
| 751-850 | Extreme Speculation |
| 601-750 | Very High Speculation |
| 451-600 | High Speculation |
| 301-450 | Moderate Speculation |
| 151-300 | Low Speculation |
| 0-150 | Minimal Speculative Upside |

### Three things it deliberately does NOT do

**It does not penalize a high valuation.** No P/E, EV/EBITDA, EV/Sales or
peer-multiple comparison appears anywhere. Expensiveness belongs to the
Fundamental Score. Price enters through exactly one channel -- market-cap
**scale**, in the "current market cap → potential market cap" ratio, because
a $4T company genuinely cannot 10x as easily as a $4B one. A unit test
changes a company's reported P/E, P/S and EV/EBITDA and asserts the score
does not move.

**It does not reward volatility.** Beta, scenario dispersion, short interest
and retail popularity appear in no formula. A unit test sets beta to 4.5,
asserts an identical score, and additionally asserts no subscore label
anywhere contains "beta", "volatil", "dispersion" or "short interest".

**It does not invent catalysts.** The tool has no access to news or company
calendars. It names the catalyst *archetypes* that apply to an industry
("NRC design certification milestones", "Phase 3 data readouts"), labeled as
industry-level types, never as scheduled company events.

### Where the forward-looking data comes from

Financial statements cannot tell you whether a company faces a $10B or a
$1T market. Every forward-looking assumption therefore lives in one
auditable file, **`src/themes.py`** -- estimated TAM, industry growth,
commercial maturity, breakthrough potential, catalyst density, plausible
winner economics -- each with a plain-language source note.
`src/speculative.py` contains no industry constants of its own. Disagree
with an estimate? Change it in one place and every score updates.

These figures are **order-of-magnitude estimates, not facts**, drawn from
published market research where different houses often differ by 2-5x. They
are consumed on a log scale specifically to limit their leverage.

### Two gates against hype

1. **Keyword threshold.** Matching an industry needs either a match in the
   company's own `industry`/`sector`/name field, or two distinct keywords in
   its business summary. This exists because of a real false positive found
   in testing: Exxon Mobil's description lists "low-carbon data center"
   among a dozen business lines, which had reclassified an oil major as an
   AI company. Now pinned as a regression test.
2. **Evidence requirement.** Theme membership alone earns little.
   `theme_evidence_strength()` independently checks revenue, growth, R&D,
   gross margin and capital investment. A shell company with a trendy
   business summary and no financials receives only 35% of its industry's
   raw optionality.

### Required explanation above 700

Any score above 700 produces a five-part structured explanation: why the
opportunity is potentially explosive, the industry/TAM driving it, the
applicable catalyst types, the upside scenario with full arithmetic, and the
biggest reason the thesis could fail (chosen mechanically as the
weakest-scoring category, so it is always the model's own most negative
finding). Above 850 it must additionally justify the "explosive"
qualification.

### Confidence, and what the score is not

**Speculative Confidence** (0-100) measures how much underlying data was
available -- confidence in the *analysis*, not in the outcome.
**Corroborating evidence** (0-100) measures how much the financials support
the industry claim. Neither is a probability, and **the Speculation Score is
explicitly not a probability**: a 900 does not mean a 90% chance of
anything.

## Project Structure

```
stock-valuation-tool/
├── app.py                      # Streamlit UI -- layout only, no financial logic
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data.py                 # yfinance data retrieval; the only module that touches the network
│   ├── financial_analysis.py   # Historical statement analysis: growth, margins, FCF, ROE, ROIC, balance sheet
│   ├── dcf.py                  # DCF engine: CAPM, WACC, automatic assumptions, FCFF projection, terminal value
│   ├── comparables.py          # Comparable-company peer selection, multiples, implied valuation
│   ├── scenarios.py            # Bull / Base / Bear / Extreme Bull scenario modeling
│   ├── scoring.py              # 0-1000 Fundamental Investment Score and rating
│   ├── speculative.py          # 0-1000 forward-looking Speculation Score and rating
│   ├── themes.py               # Auditable emerging-industry knowledge base (TAM, growth, maturity, catalysts)
│   ├── sensitivity.py          # WACC/terminal-growth and growth/margin sensitivity grids
│   └── charts.py               # Plotly chart construction
│
├── tests/
│   ├── conftest.py             # Synthetic, hand-computable company fixture (no network calls)
│   ├── test_data.py
│   ├── test_dcf.py
│   ├── test_financial_analysis.py
│   ├── test_scenarios.py
│   ├── test_scoring.py
│   └── test_speculative.py
│
└── docs/
    └── methodology.md          # Full formula-by-formula methodology reference
```

Every file in `src/` does one job. `app.py` never computes a financial
metric directly -- it only calls into `src/` and renders the result. This
separation is what makes the test suite possible: every formula can be
tested in isolation, with no Streamlit or network dependency.

## Technology

- Python 3.10+
- pandas / NumPy -- data manipulation and calculation
- yfinance -- financial data retrieval (Yahoo Finance)
- Streamlit -- web application framework
- Plotly -- interactive charting
- pytest -- unit testing

## Installation

Requires **Python 3.10 or newer**. If you don't have Python, get it from
[python.org/downloads](https://www.python.org/downloads/) and tick
**"Add python.exe to PATH"** on the installer's first screen.

### Windows — no command line needed

1. Click the green **Code** button at the top of this page → **Download ZIP**.
2. Right-click the downloaded `.zip` → **Extract All…**
3. Open the extracted folder and double-click **`Start App.bat`**.

It finds Python, installs the dependencies and launches the app.

> **Extract the ZIP before running it.** Windows will let you double-click
> `Start App.bat` from inside the zip, but it copies only that one file to a
> temporary folder, so the app can't find anything else and the install fails.
> Likewise, use **Download ZIP** rather than the download arrow on an
> individual file — that gives you only that file.

### Any platform — with the command line

```bash
git clone https://github.com/CadenWiseltier/stock-valuation-tool.git
```

```bash
cd stock-valuation-tool && pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

1. Launch the application with the command above.
2. Enter a ticker (e.g. `AMZN`).
3. Click **Analyze Stock**.
4. Review the **Fundamental Investment Score** and, below it, the
   **Speculation Score** at the top of the report.
5. Expand any section (Financial Performance, DCF Valuation, Comparable
   Companies, Bull/Base/Bear, Sensitivity Analysis, Investment Score
   Breakdown, Speculation Score Breakdown, Risks, Assumptions) for full
   detail.
6. Optionally open **Advanced Assumptions** before analyzing to override
   any automatically generated model input.

## Deploying it as a public website

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) for
free, straight from this repository — no server, database or config required.

1. Sign in at **share.streamlit.io** with this GitHub account.
2. **New app** → select this repository → main file `app.py` → **Deploy**.

That's the whole process; `requirements.txt` is all it needs.

### What to expect from the free data source

Market data comes from Yahoo Finance via the unofficial `yfinance` library.
It is free and requires no API key, but it is **rate-limited per IP address**.
On a public deployment every visitor shares the server's IP, so under traffic
Yahoo will start refusing requests.

Two things already soften this:

- Results are cached server-side for an hour and shared across all visitors,
  so popular tickers are served instantly without a new request.
- Rate limiting is detected and reported as such, rather than being
  misreported as an invalid ticker (see `DataProviderError` in `src/data.py`).

If the app outgrows that, `src/data.py` is the **only** module that touches
the network, and everything downstream depends on the `StockDataBundle`
schema rather than on yfinance's data shapes. Moving to a paid provider with
an API key (Financial Modeling Prep, Tiingo, Polygon) means rewriting that one
file — put the key in Streamlit's **Secrets** manager, never in the repo.
`.streamlit/secrets.toml` is already in `.gitignore`.

## Example

Try `AMZN`, `AAPL`, or `MSFT` for a full report. Both `AMZN` and `AAPL`
have been used as end-to-end smoke tests during development (see
[docs/methodology.md](docs/methodology.md#8-known-limitations) for a
worked example of a real limitation surfaced by the `AMZN` run).

## Limitations

- **Data availability**: the free data provider typically supplies ~4
  years of annual statements; older history is not available.
- **API limitations**: Yahoo Finance is unofficial and free -- it can rate
  limit, lag real-time prices, or occasionally omit a field. The app
  displays "N/A" for anything it cannot reliably compute rather than
  guessing.
- **Forecast uncertainty**: any 5-year cash-flow forecast is inherently
  uncertain. The sensitivity and scenario sections exist specifically to
  make that uncertainty visible rather than hide it behind a single
  number.
- **Model assumptions**: automatic assumptions (growth fade, margin
  normalization, CapEx/D&A ratios) are documented, simple, and
  transparent -- but they are simplifications of what a professional
  analyst would build with company guidance and a detailed operating
  model. A historical-average CapEx assumption in particular can
  understate free cash flow for a company mid-way through an unusually
  large, temporary capital-investment cycle -- see
  [docs/methodology.md](docs/methodology.md#8-known-limitations).
- **Comparable-company limitations**: peers are selected by sector
  classification from a curated list, not a professional peer-screening
  service, and do not account for differences in business model, scale,
  or growth stage within a sector.
- **Sensitivity of DCF valuation**: small changes in WACC or terminal
  growth materially change the DCF output, as the Sensitivity Analysis
  section is designed to show directly.

## Testing

```bash
pytest tests/ -v
```

94 unit tests cover CAPM, WACC, the FCFF formula, terminal value,
enterprise-to-equity value bridging, revenue/margin/FCF calculations, ROE,
ROIC, scenario ordering (including the Extreme Bull tier),
null-vs-zero-vs-negative handling, and both scores' bounds and category
weighting.

The Speculation Score's defining behaviours are pinned individually: that
valuation multiples do not affect it, that beta and volatility do not affect
it, that a smaller company has more room to multiply, that an emerging
industry outscores a mature one on identical financials, that a trendy
keyword alone cannot manufacture a high score, that an incidental keyword
mention cannot reclassify a company, and that no thesis or risk bullet is
ever fabricated.

The Fundamental Score's recalibration is pinned by its own regression tests:
that inapplicable metrics redistribute rather than score half credit, that a
debt-free balance sheet is not penalized for having no debt, that a high-ROIC
compounder earns a multiple premium while a low-ROIC business earns a
value-trap discount, that growth is measured per-share and haircut for margin
collapse, that an implausible DCF is reported unavailable rather than scored,
and that a small-sample valuation percentile is shrunk toward neutral.

Everything runs against a synthetic, hand-computable company fixture, so the
suite is fully offline and deterministic.

## License

[MIT](LICENSE) — free to use, modify and distribute, with attribution and
without warranty.

The MIT warranty disclaimer applies to the financial content as well: this is
an educational project, its outputs are the mechanical results of a model
built on free third-party data, and nothing it produces is financial advice
or should be relied upon for an investment decision.

---

*Built as a portfolio project demonstrating financial modeling, equity
valuation, and Python engineering for finance/investment internship
applications.*
