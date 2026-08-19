# Credit Risk Scorecard & PD Model

**[Open the interactive dashboard](https://YOUR-GITHUB-USERNAME.github.io/credit-pd-scorecard/)**
(enable GitHub Pages on this repo → Settings → Pages → Deploy from branch `main` / root)


A retail credit-risk scorecard pipeline on real data: Weight-of-Evidence
(WOE) feature engineering, Information Value (IV) feature selection, a
logistic-regression PD model, and an IFRS 9 staging/ECL overlay — the
standard architecture behind a bank's retail credit scorecard.

Built as part of the Risk Management and Data Analytics & Machine Learning
coursework of my Master of Finance.

## Data

**UCI Statlog (German Credit) dataset** — 1,000 real loan applications,
20 attributes (checking account status, credit history, purpose, duration,
savings, employment, property, age, etc.), binary good/bad credit label.
[Source](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
Downloaded automatically by the pipeline; a local copy ships in `data/`.

## Pipeline

1. **`woe.py`** — bins numeric features into quintiles, computes Weight-of-Evidence
   and Information Value per feature (with Laplace smoothing so sparse bins
   don't blow up), flags features by predictive strength.
2. **`pd_model.py`** — keeps features with IV ≥ 0.10 (a standard scorecard
   cutoff), trains a logistic regression on WOE-transformed features,
   evaluates with AUC and the KS statistic, and converts log-odds to
   conventional scorecard points via the points-to-double-odds (PDO) transform.
3. **`ifrs9_staging.py`** — classifies each account into IFRS 9 Stage 1/2/3
   based on PD deterioration since origination and default status, then
   computes Expected Credit Loss per stage (12-month ECL for Stage 1,
   lifetime ECL for Stage 2/3).

## Run it

```bash
pip install -r requirements.txt
cd src
python run_report.py
```

Outputs land in `output/`: `summary_report.txt` and three plots
(`roc_curve.png`, `information_value.png`, `ifrs9_coverage.png`).

## Sample output

```
Dataset: UCI Statlog German Credit, 1000 accounts, default rate 30.0%

-- Information Value ranking (top 6 kept) --
 checking_status   0.659   Medium-to-strong predictor
  credit_history   0.291   Medium predictor
 duration_months   0.213   Medium predictor
  savings_status   0.188   Medium predictor
         purpose   0.164   Medium predictor
        property   0.112   Medium predictor

-- Model performance (held-out test set) --
  AUC: 0.801   KS statistic: 0.497

-- IFRS 9 staging & Expected Credit Loss --
       accounts  total_EAD  total_ECL  coverage_ratio
stage
1           311     835,133     38,714            5.0%
2           389   1,254,687    406,478           32.0%
3           300   1,181,438    531,647           45.0%
```

An AUC of 0.80 and KS of ~0.50 are in the realistic range for a retail
scorecard built on this dataset — not inflated, and the coverage ratio
rising monotonically from Stage 1 to Stage 3 is the basic sanity check
for any IFRS 9 implementation.

## Why these design choices
- **WOE/IV instead of raw features or one-hot encoding** because that's the
  actual industry standard for regulated scorecards — WOE handles
  categorical and non-linear numeric effects in one linear-model-friendly
  step, and IV gives a defensible, auditable feature-selection criterion.
- **IV ≥ 0.10 cutoff** is a widely used scorecard convention (below ~0.02 is
  "not predictive," above ~0.5 usually signals leakage) — applied here,
  not just asserted, via the `iv_strength()` sanity check.
- **IFRS 9 staging is a simplified illustration**, not a production engine —
  real implementations also use days-past-due triggers, watchlist overrides,
  and forward-looking macro scenarios. That's stated plainly in the code
  docstring so it isn't oversold in an interview.

## Next extensions
- Replace the illustrative origination-PD noise with a proper PD-at-origination
  vintage model
- Add a challenger model (gradient boosting) and compare AUC/KS/stability
- Population Stability Index (PSI) monitoring for model drift

## Requirements
See `requirements.txt`. Python 3.10+.

## Interactive front end

`index.html` at the repo root is a self-contained, static, interactive dashboard
(Plotly.js via CDN) that reads `output/model_data.json` — no server needed, so it
runs directly on GitHub Pages. Regenerate that JSON after any pipeline change with:

```bash
cd src
python export_data.py
```
