# HDB Resale Price Model — v2

A rebuild of the C3.2 practice model, designed to actually be deployed in
your Streamlit app rather than just clear a course benchmark.

## What changed vs. the course notebook

| | Course notebook (best result) | This model |
|---|---|---|
| Features | floor area, lease year, floor level, flat_type, town, flat_model | + `remaining_lease_years` (parsed from the data, not derived from lease year), `flat_age_at_txn`, `months_since_2017` (time trend) |
| Model | Stacked RF + Gradient Boosting | Tuned XGBoost |
| Evaluation | Random 80/20 split only | Random split **and** a time-based holdout (last 6 months) |
| MAE (random split) | ~S$72,000 | **S$28,000** |
| MAE (last 6 months, unseen future) | not measured | **S$34,000** (5.1% MAPE, R² 0.948) |

The single biggest win was **`months_since_2017`** — a simple trend
feature. HDB prices moved a lot from 2017 to mid-2026, and without telling
the model "this transaction happened later," it has no way to know that a
4-room in Bishan sold for more in 2026 than in 2019. This is a case where
one honest feature beat switching model families.

## Why the time-based split matters

The course notebook only checks a random 80/20 split. That's optimistic
for a price model: random splitting lets the model "see" price levels from
every era of the data during training, so it's partly grading itself on
transactions from time periods it already knows. Your Streamlit app will
only ever be asked to price transactions that haven't happened yet — so
this model is also scored on a **held-out set of the most recent 6
months**, which it never touched during training or tuning. That's the
number that should set your expectations for the live app: typically
**within about S$34k (5.1%)**.

## Files

- `features.py` — all feature engineering, shared by training and inference so they can never drift apart.
- `train_model.py` — trains Linear Regression, Random Forest, and XGBoost (default + tuned), prints a comparison table, and saves the winning pipeline.
- `hdb_price_model.joblib` — the trained, deployable artifact (preprocessing + model bundled together).
- `predict.py` — a thin loader/predict function to import into your app.
- `app_example.py` — a minimal Streamlit form wired up to the model, to show the integration pattern.

## Using it in your app

```python
from predict import predict_price

price = predict_price(
    town="BISHAN", flat_type="4 ROOM", flat_model="Model A",
    floor_area_sqm=93.0, storey_range="10 TO 12",
    lease_commence_date=1990, remaining_lease_years=63.0,
    txn_year=2026, txn_month=6,
)
```

Copy `hdb_price_model.joblib`, `features.py`, and `predict.py` next to
your `app.py` and merge in the relevant bits from `app_example.py`.

## Retraining

Prices will keep drifting, so re-run `train_model.py` every few months as
new transactions land (it re-downloads the latest CSV automatically —
delete `hdb_raw_cache.parquet` first to force a fresh pull). Watch the
"TIME split" MAE over time: if it starts drifting up, that's your signal
the model needs a refresh, not just more historical rows.

## Honest limitations

- No location granularity beyond `town` — two blocks in the same town can
  differ by MRT distance, which this model can't see. That's the natural
  next feature to add (would need geocoding block/street_name, which was
  intentionally left out of this pass — flag if you want to go there next).
- Random Forest in the comparison table was deliberately kept small
  (25 trees, depth 10) to fit the sandbox's single CPU core — it's there
  to illustrate the model-family comparison, not as a competitive baseline.
  XGBoost was the clear winner either way.
- `flat_model` has 21 categories and some are rare (e.g. `Multi Generation`
  appears alongside `MULTI-GENERATION` flat_type) — worth a data-cleaning
  pass if you want to squeeze out more accuracy.
