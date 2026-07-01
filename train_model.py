"""
Train an HDB resale price model — from scratch, built to be genuinely
deployable (not just a teaching exercise).

What this does differently from the C3.2 practice notebook:
  1. More signal: adds remaining_lease_years (already in the raw data,
     just needed parsing), flat_age_at_txn, and months_since_2017 (a time
     trend — HDB prices moved a LOT between 2017 and 2026, and the
     original notebook never told the model that).
  2. Honest evaluation: in addition to a random train/test split, it also
     evaluates on a TIME-based holdout (last 6 months of transactions).
     That's the split that matters for a real app: you're always
     predicting prices for transactions that haven't happened yet, not
     ones randomly drawn from the past. A model can look great on a
     random split and still be shaky on genuinely new months.
  3. Model comparison: Linear Regression -> Random Forest -> XGBoost
     (default) -> XGBoost (tuned via RandomizedSearchCV), so you can see
     exactly how much each change is worth in dollars.
  4. A single deployable artifact: the winning model is refit on ALL
     available data and saved as one pipeline object (preprocessing +
     model together) via joblib, ready to load in the Streamlit app.

Run with:  python train_model.py
Takes about 4-5 minutes end to end (the RandomizedSearchCV step dominates).
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import ALL_FEATURES, CATEGORICAL_FEATURES, engineer_features

warnings.filterwarnings("ignore")

DATA_URL = (
    "https://raw.githubusercontent.com/flexfengfeng/6m-data-C3.2/"
    "main/notebooks/data/"
    "Resale_flat_prices_based_on_registration_date_from_Jan-2017_onwards.csv"
)
TEST_HOLDOUT_MONTHS = 6  # how many most-recent months to hold out as the "future" test set
RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )


def score(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def fmt(m: dict) -> str:
    return f"MAE=${m['MAE']:,.0f}  MAPE={m['MAPE']:.1%}  R2={m['R2']:.3f}"


def main():
    print("Loading data...")
    import os
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "hdb_raw_cache.parquet")
    if os.path.exists(cache_path):
        raw = pd.read_parquet(cache_path)
    else:
        raw = pd.read_csv(DATA_URL)
        raw.to_parquet(cache_path)
    df = engineer_features(raw)
    X, y = df[ALL_FEATURES], df["resale_price"]
    print(f"{len(df):,} rows, {df['month'].min()} to {df['month'].max()}")

    # ---- Split 1: random (comparable to the original course notebook) ----
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ---- Split 2: time-based (the honest, "predict the future" check) ----
    cutoff = df["months_since_2017"].max() - TEST_HOLDOUT_MONTHS
    train_mask = df["months_since_2017"] <= cutoff
    Xt_train, yt_train = X[train_mask], y[train_mask]
    Xt_test, yt_test = X[~train_mask], y[~train_mask]
    print(
        f"Time-split: train on <= month {cutoff}, "
        f"test on last {TEST_HOLDOUT_MONTHS} months "
        f"({len(Xt_test):,} rows)"
    )

    results = {}

    # ---- Baseline: Linear Regression ----
    lin = Pipeline([("pre", make_preprocessor()), ("model", LinearRegression())])
    lin.fit(Xr_train, yr_train)
    results["Linear Regression (random split)"] = score(yr_test, lin.predict(Xr_test))

    # ---- Random Forest ----
    rf = Pipeline(
        [
            ("pre", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=25, max_depth=10, random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    rf.fit(Xr_train, yr_train)
    results["Random Forest (random split)"] = score(yr_test, rf.predict(Xr_test))

    # ---- XGBoost, default-ish params, random split ----
    xgb_default = Pipeline(
        [
            ("pre", make_preprocessor()),
            ("model", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                    random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    xgb_default.fit(Xr_train, yr_train)
    results["XGBoost, default (random split)"] = score(yr_test, xgb_default.predict(Xr_test))

    # Same default XGBoost, but on the honest time split, to show the gap
    xgb_default_time = Pipeline(
        [
            ("pre", make_preprocessor()),
            ("model", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                    random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    xgb_default_time.fit(Xt_train, yt_train)
    results["XGBoost, default (TIME split — future months)"] = score(
        yt_test, xgb_default_time.predict(Xt_test)
    )

    # ---- XGBoost, tuned via RandomizedSearchCV, trained on the time-split
    #      training set so the search never sees the "future" test months ----
    print("Tuning XGBoost (RandomizedSearchCV, ~3-4 min)...")
    param_dist = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [4, 5, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }
    tuning_pipe = Pipeline(
        [("pre", make_preprocessor()), ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1))]
    )
    search = RandomizedSearchCV(
        tuning_pipe,
        param_dist,
        n_iter=6,
        cv=2,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(Xt_train, yt_train)
    print(f"  search took {time.time() - t0:.0f}s. Best params: {search.best_params_}")

    tuned = search.best_estimator_
    results["XGBoost, TUNED (TIME split — future months)"] = score(yt_test, tuned.predict(Xt_test))

    # Overfitting check for the winning model
    train_score = score(yt_train, tuned.predict(Xt_train))
    print(f"\nOverfitting check (tuned XGBoost): train {fmt(train_score)}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:45s}  {fmt(m)}")
    print("=" * 70)

    # ---- Final deployable model: refit best config on ALL data ----
    print("\nRefitting winning config on the FULL dataset for deployment...")
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    final_model = Pipeline(
        [
            ("pre", make_preprocessor()),
            ("model", XGBRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    final_model.fit(X, y)

    artifact = {
        "pipeline": final_model,
        "features": ALL_FEATURES,
        "trained_on_rows": len(df),
        "trained_on_date_range": (df["month"].min(), df["month"].max()),
        "held_out_eval_metrics": results["XGBoost, TUNED (TIME split — future months)"],
        "best_params": best_params,
    }
    joblib.dump(artifact, "hdb_price_model.joblib")
    print("Saved -> hdb_price_model.joblib")

    # Feature importance, for sanity-checking / a chart in the app
    pre = final_model.named_steps["pre"]
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    numeric_cols = [c for c in ALL_FEATURES if c not in CATEGORICAL_FEATURES]
    feature_names = np.concatenate([cat_names, numeric_cols])
    importances = final_model.named_steps["model"].feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    print("\nTop 15 features by importance:")
    print(top.to_string())


if __name__ == "__main__":
    main()