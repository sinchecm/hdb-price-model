"""
Load the trained model once and predict prices for new inputs.
Import this from your Streamlit app.py — do NOT retrain inside the app.
"""

import joblib
from features import engineer_features_for_single_input

MODEL_PATH = "hdb_price_model.joblib"

_artifact = None


def load_artifact(path: str = MODEL_PATH) -> dict:
    global _artifact
    if _artifact is None:
        _artifact = joblib.load(path)
    return _artifact


def predict_price(
    town: str,
    flat_type: str,
    flat_model: str,
    floor_area_sqm: float,
    storey_range: str,
    lease_commence_date: int,
    remaining_lease_years: float,
    txn_year: int,
    txn_month: int,
) -> float:
    """Returns a single predicted resale price in SGD."""
    artifact = load_artifact()
    row = engineer_features_for_single_input(
        town=town,
        flat_type=flat_type,
        flat_model=flat_model,
        floor_area_sqm=floor_area_sqm,
        storey_range=storey_range,
        lease_commence_date=lease_commence_date,
        remaining_lease_years=remaining_lease_years,
        txn_year=txn_year,
        txn_month=txn_month,
    )
    pred = artifact["pipeline"].predict(row)[0]
    return float(pred)


if __name__ == "__main__":
    # quick smoke test
    price = predict_price(
        town="BISHAN",
        flat_type="4 ROOM",
        flat_model="Model A",
        floor_area_sqm=93.0,
        storey_range="10 TO 12",
        lease_commence_date=1990,
        remaining_lease_years=63.0,
        txn_year=2026,
        txn_month=6,
    )
    print(f"Example prediction: S${price:,.0f}")
    artifact = load_artifact()
    print("Model held-out (future-months) accuracy:", artifact["held_out_eval_metrics"])
