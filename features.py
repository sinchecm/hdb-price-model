"""
Feature engineering for the HDB resale price model.

This module is imported both by the training script and by the Streamlit
app, so the exact same transformation logic runs at train time and at
inference time. That's the #1 way deployed models silently break — never
duplicate this logic in two places.
"""

import pandas as pd

NUMERIC_FEATURES = [
    "floor_area_sqm",
    "remaining_lease_years",
    "floor_level",
    "flat_age_at_txn",
    "months_since_2017",
]

CATEGORICAL_FEATURES = [
    "town",
    "flat_type",
    "flat_model",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _parse_remaining_lease(s: str) -> float:
    """'61 years 04 months' -> 61.33. Also handles '61 years' with no months."""
    parts = str(s).split()
    years = int(parts[0])
    months = int(parts[2]) if len(parts) > 2 else 0
    return years + months / 12


def _storey_mid(storey_range: str) -> float:
    """'10 TO 12' -> 11.0"""
    lo, hi = storey_range.split(" TO ")
    return (int(lo) + int(hi)) / 2


def engineer_features(raw: pd.DataFrame, base_year: int = 2017) -> pd.DataFrame:
    """
    Takes the raw HDB resale CSV columns and returns a dataframe with
    ALL_FEATURES ready to feed into the model pipeline.

    `raw` must contain: month, town, flat_type, flat_model, storey_range,
    floor_area_sqm, lease_commence_date, remaining_lease.

    resale_price (the target) is left untouched if present, and is NOT
    required to be present (needed for inference on new/unseen flats).
    """
    df = raw.copy()

    df["remaining_lease_years"] = df["remaining_lease"].apply(_parse_remaining_lease)
    df["floor_level"] = df["storey_range"].apply(_storey_mid)

    txn_year = df["month"].str.slice(0, 4).astype(int)
    txn_month = df["month"].str.slice(5, 7).astype(int)
    df["months_since_2017"] = (txn_year - base_year) * 12 + (txn_month - 1)
    df["flat_age_at_txn"] = txn_year - df["lease_commence_date"]

    return df


def engineer_features_for_single_input(
    town: str,
    flat_type: str,
    flat_model: str,
    floor_area_sqm: float,
    storey_range: str,
    lease_commence_date: int,
    remaining_lease_years: float,
    txn_year: int,
    txn_month: int,
    base_year: int = 2017,
) -> pd.DataFrame:
    """
    Convenience builder for a single prediction request coming from a
    Streamlit form, where you already have clean numeric inputs rather
    than raw CSV-style strings (e.g. a slider gives remaining lease in
    years directly, rather than a '61 years 04 months' string).
    """
    row = {
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "floor_area_sqm": floor_area_sqm,
        "remaining_lease_years": remaining_lease_years,
        "floor_level": _storey_mid(storey_range),
        "flat_age_at_txn": txn_year - lease_commence_date,
        "months_since_2017": (txn_year - base_year) * 12 + (txn_month - 1),
    }
    return pd.DataFrame([row])[ALL_FEATURES]
