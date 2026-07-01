"""
Minimal example of wiring hdb_price_model.joblib into a Streamlit app.
Merge the relevant bits into your existing app.py — this isn't meant to
replace whatever UI/layout you already have.

Run with: streamlit run app_example.py
"""

import datetime

import streamlit as st
from predict import load_artifact, predict_price

st.title("HDB Resale Price Estimator")

artifact = load_artifact()
metrics = artifact["held_out_eval_metrics"]
st.caption(
    f"Model tested on the most recent 6 months of transactions it never trained on: "
    f"typically **±S${metrics['MAE']:,.0f}** ({metrics['MAPE']:.1%}), R² = {metrics['R2']:.3f}."
)

col1, col2 = st.columns(2)
with col1:
    town = st.selectbox("Town", sorted([
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
        "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
        "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
        "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
        "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
        "TOA PAYOH", "WOODLANDS", "YISHUN",
    ]))
    flat_type = st.selectbox(
        "Flat type", ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
    )
    flat_model = st.text_input("Flat model (e.g. Model A, Improved, DBSS)", value="Model A")
    floor_area_sqm = st.number_input("Floor area (sqm)", min_value=20.0, max_value=250.0, value=90.0)

with col2:
    storey_range = st.selectbox(
        "Storey range", ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18", "19 TO 21"]
    )
    lease_commence_date = st.number_input("Lease commence year", min_value=1960, max_value=2026, value=1990)
    remaining_lease_years = st.number_input("Remaining lease (years)", min_value=0.0, max_value=99.0, value=63.0)
    today = datetime.date.today()
    txn_year = st.number_input("Transaction year", min_value=2017, max_value=2030, value=today.year)
    txn_month = st.number_input("Transaction month", min_value=1, max_value=12, value=today.month)

if st.button("Estimate price", type="primary"):
    price = predict_price(
        town=town,
        flat_type=flat_type,
        flat_model=flat_model,
        floor_area_sqm=floor_area_sqm,
        storey_range=storey_range,
        lease_commence_date=int(lease_commence_date),
        remaining_lease_years=remaining_lease_years,
        txn_year=int(txn_year),
        txn_month=int(txn_month),
    )
    st.metric("Estimated resale price", f"S${price:,.0f}")
    st.caption(f"Expect this to be off by roughly ±S${metrics['MAE']:,.0f} on average.")
