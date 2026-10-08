import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Boston Housing | ML Demonstration",
    page_icon="🏠",
    layout="wide",
)


# ============================================================
# 2. LOAD TRAINED MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_assets():
    model_file = BASE / "boston_housing_model_v1_0.joblib"
    meta_file = BASE / "model_metadata.json"

    if not model_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            "Deployment artifacts missing. "
            "Train and export the notebook first."
        )

    model = joblib.load(model_file)

    meta = json.loads(
        meta_file.read_text(encoding="utf-8")
    )

    # Validate feature schema
    if list(model.feature_names_in_) != meta["feature_names"]:
        raise ValueError(
            "Model feature schema does not match metadata."
        )

    return model, meta


# ============================================================
# 3. PAGE HEADER
# ============================================================

st.title("🏠 Boston Housing Price Prediction")

st.caption(
    "Historical educational dataset | "
    "XGBoost Regression | Inference Only"
)

st.warning(
    "Educational demonstration using historical census-era data. "
    "This application is not a current property valuation or "
    "lending tool. Some features may encode sensitive "
    "socioeconomic proxies."
)


# ============================================================
# 4. INITIALISE MODEL
# ============================================================

try:
    model, meta = load_assets()

except Exception as exc:
    st.error(f"Could not load application artifacts: {exc}")
    st.stop()


# ============================================================
# 5. MODEL INFORMATION
# ============================================================

with st.expander("Model Information", expanded=False):
    st.json({
        "model": meta["model_name"],
        "target_units": meta["target_units"],
        "test_metrics": meta["metrics"],
        "trained_features": meta["feature_names"],
    })


# ============================================================
# 6. USER INPUT FORM
# ============================================================

st.subheader("Property and Neighbourhood Features")

st.write(
    "Enter features in their original dataset units. "
    "Default values are based on the training data."
)

values = {}

with st.form("predict_form"):

    cols = st.columns(3)

    for ix, name in enumerate(meta["feature_names"]):

        with cols[ix % 3]:

            categorical_options = meta.get(
                "categorical_options", {}
            )

            if name in categorical_options:

                options = categorical_options[name]

                default_value = str(
                    meta["feature_defaults"][name]
                )

                default_index = (
                    options.index(default_value)
                    if default_value in options
                    else 0
                )

                values[name] = st.selectbox(
                    name,
                    options,
                    index=default_index,
                )

            else:

                default_value = float(
                    meta["feature_defaults"][name]
                )

                if name in ("CHAS", "RAD", "TAX"):

                    values[name] = st.number_input(
                        name,
                        value=default_value,
                        step=1.0,
                        format="%.0f",
                    )

                else:

                    values[name] = st.number_input(
                        name,
                        value=default_value,
                        format="%.5f",
                    )

    submitted = st.form_submit_button(
        "Predict Historical Median Home Value",
        type="primary",
    )


# ============================================================
# 7. RUN PREDICTION
# ============================================================

if submitted:

    input_df = pd.DataFrame(
        [
            {
                name: values[name]
                for name in meta["feature_names"]
            }
        ],
        columns=meta["feature_names"],
    )

    # Preserve categorical feature types
    for name in meta.get("categorical_options", {}):
        input_df[name] = input_df[name].astype(str)

    try:

        prediction = float(
            model.predict(input_df)[0]
        )

        predicted_usd = prediction * 1000

        st.success("Prediction completed successfully.")

        st.metric(
            label="Estimated MEDV (Historical USD)",
            value=f"${predicted_usd:,.0f}",
        )

        st.caption(
            f"Model prediction: ${prediction:.2f} thousand "
            "in historical dataset units. "
            "Not inflation-adjusted."
        )

        with st.expander("Prediction Input Audit"):

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )


# ============================================================
# 8. APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "Source: Boston Housing research dataset. "
    "Predictions are for educational purposes only "
    "and should not be used for consequential decisions."
)
