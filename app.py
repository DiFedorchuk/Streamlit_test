from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from train import load_artifact


ARTIFACT_PATH = Path("aussie_rain.joblib")
DEFAULT_DATA_PATH = Path("weather-dataset-rattle-package") / "weatherAUS.csv"
IMAGE_PATH = Path("image") / "weather.jpg"


@st.cache_resource
def get_artifact(path: str) -> dict:
    return load_artifact(path)


def _build_input_form(artifact: dict, reference_df: pd.DataFrame | None) -> dict:
    values: dict = {}
    input_cols = artifact["input_cols"]
    numeric_cols = artifact["numeric_cols"]
    categorical_cols = artifact["categorical_cols"]
    encoder = artifact["encoder"]

    left_col, right_col = st.columns(2)
    for idx, col in enumerate(input_cols):
        target_col = left_col if idx % 2 == 0 else right_col
        with target_col:
            if col in numeric_cols:
                default_val = 0.0
                if reference_df is not None and col in reference_df.columns:
                    series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
                    if not series.empty:
                        default_val = float(series.median())
                values[col] = st.number_input(
                    col,
                    value=float(default_val),
                    step=0.1,
                    format="%.3f",
                )
            elif col in categorical_cols:
                cat_index = categorical_cols.index(col)
                options = [str(x) for x in encoder.categories_[cat_index]]
                values[col] = st.selectbox(col, options=options, index=0)

    return values


def preprocess_user_input(user_input: dict, artifact: dict) -> pd.DataFrame:
    input_cols = artifact["input_cols"]
    numeric_cols = artifact["numeric_cols"]
    categorical_cols = artifact["categorical_cols"]
    encoded_cols = artifact["encoded_cols"]

    input_df = pd.DataFrame([user_input])[input_cols].copy()

    imputed_numeric = pd.DataFrame(
        artifact["imputer"].transform(input_df[numeric_cols]),
        columns=numeric_cols,
        index=input_df.index,
    )
    scaled_numeric = pd.DataFrame(
        artifact["scaler"].transform(imputed_numeric),
        columns=numeric_cols,
        index=input_df.index,
    )
    encoded_categorical = pd.DataFrame(
        artifact["encoder"].transform(input_df[categorical_cols]),
        columns=encoded_cols,
        index=input_df.index,
    )

    return pd.concat([scaled_numeric, encoded_categorical], axis=1)


def predict_with_probability(processed_input: pd.DataFrame, artifact: dict) -> tuple[str, float]:
    model = artifact["model"]
    classes = list(model.classes_)
    probabilities = model.predict_proba(processed_input)[0]

    best_idx = int(np.argmax(probabilities))
    predicted_label = str(classes[best_idx])
    predicted_probability = float(probabilities[best_idx])
    return predicted_label, predicted_probability


def main() -> None:
    st.set_page_config(page_title="Aussie Rain Predictor", layout="wide")
    st.title("Rain Forecast Application")
    st.caption("Enter weather data, preprocess it, and forecast rain for tomorrow.")
    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), use_container_width=True)

    if not ARTIFACT_PATH.exists():
        st.error(
            "Missing aussie_rain.joblib in the current folder. Train or copy it first."
        )
        st.stop()

    artifact = get_artifact(str(ARTIFACT_PATH))

    reference_df = None
    if DEFAULT_DATA_PATH.exists():
        reference_df = pd.read_csv(DEFAULT_DATA_PATH)

    st.subheader("Weather Input")
    st.write("Fill in the weather features, then click the forecast button.")
    input_data = _build_input_form(artifact, reference_df)

    if st.button("Forecast rain", type="primary"):
        try:
            processed_input = preprocess_user_input(input_data, artifact)
            prediction, probability = predict_with_probability(processed_input, artifact)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Forecast failed: {exc}")
            return

        label = "Yes" if prediction == "Yes" else "No"
        st.success(f"Rain tomorrow: {label}")
        st.metric("Forecast probability", f"{probability:.2%}")
        st.caption(
            "Data flow: input -> imputation -> scaling -> categorical encoding -> model prediction"
        )


if __name__ == "__main__":
    # Guard against launching with `python app.py` instead of `streamlit run app.py`.
    if get_script_run_ctx() is None:
        print("This is a Streamlit app.")
        print("Run it with: streamlit run app.py")
    else:
        main()
