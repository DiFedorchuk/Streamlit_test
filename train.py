from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


DEFAULT_DATA_PATH = Path("weather-dataset-rattle-package") / "weatherAUS.csv"
DEFAULT_ARTIFACT_PATH = Path("aussie_rain.joblib")


def load_artifact(artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict:
    return joblib.load(artifact_path)


def _validate_required_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _transform_inputs(
    inputs_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoded_cols: list[str],
    imputer: SimpleImputer,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    imputed_numeric = pd.DataFrame(
        imputer.transform(inputs_df[numeric_cols]),
        columns=numeric_cols,
        index=inputs_df.index,
    )
    scaled_numeric = pd.DataFrame(
        scaler.transform(imputed_numeric),
        columns=numeric_cols,
        index=inputs_df.index,
    )
    encoded = pd.DataFrame(
        encoder.transform(inputs_df[categorical_cols]),
        columns=encoded_cols,
        index=inputs_df.index,
    )
    return pd.concat([scaled_numeric, encoded], axis=1)


def retrain_from_dataframe(
    new_df: pd.DataFrame,
    base_artifact: dict,
    random_state: int = 42,
) -> tuple[dict, dict]:
    input_cols = list(base_artifact["input_cols"])
    target_col = str(base_artifact.get("target_col", "RainTomorrow"))
    numeric_cols = list(base_artifact["numeric_cols"])
    categorical_cols = list(base_artifact["categorical_cols"])

    _validate_required_columns(new_df, input_cols + [target_col], "New training data")

    model_df = new_df.dropna(subset=["RainToday", target_col]).copy()
    if len(model_df) < 100:
        raise ValueError(
            f"Not enough usable rows after dropping missing target rows: {len(model_df)}"
        )

    train_val_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=random_state,
        stratify=model_df[target_col],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=random_state,
        stratify=train_val_df[target_col],
    )

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    test_inputs = test_df[input_cols].copy()

    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()
    test_targets = test_df[target_col].copy()

    imputer = SimpleImputer(strategy="mean").fit(train_inputs[numeric_cols])
    imputed_train_numeric = pd.DataFrame(
        imputer.transform(train_inputs[numeric_cols]),
        columns=numeric_cols,
        index=train_inputs.index,
    )
    scaler = MinMaxScaler().fit(imputed_train_numeric)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train_inputs[categorical_cols]
    )
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    x_train = _transform_inputs(
        train_inputs,
        numeric_cols,
        categorical_cols,
        encoded_cols,
        imputer,
        scaler,
        encoder,
    )
    x_val = _transform_inputs(
        val_inputs,
        numeric_cols,
        categorical_cols,
        encoded_cols,
        imputer,
        scaler,
        encoder,
    )
    x_test = _transform_inputs(
        test_inputs,
        numeric_cols,
        categorical_cols,
        encoded_cols,
        imputer,
        scaler,
        encoder,
    )

    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(x_train, train_targets)

    train_preds = model.predict(x_train)
    val_preds = model.predict(x_val)
    test_preds = model.predict(x_test)

    metrics = {
        "rows_used": int(len(model_df)),
        "train_accuracy": float(accuracy_score(train_targets, train_preds)),
        "val_accuracy": float(accuracy_score(val_targets, val_preds)),
        "test_accuracy": float(accuracy_score(test_targets, test_preds)),
        "train_f1_yes": float(f1_score(train_targets, train_preds, pos_label="Yes")),
        "val_f1_yes": float(f1_score(val_targets, val_preds, pos_label="Yes")),
        "test_f1_yes": float(f1_score(test_targets, test_preds, pos_label="Yes")),
    }

    new_artifact = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "input_cols": input_cols,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_cols": encoded_cols,
    }
    return new_artifact, metrics


def save_artifact(artifact: dict, out_path: str | Path = DEFAULT_ARTIFACT_PATH) -> None:
    joblib.dump(artifact, out_path)


def predict_input(single_input: dict, artifact: dict) -> tuple[str, float]:
    numeric_cols = artifact["numeric_cols"]
    categorical_cols = artifact["categorical_cols"]
    encoded_cols = artifact["encoded_cols"]
    input_cols = artifact["input_cols"]

    input_df = pd.DataFrame([single_input])
    _validate_required_columns(input_df, input_cols, "Prediction input")
    input_df = input_df[input_cols].copy()

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
    encoded = pd.DataFrame(
        artifact["encoder"].transform(input_df[categorical_cols]),
        columns=encoded_cols,
        index=input_df.index,
    )

    x_input = pd.concat([scaled_numeric, encoded], axis=1)
    pred = artifact["model"].predict(x_input)[0]
    prob = artifact["model"].predict_proba(x_input)[0][
        list(artifact["model"].classes_).index(pred)
    ]
    return str(pred), float(prob)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain Aussie rain logistic regression model on a new CSV dataset."
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV file with weather data.",
    )
    parser.add_argument(
        "--artifact-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to existing aussie_rain.joblib (schema source).",
    )
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Output path for retrained joblib artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_artifact = load_artifact(args.artifact_path)
    new_df = pd.read_csv(args.data_path)
    new_artifact, metrics = retrain_from_dataframe(new_df, base_artifact)
    save_artifact(new_artifact, args.out_path)

    print(f"Saved retrained artifact to: {args.out_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
