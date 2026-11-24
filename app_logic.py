"""Shared validation, prediction, testing, and monitoring helpers."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


PROMOTION_LABELS = {
    "No promotion": "none",
    "Lunch special": "lunch_special",
    "Family bundle": "family_bundle",
    "Percent off": "percent_off",
}
SUPPORTED_PROMOTIONS = set(PROMOTION_LABELS.values())
EXPECTED_INCOMING_COLUMNS = [
    "date",
    "hour",
    "day_of_week",
    "is_weekend",
    "promotion",
    "order_count",
]
OPENING_HOUR = 11
CLOSING_HOUR = 22
SHIFT_START = pd.Timestamp("2026-09-01")
ROLLING_WINDOW = 24


class InputValidationError(ValueError):
    """An input cannot safely be passed to the trained model."""


class PredictionError(RuntimeError):
    """The fitted model could not produce a prediction."""


def make_prediction_input(
    selected_date: date | pd.Timestamp | str | None,
    hour: int,
    promotion: str,
) -> pd.DataFrame:
    """Validate one request and build the seven features used during training."""
    if selected_date is None or pd.isna(selected_date):
        raise InputValidationError("A date is required.")

    try:
        timestamp = pd.Timestamp(selected_date)
    except (TypeError, ValueError) as error:
        raise InputValidationError("The date is not valid.") from error

    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)

    if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
        raise InputValidationError("Hour must be a whole number from 0 to 23.")

    if promotion not in SUPPORTED_PROMOTIONS:
        supported = ", ".join(sorted(SUPPORTED_PROMOTIONS))
        raise InputValidationError(f"Unsupported promotion. Use one of: {supported}.")

    return pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": timestamp.day_name(),
                "is_weekend": int(timestamp.dayofweek >= 5),
                "promotion": promotion,
                "month": timestamp.month,
                "day_of_month": timestamp.day,
                "day_of_year": timestamp.dayofyear,
            }
        ]
    )


def predict_orders(
    model: Any,
    selected_date: date | pd.Timestamp | str | None,
    hour: int,
    promotion: str,
) -> float:
    """Return a non-negative prediction while presenting model failures safely."""
    input_row = make_prediction_input(selected_date, hour, promotion)
    try:
        raw_prediction = model.predict(input_row)
        prediction = float(raw_prediction[0])
    except Exception as error:
        raise PredictionError(
            "The model could not make a prediction. Check the model file and input data."
        ) from error

    if pd.isna(prediction):
        raise PredictionError("The model returned an empty prediction.")
    return max(0.0, prediction)


def validate_incoming_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded production-simulation dataset."""
    missing = [column for column in EXPECTED_INCOMING_COLUMNS if column not in data.columns]
    extra = [column for column in data.columns if column not in EXPECTED_INCOMING_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra)}")
        raise InputValidationError("CSV columns are invalid (" + "; ".join(details) + ").")
    if data.empty:
        raise InputValidationError("The CSV contains no records.")

    clean = data.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    if clean["date"].isna().any():
        raise InputValidationError("Every row must contain a valid date.")
    if not clean["date"].dt.year.eq(2026).all():
        raise InputValidationError("This simulation expects synthetic records from 2026.")

    numeric_hour = pd.to_numeric(clean["hour"], errors="coerce")
    if numeric_hour.isna().any() or not numeric_hour.eq(numeric_hour.round()).all():
        raise InputValidationError("Every hour must be a whole number.")
    clean["hour"] = numeric_hour.astype(int)
    if not clean["hour"].between(OPENING_HOUR, CLOSING_HOUR).all():
        raise InputValidationError(
            f"Hours must be between {OPENING_HOUR}:00 and {CLOSING_HOUR}:00."
        )

    if clean["promotion"].isna().any() or not clean["promotion"].isin(SUPPORTED_PROMOTIONS).all():
        invalid = sorted(clean.loc[~clean["promotion"].isin(SUPPORTED_PROMOTIONS), "promotion"].astype(str).unique())
        raise InputValidationError(f"Unsupported promotion values: {', '.join(invalid)}.")

    weekend = pd.to_numeric(clean["is_weekend"], errors="coerce")
    if weekend.isna().any() or not weekend.isin([0, 1]).all():
        raise InputValidationError("is_weekend must contain only 0 or 1.")
    clean["is_weekend"] = weekend.astype(int)

    order_count = pd.to_numeric(clean["order_count"], errors="coerce")
    if order_count.isna().any() or (order_count < 0).any():
        raise InputValidationError("order_count must contain non-negative numbers.")
    clean["order_count"] = order_count

    expected_days = clean["date"].dt.day_name()
    if not clean["day_of_week"].eq(expected_days).all():
        raise InputValidationError("day_of_week does not match the date in one or more rows.")
    expected_weekend = clean["date"].dt.dayofweek.ge(5).astype(int)
    if not clean["is_weekend"].eq(expected_weekend).all():
        raise InputValidationError("is_weekend does not match the date in one or more rows.")

    clean = clean.sort_values(["date", "hour"]).reset_index(drop=True)
    if clean.duplicated(["date", "hour"]).any():
        raise InputValidationError("Each date and hour combination must be unique.")
    return clean


class _FailingModel:
    def predict(self, _input: pd.DataFrame) -> list[float]:
        raise RuntimeError("Deliberate test failure")


def run_testing_lab(model: Any) -> pd.DataFrame:
    """Run five beginner-friendly checks and return their outcomes."""
    cases: list[dict[str, str]] = []

    try:
        value = predict_orders(model, date(2026, 1, 10), 19, "family_bundle")
        passed = value >= 0
        detail = f"The model returned a valid non-negative prediction ({value:.1f} orders)."
    except Exception as error:
        passed, detail = False, f"Unexpected error: {error}"
    cases.append({"Test": "Valid inputs", "Result": "PASS" if passed else "FAIL", "Explanation": detail})

    validation_cases = [
        ("Invalid hour", date(2026, 1, 10), 24, "none", "The app rejected hour 24 before calling the model."),
        ("Missing date", None, 19, "none", "The app required a date before calling the model."),
        (
            "Unsupported promotion",
            date(2026, 1, 10),
            19,
            "buy_one_get_one",
            "The app rejected a promotion category the model does not support.",
        ),
    ]
    for name, test_date, hour, promotion, success_detail in validation_cases:
        try:
            predict_orders(model, test_date, hour, promotion)
        except InputValidationError:
            passed, detail = True, success_detail
        except Exception as error:
            passed, detail = False, f"The wrong error was raised: {error}"
        else:
            passed, detail = False, "The invalid input was not rejected."
        cases.append({"Test": name, "Result": "PASS" if passed else "FAIL", "Explanation": detail})

    try:
        predict_orders(_FailingModel(), date(2026, 1, 10), 19, "none")
    except PredictionError as error:
        passed = True
        detail = f"The model failure was caught safely: {error}"
    except Exception as error:
        passed, detail = False, f"An unhandled error escaped: {error}"
    else:
        passed, detail = False, "The deliberate model failure was not detected."
    cases.append({"Test": "Prediction error", "Result": "PASS" if passed else "FAIL", "Explanation": detail})

    return pd.DataFrame(cases)


def monitoring_warning(history: pd.DataFrame, window: int = ROLLING_WINDOW) -> str | None:
    """Warn when the latest error window is materially worse than the prior window."""
    if len(history) < window * 2:
        return None
    previous_mae = history["absolute_error"].iloc[-window * 2 : -window].mean()
    recent_mae = history["absolute_error"].iloc[-window:].mean()
    if recent_mae >= previous_mae * 1.75 and recent_mae - previous_mae >= 4:
        return (
            f"Recent errors are increasing: rolling MAE is {recent_mae:.2f}, "
            f"compared with {previous_mae:.2f} in the previous {window} hours."
        )
    return None
