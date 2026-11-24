import unittest
from datetime import date

import joblib
import pandas as pd

from app_logic import (
    InputValidationError,
    PredictionError,
    make_prediction_input,
    monitoring_warning,
    predict_orders,
    validate_incoming_data,
)


class BrokenModel:
    def predict(self, _input):
        raise RuntimeError("broken")


class AppLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("models/pizza_order_pipeline.joblib")

    def test_valid_prediction(self):
        prediction = predict_orders(self.model, date(2026, 1, 10), 19, "family_bundle")
        self.assertGreaterEqual(prediction, 0)

    def test_invalid_hour(self):
        with self.assertRaises(InputValidationError):
            make_prediction_input(date(2026, 1, 10), 24, "none")

    def test_missing_date(self):
        with self.assertRaises(InputValidationError):
            make_prediction_input(None, 19, "none")

    def test_unsupported_promotion(self):
        with self.assertRaises(InputValidationError):
            make_prediction_input(date(2026, 1, 10), 19, "unknown")

    def test_model_error_is_wrapped(self):
        with self.assertRaises(PredictionError):
            predict_orders(BrokenModel(), date(2026, 1, 10), 19, "none")

    def test_incoming_dataset_validates(self):
        data = pd.read_csv("data/incoming_orders.csv")
        clean = validate_incoming_data(data)
        self.assertEqual(len(clean), 4380)

    def test_monitoring_warning(self):
        history = pd.DataFrame({"absolute_error": [2.0] * 24 + [10.0] * 24})
        self.assertIsNotNone(monitoring_warning(history))


if __name__ == "__main__":
    unittest.main()
