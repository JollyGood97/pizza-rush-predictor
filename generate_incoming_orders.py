"""Generate a reproducible, synthetic 2026 dataset for monitoring practice."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 2026
SHIFT_START = pd.Timestamp("2026-09-01")
OUTPUT_PATH = Path(__file__).parent / "data" / "incoming_orders.csv"


def choose_promotion(rng: np.random.Generator, hour: int, weekday: int) -> str:
    draw = rng.random()
    if weekday < 5 and 11 <= hour <= 14 and draw < 0.13:
        return "lunch_special"
    if weekday in (4, 5, 6) and 17 <= hour <= 21 and draw < 0.11:
        return "family_bundle"
    if draw < 0.045:
        return "percent_off"
    return "none"


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    hour_factor = {
        11: 0.65, 12: 1.15, 13: 1.28, 14: 0.82, 15: 0.55, 16: 0.62,
        17: 0.90, 18: 1.38, 19: 1.62, 20: 1.48, 21: 1.05, 22: 0.70,
    }
    promotion_factor = {
        "none": 1.00,
        "lunch_special": 1.25,
        "family_bundle": 1.34,
        "percent_off": 1.18,
    }

    for timestamp in pd.date_range("2026-01-01", "2026-12-31", freq="D"):
        is_weekend = int(timestamp.dayofweek >= 5)
        weekday = timestamp.dayofweek
        day_factor = [0.86, 0.88, 0.92, 0.98, 1.15, 1.28, 1.18][weekday]
        seasonal_factor = 1.0 + 0.09 * np.cos(
            2 * np.pi * (timestamp.dayofyear - 355) / 365.25
        )
        holiday_factor = 1.0
        if timestamp.month == 12 and timestamp.day >= 15:
            holiday_factor = 1.16
        elif timestamp.month == 1 and timestamp.day <= 3:
            holiday_factor = 1.10
        for hour in range(11, 23):
            promotion = choose_promotion(rng, hour, weekday)
            expected = (
                17.5
                * day_factor
                * hour_factor[hour]
                * promotion_factor[promotion]
                * seasonal_factor
                * holiday_factor
            )

            # Synthetic concept shift: a nearby competitor closes on 1 September.
            # Demand rises for the same hour/day/promotion conditions from this date.
            if timestamp >= SHIFT_START:
                expected *= 1.45

            noisy_mean = expected * rng.lognormal(mean=-0.5 * 0.13**2, sigma=0.13)
            order_count = max(
                0,
                int(round(rng.normal(noisy_mean, np.sqrt(noisy_mean * 1.25)))),
            )
            rows.append(
                {
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": timestamp.day_name(),
                    "is_weekend": is_weekend,
                    "promotion": promotion,
                    "order_count": order_count,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    incoming = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    incoming.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {len(incoming):,} synthetic rows at {OUTPUT_PATH}")
    print(f"Demand shift begins: {SHIFT_START.date()}")
