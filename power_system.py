from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class CostConfig:
    linear_cost: float = 2.5
    quadratic_cost: float = 0.001
    storage_cost: float = 1.2
    shortage_penalty: float = 25.0


def generate_dummy_dataset(
    districts: List[str],
    months_per_district: int = 48,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic historical data for each district."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | str]] = []

    for district in districts:
        district_base = rng.uniform(700, 1400)
        temp_offset = rng.uniform(-3, 3)
        day_factor = rng.uniform(1.1, 1.5)
        night_factor = rng.uniform(0.7, 1.0)

        prev_target = district_base
        for month_idx in range(months_per_district):
            seasonal = 8 * np.sin((2 * np.pi * month_idx) / 12.0)
            temperature = rng.uniform(18, 40) + seasonal + temp_offset
            day_consumption = district_base * day_factor + 14 * temperature + rng.normal(0, 30)
            night_consumption = district_base * night_factor + 8 * temperature + rng.normal(0, 20)

            target = (
                0.22 * temperature
                + 0.52 * day_consumption
                + 0.34 * night_consumption
                + 0.08 * prev_target
                + rng.normal(0, 25)
            )
            prev_target = target

            rows.append(
                {
                    "district": district,
                    "temperature": float(temperature),
                    "day_power": float(day_consumption),
                    "night_power": float(night_consumption),
                    "next_month_consumption": float(max(target, 100.0)),
                }
            )

    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Train one regression model per district.
    Returns: {district_name: {"model": fitted_model, "rmse": validation_rmse}}
    """
    district_models: Dict[str, Dict[str, object]] = {}
    feature_cols = ["temperature", "day_power", "night_power"]
    target_col = "next_month_consumption"

    for district_name, district_df in df.groupby("district"):
        X = district_df[feature_cols]
        y = district_df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_split=3,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        district_models[district_name] = {"model": model, "rmse": rmse}

    return district_models


def predict_district(
    district_name: str,
    district_models: Dict[str, Dict[str, object]],
    temperature: float,
    day_power: float,
    night_power: float,
) -> float:
    """Predict next-month demand x_i for a single district."""
    model = district_models[district_name]["model"]
    x_input = pd.DataFrame(
        [{"temperature": temperature, "day_power": day_power, "night_power": night_power}]
    )
    prediction = float(model.predict(x_input)[0])
    return max(prediction, 0.0)


def compute_buffer(
    predicted_demand: float,
    rmse: float,
    percent: float = 0.10,
) -> float:
    """
    Compute y_i uncertainty buffer using max of:
      - percentage of prediction
      - model RMSE
    """
    return float(max(predicted_demand * percent, rmse))


def total_cost(P: float, D: float, config: CostConfig) -> float:
    """Cost function used by the optimizer."""
    generation = config.linear_cost * P + config.quadratic_cost * (P**2)
    storage = config.storage_cost * max(0.0, P - D)
    shortage = config.shortage_penalty * max(0.0, D - P)
    return float(generation + storage + shortage)


def optimize_generation(
    D: float,
    config: CostConfig,
    initial_temp: float = 300.0,
    cooling_rate: float = 0.985,
    min_temp: float = 1e-3,
    iterations_per_temp: int = 80,
    max_step: float = 120.0,
    seed: int = 42,
) -> Tuple[float, float]:
    """Simulated annealing local search to find optimal generation P."""
    random.seed(seed)
    current_P = max(D, 0.0)
    current_cost = total_cost(current_P, D, config)
    best_P = current_P
    best_cost = current_cost
    temp = initial_temp

    while temp > min_temp:
        for _ in range(iterations_per_temp):
            delta = random.uniform(-max_step, max_step)
            neighbor_P = max(0.0, current_P + delta)
            neighbor_cost = total_cost(neighbor_P, D, config)
            cost_diff = neighbor_cost - current_cost

            if cost_diff < 0:
                accept = True
            else:
                accept_prob = np.exp(-cost_diff / max(temp, 1e-12))
                accept = random.random() < float(accept_prob)

            if accept:
                current_P = neighbor_P
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_P = current_P
                    best_cost = current_cost

        temp *= cooling_rate

    return float(best_P), float(best_cost)


def run_pipeline(
    input_features: List[Dict[str, float | str]] | None = None,
    district_names: List[str] | None = None,
) -> Dict[str, object]:
    """
    End-to-end:
      1) train district models
      2) predict district demand
      3) add uncertainty buffer
      4) aggregate demand
      5) optimize generation
    """
    if district_names is None:
        district_names = ["District A", "District B", "District C", "District D"]

    historical_df = generate_dummy_dataset(district_names)
    models = train_model(historical_df)

    if input_features is None:
        # Use latest observed feature profile from each district as inference input
        input_features = []
        for name in district_names:
            latest_row = historical_df[historical_df["district"] == name].iloc[-1]
            input_features.append(
                {
                    "name": name,
                    "temperature": float(latest_row["temperature"]),
                    "day_power": float(latest_row["day_power"]),
                    "night_power": float(latest_row["night_power"]),
                }
            )

    district_outputs: List[Dict[str, float | str]] = []
    total_required_power = 0.0
    for item in input_features:
        name = str(item["name"])
        pred = predict_district(
            district_name=name,
            district_models=models,
            temperature=float(item["temperature"]),
            day_power=float(item["day_power"]),
            night_power=float(item["night_power"]),
        )
        rmse = float(models[name]["rmse"])
        buffer_val = compute_buffer(predicted_demand=pred, rmse=rmse)
        required = pred + buffer_val
        total_required_power += required

        district_outputs.append(
            {
                "name": name,
                "predicted_demand": round(pred, 2),
                "buffer": round(buffer_val, 2),
                "required_power": round(required, 2),
            }
        )

    config = CostConfig()
    optimal_P, minimum_cost = optimize_generation(D=total_required_power, config=config)

    excess_power = max(0.0, optimal_P - total_required_power)
    shortage = max(0.0, total_required_power - optimal_P)

    result = {
        "districts": district_outputs,
        "total_required_power": round(total_required_power, 2),
        "optimization": {
            "optimal_generation": round(optimal_P, 2),
            "excess_power": round(excess_power, 2),
            "shortage": round(shortage, 2),
            "total_cost": round(minimum_cost, 2),
        },
    }
    return result


app = FastAPI(title="State Power Planning API", version="1.0.0")


@app.get("/optimize")
def optimize_endpoint() -> Dict[str, object]:
    """Return optimized generation plan using generated demo data."""
    return run_pipeline()


if __name__ == "__main__":
    import json

    output = run_pipeline()
    print(json.dumps(output, indent=2))
