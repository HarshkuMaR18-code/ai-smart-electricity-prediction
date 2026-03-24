# State Power Demand Prediction and Generation Optimization

Python system for district-level power demand forecasting, uncertainty-aware planning, statewide aggregation, and generation cost optimization using simulated annealing.

## Features

- Trains **district-specific regression models** with `scikit-learn`
- Uses input features:
  - Temperature
  - Daytime power consumption
  - Nighttime power consumption
- Predicts next-month demand for each district (`x_i`)
- Computes uncertainty buffer (`y_i`) using model RMSE and percentage rule
- Computes district required power: `required_i = x_i + y_i`
- Aggregates statewide demand: `D = sum(required_i)`
- Optimizes generation `P` using **Simulated Annealing**
- Exposes FastAPI endpoint with structured JSON output
- Includes dummy dataset generation for end-to-end execution

## Project Structure

- `power_system.py` - complete pipeline, optimizer, and API
- `requirements.txt` - Python dependencies

## Cost Function

The optimizer minimizes:

`total_cost(P) = generation_cost(P) + storage_cost * max(0, P - D) + shortage_penalty * max(0, D - P)`

Where:
- `generation_cost(P)` is linear + quadratic in this implementation
- `storage_cost` penalizes over-generation (excess power)
- `shortage_penalty` heavily penalizes under-generation

## Core Functions

- `train_model(df)`
- `predict_district(district_name, district_models, temperature, day_power, night_power)`
- `compute_buffer(predicted_demand, rmse, percent=0.10)`
- `optimize_generation(D, config, ...)`
- `run_pipeline(input_features=None, district_names=None)`

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run End-to-End Script

```bash
python3 power_system.py
```

This prints JSON output with:
- district predictions and buffers
- total required state power
- optimal generation and cost

## Run FastAPI Server

```bash
python3 -m uvicorn power_system:app --reload
```

API endpoint:
- `GET /optimize`

Open in browser:
- `http://127.0.0.1:8000/optimize`
- Interactive docs: `http://127.0.0.1:8000/docs`

## Response Format

Example response:

```json
{
  "districts": [
    {
      "name": "District A",
      "predicted_demand": 1200,
      "buffer": 120,
      "required_power": 1320
    },
    {
      "name": "District B",
      "predicted_demand": 900,
      "buffer": 90,
      "required_power": 990
    }
  ],
  "total_required_power": 2310,
  "optimization": {
    "optimal_generation": 2400,
    "excess_power": 90,
    "shortage": 0,
    "total_cost": 12345.67
  }
}
```

## Notes

- The current `GET /optimize` endpoint uses generated historical data and latest profile-based inference inputs.
- Buffers are computed as `max(10% of prediction, RMSE)`.
- You can tune optimizer and cost parameters in `CostConfig`.

## Potential Extensions

- Add `POST /optimize` endpoint for user-provided district inputs
- Persist trained models (joblib/pickle)
- Add model/version monitoring and retraining schedules
- Add tests for prediction and optimization modules
# ai-smart-electricity-prediction
AI-based system to predict electricity demand and ensure fair power distribution using neural networks.
