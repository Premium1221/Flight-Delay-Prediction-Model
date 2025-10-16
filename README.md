# Flight Delay Prediction Model & Web App

An end‑to‑end, local Flask web app that predicts whether a flight will depart with a 15+ minute delay based on airline, route, date/time, and distance. The app loads a trained model if available and gracefully falls back to a consistent demo predictor when no model is present.

## Key Features
- Interactive web form to enter flight details (airline, origin, destination, date, time, distance)
- Model‑backed prediction of “Delayed (>= 15 min)” vs “On‑time (< 15 min)”
- Probability and an estimated delay duration when delayed
- Model information page with feature list and (if supported) feature importances
- Safe fallback “dummy” mode with consistent, deterministic demo predictions

## Project Structure
- `app.py` — Main Flask app with routes: `/`, `/predict`, `/model-info`
- `model_loader.py` — Utility to load models and lightweight preprocessing (optional path)
- `run.py` — Convenience script to start the Flask app
- `templates/` — HTML templates (`index.html`, `result.html`, `model_info.html`, `error.html`)
- `static/` — Front‑end assets (CSS/JS)
- `requirements.txt` — Python dependencies
- `FD_iteration_one.ipynb`, `FD_iteration_TWO.ipynb` — Training/experimentation notebooks
- `data_dictionarymd.md` — Dataset field descriptions
- Model artifacts (optional): `Flight Delay Prediction Model.pkl`, `flight_delay_model.pkl`, `model_features.pkl`, etc.

## Requirements
- Python 3.10+ recommended (tested with 3.12)
- Install dependencies:
  - `pip install -r requirements.txt`
  - Includes: Flask, pandas, numpy, scikit‑learn, joblib

## Quick Start
1) Create and activate a virtual environment
- Windows (PowerShell)
  - `py -m venv .venv`
  - `.\\.venv\\Scripts\\Activate.ps1`
- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

2) Install dependencies
- `pip install -r requirements.txt`

3) Start the app
- Option A: `python run.py`
- Option B: `python app.py`
- App runs at `http://127.0.0.1:5000`

## Using the App
- Open `http://127.0.0.1:5000`
- Enter:
  - Airline (e.g., AA, DL, UA, WN, B6, AS, NK, F9)
  - Origin and destination IATA codes (e.g., DFW → LAX)
  - Flight date (auto‑computes DayOfWeek/Month behind the scenes)
  - Scheduled departure time (HH:MM)
  - Distance (miles)
- Submit to view:
  - Predicted status (Delayed or On‑time)
  - Probability and, if delayed, an estimated delay duration
  - Echo of input features; feature importances if available
- Click “View Model Information” for model status, features and importances.

## Model Files and Expected Inputs
The app attempts to auto‑load a trained model from any of these files in the project root:
- `Flight Delay Prediction Model.pkl`
- `flight_delay_model.pkl`
- `model.pkl`
- `FD_model.pkl`

Optional support files (if you have them):
- `model_features.pkl` — List of model input feature names used for inference and importance display
- `encoders.pkl`, `scalers.pkl`, `feature_names.pkl` — If you use `model_loader.py` for a custom pipeline

If no model is found or if dependencies are missing, the app runs in a deterministic “dummy” mode for demo purposes. The UI clearly indicates when dummy mode is active.

### Default Feature List (app fallback)
If `model_features.pkl` is not provided, the app falls back to this 16‑feature list for inference formatting and display:
- `AIRLINE`, `AIRLINE_DOT`, `AIRLINE_CODE`, `DOT_CODE`, `ORIGIN`, `ORIGIN_CITY`, `DEST`, `DEST_CITY`, `CRS_DEP_TIME`, `DEP_DELAY`, `CRS_ELAPSED_TIME`, `DISTANCE`, `ROUTE`, `DEP_HOUR`, `DAY_OF_WEEK`, `MONTH`

Notes:
- The web form collects a subset directly (airline, origin, destination, date/time, distance). The app supplies other fields via deterministic mappings and estimates.
- For demonstration, `DEP_DELAY` is synthesized internally to mimic patterns. Do not include realized delay fields when training real‑world predictive models, as that constitutes data leakage.

## Programmatic Usage (cURL)
You can post the same fields the form sends to `/predict` using standard form encoding:

```
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "airline=DL" \
  -d "origin=ATL" \
  -d "dest=LAX" \
  -d "flight_date=2025-06-15" \
  -d "scheduled_dep_time=14:30" \
  -d "distance=1950" \
  -d "day_of_week=7" \
  -d "month=6"
```

## Training and Exporting a Model
See the notebooks:
- `FD_iteration_one.ipynb`
- `FD_iteration_TWO.ipynb`

General guidance to persist your model for the app:
- Fit your pipeline/model on your training data
- Export with joblib/pickle and place the file in the project root:

```python
from joblib import dump
# model is your trained estimator or pipeline
dump(model, "Flight Delay Prediction Model.pkl")

# Optionally store the ordered list of input feature names used at inference
dump(feature_names, "model_features.pkl")
```

If your model exposes `feature_importances_` and you provide `model_features.pkl` with matching order/length, the Model Info and Results pages will display feature importances.

## Routes
- `/` — Home page with the prediction form
- `/predict` — POST endpoint to score a single flight from form data
- `/model-info` — Model status, features, and feature importances (if available)

## Data Dictionary
A concise field reference is available in `data_dictionarymd.md`.

## Troubleshooting
- “Model file not found” or dummy mode is active:
  - Ensure a model file named as listed above is in the project root
  - Verify the environment matches the model’s dependencies
- Feature importances missing:
  - Your model may not implement `feature_importances_`, or feature count/order may not match `model_features.pkl`
- Port already in use:
  - Edit `run.py` to change `port`, or stop the process using port 5000
- Import errors:
  - Activate your venv and run `pip install -r requirements.txt`

## Notes and Limitations
- For demonstration, the app can operate without a trained model (dummy mode). Do not rely on dummy predictions for operational decisions.
- Avoid data leakage when training: exclude realized delay fields (e.g., `DEP_DELAY`, `ARR_DELAY`) from features used to predict pre‑departure delays.
- The included notebooks illustrate experimentation paths; align final preprocessing with what you serialize for inference.

## License
This project is provided for educational purposes within your course context. If you plan to publish or open‑source it, add an explicit license file.

