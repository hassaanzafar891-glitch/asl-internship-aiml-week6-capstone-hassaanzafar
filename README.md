# Week 6 Capstone — NYC Taxi Payment Predictor (Streamlit Demo)

This is the final capstone project for the Advance Soft Logics AI/ML
internship. It takes the best-performing model from the internship — a
tuned Random Forest classifier that predicts how a NYC taxi ride was paid
for (credit card vs. cash) — and wraps it in an interactive Streamlit demo
that's deployed to a public URL.

**Live demo:** https://asl-internship-aiml-week6-capstone-hassaanzafar-hfjkqknt8qgfgt.streamlit.app
## What this model does

Given a few details about a taxi ride (passengers, distance, fare, tolls,
taxi color, pickup/dropoff borough), the model predicts whether the rider
most likely paid with a **credit card** or **cash**, along with the model's
confidence.

This is the same model (and the same tuned hyperparameters) developed and
compared across Weeks 3-5:

- **Week 3** — first version of this classifier, Logistic Regression vs.
  Random Forest, selected by F1 score.
- **Week 4** — cross-validation, hyperparameter search (`RandomizedSearchCV`),
  and a Gradient Boosting ensemble added to the comparison. The tuned
  Random Forest won.
- **Week 5** — the Week 4 winner served behind a FastAPI `/predict` endpoint.
- **Week 6 (this project)** — the same winning model wrapped in a public,
  interactive Streamlit demo instead of a developer-facing API, so anyone
  can try it without writing code.

## Project structure

```
asl-internship-aiml-week6-capstone/
├── app.py                          # Streamlit UI
├── model_service.py                # Model loading, validation, prediction logic
├── scripts/
│   └── train_and_save_model.py     # Retrains and saves the model
├── models/
│   ├── rf_tuned_v1_capstone.joblib
│   └── rf_tuned_v1_capstone_metadata.json
├── tests/
│   └── test_model_service.py       # Unit tests for the serving logic
├── requirements.txt
└── README.md
```

## Setup (local)

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model (this downloads the dataset and saves the model file —
   it's already included in this repo, but you can re-run it any time):
   ```
   python scripts/train_and_save_model.py
   ```
4. Run the demo locally:
   ```
   streamlit run app.py
   ```
   It opens at `http://localhost:8501`.

## Example inputs to try

| Scenario | Passengers | Distance | Fare | Tolls | Color | Pickup | Dropoff |
|---|---|---|---|---|---|---|---|
| Short Manhattan hop | 1 | 1.2 | 7.50 | 0 | yellow | Manhattan | Manhattan |
| Airport run with tolls | 2 | 15.0 | 45.00 | 5.50 | yellow | Manhattan | EWR |
| Outer-borough ride | 1 | 4.5 | 16.00 | 0 | green | Brooklyn | Queens |

Fill these into the form and click **Predict payment method** to see the
prediction and the model's confidence.

## Model details

- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** `n_estimators=249, max_depth=11, min_samples_split=4,
  class_weight=None` (found via `RandomizedSearchCV` in Week 4)
- **Data:** [NYC Taxis dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv)
  (seaborn-data), 6,433 rides
- **Features used:** `passengers`, `distance`, `fare`, `tolls` (numeric),
  `color`, `pickup_borough`, `dropoff_borough` (categorical)
- **Target:** `payment` (credit card / cash)
- **Excluded columns:** `tip` and `total` are excluded on purpose — they
  leak the answer, since cash tips are rarely recorded, so a model trained
  on them would basically be looking up the answer instead of predicting it.
- **Test set performance:** ~73.5% accuracy, F1 = 0.837 (see
  `models/rf_tuned_v1_capstone_metadata.json` for the exact numbers from
  the last training run)

## Input validation

The form only lets you submit values the model can actually handle
(passenger count between 1 and 6, non-negative fare/distance/tolls, a
real taxi color, a real borough), but the validation also lives
independently in `model_service.py` so it's enforced even if this were
called from somewhere other than the Streamlit form. Invalid input raises
a clear `InvalidInputError` with a message naming the exact field that's
wrong, instead of crashing or silently producing a bad prediction.

## Logging

Every prediction made through the demo is appended as one JSON line to
`logs/predictions.log` (input, prediction, probability, timestamp, model
version), for basic monitoring of what the model is being asked to do.

## Tests

Run the test suite with:
```
pytest tests/ -v
```
It covers: the model loading correctly, valid input passing validation,
and each category of bad input (missing field, wrong type, negative
number, out-of-range passenger count, invalid category) being rejected
with a clear error — plus that a full prediction returns the expected
shape (prediction, probability, model version).

## Deploying to a public URL (Streamlit Community Cloud)

1. Push this repository to a **public** GitHub repo named
   `asl-internship-aiml-week6-capstone-<yourname>`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
3. Click **New app**, pick this repository and branch (`main`), and set
   the main file path to `app.py`.
4. Click **Deploy**. Streamlit Community Cloud installs everything from
   `requirements.txt` and starts the app automatically.
5. Once it's live, copy the public URL it gives you (something like
   `https://<yourname>-asl-internship-aiml-week6-capstone.streamlit.app`)
   and paste it at the top of this README, plus into your capstone report.
6. Open the live link yourself and run a prediction to confirm it actually
   works before submitting — a link that doesn't return a prediction isn't
   a finished submission.

## Limitations and next steps

See `Capstone_Reflection.docx` / the capstone report for the full writeup
on this model's limitations and what I'd try next with more time.
