# Pizza Rush Predictor

A beginner-friendly model deployment example for predicting hourly pizza orders.
The project trains a scikit-learn pipeline in a Jupyter notebook, saves the fitted
pipeline with Joblib, and loads it in a Streamlit web app.

## What can this project do

- Load and inspect a dataset with pandas.
- Prepare categorical and numeric features.
- Train and evaluate a Random Forest regression model.
- Save preprocessing and the trained model as one reusable pipeline.
- Load the saved pipeline in a Streamlit application for inference.

## Project structure

```text
pizza-rush-predictor/
|-- app.py
|-- pizza_hourly_order_prediction.ipynb
|-- requirements.txt
|-- data/
|   `-- pizza_orders.csv
`-- models/
    `-- pizza_order_pipeline.joblib
```

## Quick start: run the deployed app

The repository includes the trained `models/pizza_order_pipeline.joblib` file, so
you do not need to run the notebook before starting the app.

### 1. Clone the repository

```bash
git clone <repository-url>
cd pizza-rush-predictor
```

### 2. Create a virtual environment

Python 3.10 or newer is recommended.

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py` is unavailable, use `python` instead.

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, `python` refers to the virtual environment on Windows, macOS,
and Linux.

### 4. Start Streamlit

```bash
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`. Open that
address in a browser if it does not open automatically. Stop the app with
`Ctrl+C` in the terminal.

## Optional: retrain the model

Open `pizza_hourly_order_prediction.ipynb` in VS Code, JupyterLab, or another
Jupyter-compatible editor. Select the Python interpreter from this project's
`.venv`, then run all cells from top to bottom.

If your editor does not already provide notebook support, install it in the
active virtual environment:

```bash
python -m pip install jupyter ipykernel
```

The final notebook cell overwrites:

```text
models/pizza_order_pipeline.joblib
```

Restart the Streamlit app after retraining so it loads the new model.

## The Joblib file

The Joblib file contains both parts of the fitted scikit-learn pipeline:

1. The fitted preprocessor that encodes categories and passes through numeric
   features.
2. The fitted Random Forest that predicts the hourly order count.

Including this file demonstrates the separation between **training** and
**deployment**. Training happens in the notebook; the deployed app only loads
the saved artifact and calls `predict()`.

Joblib files are binary artifacts and should only be loaded from trusted
sources. They can also be sensitive to Python and scikit-learn version changes,
so `requirements.txt` constrains the major dependency versions used by this
project. If loading fails after an environment change, rerun the notebook in
that environment to regenerate the model.

## Troubleshooting

**The `python` or `python3` command is not found**

Install Python from [python.org](https://www.python.org/downloads/) and reopen
the terminal. On Windows, also try the `py` command.

**PowerShell blocks virtual-environment activation**

You can skip activation and call the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**The notebook remains stuck on Running**

Restart the notebook kernel and select the interpreter inside `.venv`. Avoid
having both a system Python kernel and the project kernel attached to the same
notebook.

**The saved model cannot be found**

Confirm that `models/pizza_order_pipeline.joblib` was downloaded with the
repository. Alternatively, run the notebook through its final save cell.

## Note

The data is synthetic and the model is intentionally compact. A production
forecasting system would also consider holidays, weather, local events, stock
constraints, monitoring, and scheduled retraining.
