from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_logic import (
    CLOSING_HOUR, DATA_DRIFT_START, EXPECTED_INCOMING_COLUMNS, OPENING_HOUR, PROMOTION_LABELS,
    ROLLING_WINDOW, SHIFT_START, InputValidationError, PredictionError,
    monitoring_warning, predict_orders, run_testing_lab, validate_incoming_data,
)

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "pizza_order_pipeline.joblib"
INCOMING_PATH = PROJECT_DIR / "data" / "incoming_orders.csv"

try:
    shop_name = st.secrets.get("SHOP_NAME", "Pizza Rush Predictor")
except FileNotFoundError:
    shop_name = "Pizza Rush Predictor"


@st.cache_resource
def load_model():
    """Load the existing fitted pipeline once; never train inside this app."""
    return joblib.load(MODEL_PATH)


def reset_monitoring_state(source_key: str | None = None) -> None:
    st.session_state.monitor_source_key = source_key
    st.session_state.monitor_index = 0
    st.session_state.monitor_history = []
    st.session_state.monitor_playing = False


def initialize_state() -> None:
    st.session_state.setdefault("testing_results", None)
    st.session_state.setdefault("monitor_source_key", None)
    st.session_state.setdefault("monitor_index", 0)
    st.session_state.setdefault("monitor_history", [])
    st.session_state.setdefault("monitor_playing", False)


def render_predict_tab(model) -> None:
    st.header("Predict the next hour")
    st.write("Choose a valid date, opening hour, and supported promotion.")
    with st.form("prediction_form"):
        selected_date = st.date_input("Date", value=date.today())
        selected_hour = st.slider(
            "Hour of day", min_value=OPENING_HOUR, max_value=CLOSING_HOUR, value=19
        )
        selected_promotion_label = st.selectbox("Promotion", options=list(PROMOTION_LABELS))
        submitted = st.form_submit_button(
            "Predict orders", type="primary", icon=":material/auto_awesome:", width="stretch"
        )
    if submitted:
        try:
            prediction = predict_orders(
                model, selected_date, selected_hour, PROMOTION_LABELS[selected_promotion_label]
            )
        except (InputValidationError, PredictionError) as error:
            st.error(str(error))
        except Exception:
            st.error("Prediction failed unexpectedly. Please check the model and try again.")
        else:
            st.metric("Predicted hourly orders", f"{round(prediction)} pizzas", border=True)
            st.caption(
                f"Estimate for {selected_date:%A, %d %B %Y} at {selected_hour:02d}:00 "
                f"with {selected_promotion_label.lower()}."
            )


def render_testing_tab(model) -> None:
    st.header("Testing lab")
    st.info(
        "Model testing checks whether predictions are sensible and accurate. Application testing "
        "checks whether the app handles valid, invalid, and failing inputs safely."
    )
    st.write(
        "These checks use the already-trained model. They do not retrain or replace it, and "
        "deliberate invalid inputs never appear in the normal prediction form."
    )
    if st.button("Run five test cases", type="primary", icon=":material/science:"):
        st.session_state.testing_results = run_testing_lab(model)
    results = st.session_state.testing_results
    if results is not None:
        passed = int(results["Result"].eq("PASS").sum())
        with st.container(horizontal=True):
            st.metric("Tests passed", f"{passed}/{len(results)}", border=True)
            st.metric("Tests failed", int(results["Result"].eq("FAIL").sum()), border=True)
        if passed == len(results):
            st.success("All application checks passed.")
        else:
            st.error("One or more checks failed. Read the explanations below.")
        st.dataframe(
            results, hide_index=True, width="stretch",
            alt="Results for five Pizza Rush application test cases",
            column_config={
                "Test": st.column_config.TextColumn(width="medium"),
                "Result": st.column_config.TextColumn(width="small"),
                "Explanation": st.column_config.TextColumn(width="large"),
            },
        )


def build_monitoring_chart(history: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=history["timestamp"], y=history["actual_orders"], mode="lines",
        name="Actual orders", line={"width": 2},
    ))
    figure.add_trace(go.Scatter(
        x=history["timestamp"], y=history["predicted_orders"], mode="lines",
        name="Predicted orders", line={"width": 2},
    ))
    if history["timestamp"].max() >= SHIFT_START:
        figure.add_vline(
            x=SHIFT_START.to_pydatetime(), line_dash="dash", line_color="orange",
            annotation_text="Demand shift begins", annotation_position="top left",
        )
    if history["timestamp"].max() >= DATA_DRIFT_START:
        figure.add_vline(
            x=DATA_DRIFT_START.to_pydatetime(), line_dash="dot", line_color="purple",
            annotation_text="Promotion mix changes", annotation_position="top right",
        )
    figure.update_layout(
        title="Actual vs predicted incoming orders", xaxis_title="Simulated production time",
        yaxis_title="Orders per hour", hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 20, "r": 20, "t": 75, "b": 20},
    )
    return figure


def render_monitoring_explanations() -> None:
    with st.expander("Production concepts", icon=":material/school:"):
        st.markdown(
            """
**Monitoring predictions against outcomes:** A prediction is useful only when we later compare it with what actually happened. Absolute error measures the gap for one hour; MAE averages those gaps.

**Data drift:** The inputs change. From 1 October, this synthetic shop uses percent-off promotions much more often. Watch the promotion share below rise.

**Concept drift:** The relationship between the inputs and orders changes. In this simulation, a competitor closes, so the same hour and promotion can produce more orders than before.

**Why retrain:** If errors stay high after a genuine change, newer examples can help the model learn the new relationship. Retraining should follow investigation and testing, not happen automatically after one unusual hour.

**A real system:** Instead of uploading a CSV, production software would collect predictions and completed orders automatically from databases, APIs, delivery platforms, or the shop's point-of-sale system.
"""
        )


def process_monitoring_record(model, row: pd.Series) -> dict:
    """Predict one incoming row and return the values needed by the dashboard."""
    prediction = predict_orders(
        model, row["date"], int(row["hour"]), str(row["promotion"])
    )
    actual = float(row["order_count"])
    timestamp = pd.Timestamp(row["date"]) + pd.Timedelta(hours=int(row["hour"]))
    return {
        "timestamp": timestamp,
        "date": pd.Timestamp(row["date"]),
        "hour": int(row["hour"]),
        "promotion": str(row["promotion"]),
        "predicted_orders": prediction,
        "actual_orders": actual,
        "absolute_error": abs(actual - prediction),
    }


@st.fragment(run_every="1s")
def render_monitoring_tab(model) -> None:
    st.header("Monitoring simulation")
    st.warning("This is simulated production monitoring using synthetic 2026 records—not genuine live traffic.")
    st.write(
        "Upload `data/incoming_orders.csv`. Play or process one unseen hour at a time, then "
        "compares the existing model's prediction with the synthetic actual outcome."
    )
    if INCOMING_PATH.exists():
        st.download_button(
            "Download included incoming_orders.csv", data=INCOMING_PATH.read_bytes(),
            file_name="incoming_orders.csv", mime="text/csv", icon=":material/download:",
        )
    uploaded_file = st.file_uploader(
        "Upload the synthetic incoming orders CSV", type=["csv"],
        help="Required columns: " + ", ".join(EXPECTED_INCOMING_COLUMNS),
        key="incoming_csv_uploader",
    )
    if uploaded_file is None:
        render_monitoring_explanations()
        return

    uploaded_bytes = uploaded_file.getvalue()
    source_key = hashlib.sha256(uploaded_bytes).hexdigest()
    try:
        incoming = validate_incoming_data(pd.read_csv(uploaded_file))
    except (InputValidationError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"The uploaded CSV is not valid: {error}")
        render_monitoring_explanations()
        return
    if st.session_state.monitor_source_key != source_key:
        reset_monitoring_state(source_key)

    st.success(
        f"Validated {len(incoming):,} synthetic records from "
        f"{incoming['date'].min().date()} to {incoming['date'].max().date()}."
    )
    st.caption(
        "The documented demand shift begins on 1 September 2026, when a simulated nearby "
        "competitor closes. Promotion use changes on 1 October 2026 (data drift). "
        "All 2026 rows are separate from the model's 2025 training and test data."
    )
    with st.container(horizontal=True):
        play_clicked = st.button(
            "Pause" if st.session_state.monitor_playing else "Play",
            icon=":material/pause:" if st.session_state.monitor_playing else ":material/play_arrow:",
            disabled=st.session_state.monitor_index >= len(incoming),
        )
        next_clicked = st.button(
            "Next hour", type="primary", icon=":material/skip_next:",
            disabled=st.session_state.monitor_index >= len(incoming) or st.session_state.monitor_playing,
        )
        jump_clicked = st.button(
            "Jump to demand shift",
            icon=":material/fast_forward:",
            help="Loads the previous 48 hours for comparison, then pauses at 1 September 2026.",
        )
        data_jump_clicked = st.button(
            "Jump to data drift", icon=":material/fast_forward:",
            help="Loads the previous 48 hours, then pauses at 1 October 2026.",
        )
        reset_clicked = st.button("Reset simulation", icon=":material/replay:")
    if reset_clicked:
        reset_monitoring_state(source_key)
        st.rerun()
    if play_clicked:
        st.session_state.monitor_playing = not st.session_state.monitor_playing
    if jump_clicked or data_jump_clicked:
        st.session_state.monitor_playing = False
        target = DATA_DRIFT_START if data_jump_clicked else SHIFT_START
        shift_rows = incoming.index[incoming["date"] >= target]
        if shift_rows.empty:
            st.error(f"This CSV does not contain {target.date()}.")
        else:
            shift_index = int(shift_rows[0])
            baseline_start = max(0, shift_index - (ROLLING_WINDOW * 2))
            try:
                baseline_history = [
                    process_monitoring_record(model, incoming.iloc[position])
                    for position in range(baseline_start, shift_index)
                ]
            except (InputValidationError, PredictionError) as error:
                st.error(f"The comparison period could not be prepared: {error}")
            else:
                st.session_state.monitor_history = baseline_history
                st.session_state.monitor_index = shift_index
                st.success(
                    f"Jumped to {target.date()} and loaded the previous 48 hours for comparison. "
                    "Click **Play** or **Next hour** to continue."
                )
    if next_clicked or (st.session_state.monitor_playing and not play_clicked and not jump_clicked and not data_jump_clicked):
        row = incoming.iloc[st.session_state.monitor_index]
        try:
            monitoring_record = process_monitoring_record(model, row)
        except (InputValidationError, PredictionError) as error:
            st.session_state.monitor_playing = False
            st.error(f"This record could not be processed: {error}")
        except Exception:
            st.session_state.monitor_playing = False
            st.error("An unexpected prediction error occurred. The simulation was not advanced.")
        else:
            st.session_state.monitor_history.append(monitoring_record)
            st.session_state.monitor_index += 1
            if st.session_state.monitor_index >= len(incoming):
                st.session_state.monitor_playing = False

    history = pd.DataFrame(st.session_state.monitor_history)
    if history.empty:
        st.info("Click **Next hour** to process the first incoming record.")
        render_monitoring_explanations()
        return

    latest = history.iloc[-1]
    history["rolling_mae"] = history["absolute_error"].rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean()
    cumulative_mae = history["absolute_error"].mean()
    rolling_mae = history["rolling_mae"].iloc[-1]
    with st.container(horizontal=True):
        st.metric("Current date and hour", pd.Timestamp(latest["timestamp"]).strftime("%d %b %Y, %H:00"), border=True)
        st.metric("Predicted orders", f"{latest['predicted_orders']:.1f}", border=True)
        st.metric("Actual orders", f"{latest['actual_orders']:.0f}", border=True)
        st.metric("Absolute error", f"{latest['absolute_error']:.1f}", border=True)
    with st.container(horizontal=True):
        st.metric("Cumulative MAE", f"{cumulative_mae:.2f}", border=True)
        st.metric(f"Rolling MAE ({ROLLING_WINDOW} hours)", f"{rolling_mae:.2f}", border=True)
        st.metric("Progress", f"{st.session_state.monitor_index:,}/{len(incoming):,}", border=True)
    recent_promotions = history["promotion"].tail(48)
    st.metric(
        "Percent-off promotions (last 48 hours)",
        f"{recent_promotions.eq('percent_off').mean():.0%}", border=True,
    )
    if latest["date"] >= DATA_DRIFT_START:
        st.caption("Purple marker: the simulated promotion mix changes on 1 October (data drift).")

    warning_message = monitoring_warning(history)
    if warning_message:
        st.warning(warning_message + " Investigate possible drift before deciding to retrain.")
    st.plotly_chart(
        build_monitoring_chart(history), width="stretch", key="monitoring_actual_vs_predicted",
        alt="Actual and predicted hourly orders during the monitoring simulation",
    )
    if latest["date"] >= SHIFT_START:
        st.info(
            "The simulation is now in the shifted-demand period. The dashed orange marker shows "
            "when the simulated competitor closure began."
        )
    else:
        st.caption("The demand-shift marker will appear once the simulation reaches 1 September 2026.")

    with st.expander("Processed records"):
        display_history = history.copy()
        display_history["timestamp"] = display_history["timestamp"].dt.strftime("%Y-%m-%d %H:00")
        st.dataframe(
            display_history[["timestamp", "promotion", "actual_orders", "predicted_orders", "absolute_error", "rolling_mae"]].tail(100),
            hide_index=True, width="stretch",
            alt="Most recently processed synthetic incoming order records",
        )
    render_monitoring_explanations()


st.set_page_config(page_title="Pizza Rush Predictor", page_icon="🍕", layout="wide")
initialize_state()
st.title(shop_name)
st.caption("Predict demand, test application safeguards, and simulate production monitoring.")

if not MODEL_PATH.exists():
    st.error(
        "The saved model was not found. Run the notebook's existing ‘Save the complete pipeline’ "
        "cell, then refresh this page."
    )
    st.stop()
try:
    model = load_model()
except Exception:
    st.error("The saved model could not be loaded. Confirm the model artifact is present and valid.")
    st.stop()

predict_tab, testing_tab, monitoring_tab = st.tabs([
    ":material/auto_awesome: Predict",
    ":material/science: Testing Lab",
    ":material/monitoring: Monitoring",
])
with predict_tab:
    render_predict_tab(model)
with testing_tab:
    render_testing_tab(model)
with monitoring_tab:
    render_monitoring_tab(model)
