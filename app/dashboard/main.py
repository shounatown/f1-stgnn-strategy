import os
import torch
import streamlit as st

from f1_strategy_ai.data.fastf1_client import load_race_session
from f1_strategy_ai.models.st_gnn import F1StrategySTGNN
from app.dashboard.components.track_visualization import plot_animated_race
import fastf1

st.set_page_config(
    page_title="F1 strategy AI",
    layout="wide",
)



@st.cache_data(show_spinner="Loading season schedule...")
def get_season_gps(year: int) -> list[str]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["EventName"].tolist()
@st.cache_resource
def get_loaded_model(weights_path: str) -> F1StrategySTGNN:
    model = F1StrategySTGNN(node_features=11)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading telemetry...")
def get_race_data(year: int, gp_name: str):
    return load_race_session(year, gp_name, "R")


def main():
    st.title("F1 race strategy")


    st.sidebar.header("Race Configuration")
    year = st.sidebar.selectbox("Season", [2026, 2025, 2024, 2023, 2022], index=1)
    gp_list = get_season_gps(year)
    gp_name = st.sidebar.selectbox("Grand Prix", gp_list)

    lap_range = st.sidebar.slider(
        "Lap Range",
        min_value=5,
        max_value=55,
        value=(10, 20),
    )
    start_lap, end_lap = lap_range

    weights_path = st.sidebar.text_input(
        "Model weights (.pt)",
        value="models/st_gnn_2025_multi.pt",
    )

    run_btn = st.sidebar.button("Load and simulate", type="primary")

    # --- Key Performance Metrics ---
    col1, col2 = st.columns(2)
    col1.metric("Validation MAE", "1.215 s", "-0.720 s vs Baseline")
    col2.metric("Gain vs Baseline", "+37.19%")

    st.divider()

    if run_btn:
        if not os.path.exists(weights_path):
            st.error("Model weights not found at: " + weights_path)
            return

        session = get_race_data(year, gp_name)

        st.subheader(f"{gp_name} {year} | Laps {start_lap}-{end_lap}")

        with st.spinner("Generating simulation..."):
            fig = plot_animated_race(
                session,
                weights_path=weights_path,
                start_lap=start_lap,
                end_lap=end_lap,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a race in the sidebar and click **Load and Simulate**.")


if __name__ == "__main__":
    main()