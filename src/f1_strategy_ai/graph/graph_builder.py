import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd


def build_lap_graph(
    session, lap_number: int, timestamp_s: float = 40.0, distance_threshold: float = 25.0
) -> Data | None:

    # build_lap_graph: takes race telemetry at a global SessionTime of a lap and builds a PyG graph.
    laps = session.laps
    drivers = session.drivers

    current_laps = laps[laps["LapNumber"] == lap_number]
    if current_laps.empty:
        return None

    # Global reference time: leader's lap start + timestamp_s
    valid_starts = current_laps["LapStartTime"].dropna()
    if valid_starts.empty:
        return None
    ref_session_time = valid_starts.min() + pd.Timedelta(seconds=timestamp_s)

    node_features_list = []
    positions_xy = []
    driver_mapping = []

    compound_map = {
        "SOFT": 1.0,
        "MEDIUM": 2.0,
        "HARD": 3.0,
        "INTERMEDIATE": 4.0,
        "WET": 5.0,
    }

    driver_info = {}
    if hasattr(session, "results") and not session.results.empty:
        for _, row in session.results.iterrows():
            driver_info[str(row["DriverNumber"])] = row["Abbreviation"]

    prev_laps = laps[laps["LapNumber"] == (lap_number - 1)]
    prev_times_map = {}
    for _, row in prev_laps.iterrows():
        if pd.notna(row["LapTime"]):
            prev_times_map[row["Driver"]] = row["LapTime"].total_seconds()

    for driver in drivers:
        drv_str = str(driver)
        lap_data = current_laps[current_laps["DriverNumber"].astype(str) == drv_str]
        if lap_data.empty:
            continue

        pos_df = session.pos_data.get(drv_str)
        car_df = session.car_data.get(drv_str)
        if pos_df is None or car_df is None or pos_df.empty or car_df.empty:
            continue

        # Find closest sample to the global ref_session_time
        pos_clean = pos_df.dropna(subset=["X", "Y", "SessionTime"])
        car_clean = car_df.dropna(subset=["Speed", "Throttle", "Brake", "SessionTime"])
        if pos_clean.empty or car_clean.empty:
            continue

        pos_idx = (pos_clean["SessionTime"] - ref_session_time).abs().idxmin()
        car_idx = (car_clean["SessionTime"] - ref_session_time).abs().idxmin()

        pos_sample = pos_clean.loc[pos_idx]
        car_sample = car_clean.loc[car_idx]

        # Node features: speed, throttle, brake, tyre_life, prev_lap_time, compound, track_caution, lap_number, track_temp, rainfall, is_pit_lap
        speed = float(car_sample["Speed"])
        throttle = float(car_sample["Throttle"])
        brake = float(car_sample["Brake"])

        tyre_val = lap_data["TyreLife"].iloc[0]
        tyre_life = float(tyre_val) if pd.notna(tyre_val) else 1.0

        driver_code = driver_info.get(drv_str, drv_str)
        prev_lap_time = prev_times_map.get(driver_code, 90.0)

        comp_str = str(lap_data["Compound"].iloc[0]).upper() if pd.notna(lap_data["Compound"].iloc[0]) else "MEDIUM"
        compound_val = compound_map.get(comp_str, 2.0)

        status_str = (
            str(lap_data["TrackStatus"].iloc[0])
            if "TrackStatus" in lap_data.columns and pd.notna(lap_data["TrackStatus"].iloc[0])
            else "1"
        )
        track_caution = 0.0 if status_str == "1" else 1.0

        current_lap = float(lap_number)

        track_temp = 30.0
        rainfall = 0.0
        try:
            weather = lap_data.get_weather_data()
            if not weather.empty:
                if "TrackTemp" in weather.columns and pd.notna(weather["TrackTemp"].mean()):
                    track_temp = float(weather["TrackTemp"].mean())
                if "Rainfall" in weather.columns:
                    rainfall = 1.0 if weather["Rainfall"].any() else 0.0
        except Exception:
            pass

        pit_in = pd.notna(lap_data["PitInTime"].iloc[0]) if "PitInTime" in lap_data.columns else False
        pit_out = pd.notna(lap_data["PitOutTime"].iloc[0]) if "PitOutTime" in lap_data.columns else False
        is_pit_lap = 1.0 if (pit_in or pit_out) else 0.0

        node_features_list.append([
            speed, throttle, brake, tyre_life, prev_lap_time,
            compound_val, track_caution, current_lap,
            track_temp, rainfall, is_pit_lap,
        ])

        pos_x = float(pos_sample["X"]) / 10.0
        pos_y = float(pos_sample["Y"]) / 10.0

        positions_xy.append([pos_x, pos_y])
        driver_mapping.append(driver_code)

    if not node_features_list:
        return None

    x = torch.tensor(node_features_list, dtype=torch.float)
    pos = np.array(positions_xy)

    # Edges matrix
    edge_index_list = []
    num_nodes = len(positions_xy)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_m = np.linalg.norm(pos[i] - pos[j])
                if dist_m <= distance_threshold:
                    edge_index_list.append([j, i])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index)
    graph_data.driver_mapping = driver_mapping
    graph_data.pos = torch.tensor(pos, dtype=torch.float)

    return graph_data