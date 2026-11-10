import pandas as pd
import torch
from torch.utils.data import Dataset

from f1_strategy_ai.graph.graph_builder import build_lap_graph


class F1TemporalDataset(Dataset):
    #PyTorch Dataset that builds sliding window graph sequences for race laps

    def __init__(self, session, start_lap: int = 2, end_lap: int = 70, window_size: int = 5):
        self.samples = []
        self.window_size = window_size

        graphs = {}
        targets = {}

        #Built graphs and extract times lap by lap
        for lap in range(start_lap, end_lap + 1):

            g = build_lap_graph(session, lap_number=lap)
            if g is None:
                continue

            #Normalize
            g.x = g.x.clone()

            g.x[:, 0] = g.x[:, 0] / 350.0  # Speed
            g.x[:, 1] = g.x[:, 1] / 100.0  # Throttle
            g.x[:, 3] = g.x[:, 3] / 50.0  # TyreLife
            g.x[:, 4] = (g.x[:, 4] - 90.0) / 10.0  # PrevLapTime
            g.x[:, 5] = g.x[:, 5] / 3.0  # Compound
            g.x[:, 7] = g.x[:, 7] / 60.0  # LapNumber
            g.x[:, 8] = g.x[:, 8] / 50.0  # TrackTemp

            lap_df = session.laps[session.laps["LapNumber"] == lap]
            lap_times = []
            valid_lap = True

            for drv in g.driver_mapping:
                drv_row = lap_df[lap_df["Driver"] == drv]
                if drv_row.empty or pd.isna(drv_row["LapTime"].iloc[0]):
                    valid_lap = False
                    break

                lap_sec = drv_row["LapTime"].iloc[0].total_seconds() - 90.0
                lap_times.append(lap_sec)

            if valid_lap:
                graphs[lap] = g
                targets[lap] = torch.tensor(lap_times, dtype=torch.float).unsqueeze(1)

        #Built sliding windows
        sorted_laps = sorted(graphs.keys())
        for i in range(len(sorted_laps) - window_size):
            seq_laps = sorted_laps[i : i + window_size]
            target_lap = sorted_laps[i + window_size]

            #The same amount of node for each lap
            if target_lap - seq_laps[0] == window_size:
                seq_graphs = [graphs[l] for l in seq_laps]
                node_counts = {g.num_nodes for g in seq_graphs}
                if len(node_counts) == 1 and targets[target_lap].shape[0] == seq_graphs[0].num_nodes:
                    self.samples.append((seq_graphs, targets[target_lap]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]