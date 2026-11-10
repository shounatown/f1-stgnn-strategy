from f1_strategy_ai.data.fastf1_client import load_race_session
from f1_strategy_ai.graph.graph_builder import build_lap_graph


def main():
    session = load_race_session(2025, "Abu Dhabi", "R")

    lap_number = 20
    graph = build_lap_graph(session, lap_number=lap_number, distance_threshold=25.0)

    if graph is not None:
        print(f" Drivers active on track (nodes): {graph.num_nodes}")
        print(f" Active traffic/dirty-air connections (edges): {graph.num_edges}")
        print(f" Feature matrix shape (x): {graph.x.shape}")
        print(f" Connections tensor (edge_index):\n{graph.edge_index}")
    else:
        print("Failed to process telemetry.")


if __name__ == "__main__":
    main()