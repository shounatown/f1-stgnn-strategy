from app.dashboard.components.track_visualization import plot_circuit_graph
from f1_strategy_ai.data.fastf1_client import load_race_session
from f1_strategy_ai.graph.graph_builder import build_lap_graph


def main():
    session = load_race_session(2025, "Abu Dhabi", "R")
    lap_number = 20

    graph = build_lap_graph(session, lap_number=lap_number, distance_threshold=25.0)

    if graph is not None:
        fig = plot_circuit_graph(session, graph, lap_number=lap_number)
        fig.show()
        fig.write_html("circuito_f1.html", auto_open=True)
    else:
        print("Failed to process telemetry.")


if __name__ == "__main__":
    main()