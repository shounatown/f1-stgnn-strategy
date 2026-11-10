import argparse
from f1_strategy_ai.data.fastf1_client import load_race_session
from app.dashboard.components.track_visualization import plot_animated_race


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run animated F1 ST-GNN race strategy visualization."
    )
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    parser.add_argument("--gp", type=str, default="Abu Dhabi", help="Grand Prix name")
    parser.add_argument("--session", type=str, default="R", help="Session type (R, Q, etc.)")
    parser.add_argument("--start-lap", type=int, default=10, help="First lap to animate")
    parser.add_argument("--end-lap", type=int, default=25, help="Last lap to animate")
    parser.add_argument("--window-size", type=int, default=5, help="Temporal window size")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/st_gnn_abudhabi_2025.pt",
        help="Path to trained .pt model weights",
    )
    args = parser.parse_args()

    session = load_race_session(args.year, args.gp, args.session)
    fig = plot_animated_race(
        session=session,
        weights_path=args.weights,
        start_lap=args.start_lap,
        end_lap=args.end_lap,
        window_size=args.window_size,
    )
    fig.show()


if __name__ == "__main__":
    main()