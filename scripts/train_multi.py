import os
from torch.utils.data import ConcatDataset

from f1_strategy_ai.data.fastf1_client import load_race_session
from f1_strategy_ai.data.dataset import F1TemporalDataset
from f1_strategy_ai.models.st_gnn import F1StrategySTGNN
from f1_strategy_ai.training.trainer import train_model


def main():
    races = [
        (2025, "Silverstone"),
        (2025, "Monaco"),
        (2025, "Monza"),
        (2025, "Abu Dhabi"),
    ]

    train_datasets = []
    val_datasets = []

    for year, gp in races:
        print("Loading " + str(year) + " " + gp + "...")
        session = load_race_session(year, gp, "R")

        ds_train = F1TemporalDataset(session, start_lap=5, end_lap=28, window_size=5)
        ds_val = F1TemporalDataset(session, start_lap=29, end_lap=35, window_size=5)

        train_datasets.append(ds_train)
        val_datasets.append(ds_val)

    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset = ConcatDataset(val_datasets)

    print("Train sequences: " + str(len(full_train_dataset)) + " | Val sequences: " + str(len(full_val_dataset)))

    model = F1StrategySTGNN(node_features=11)
    os.makedirs("models", exist_ok=True)

    train_model(
        model,
        full_train_dataset,
        val_dataset=full_val_dataset,
        epochs=30,
        save_path="models/st_gnn_2025_multi.pt",
    )


if __name__ == "__main__":
    main()