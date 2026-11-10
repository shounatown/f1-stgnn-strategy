from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def train_model( model: nn.Module, dataset: Dataset, val_dataset: Dataset | None = None, epochs: int = 30,
    lr: float = 0.001, save_path: str = "models/st_gnn.pt" ) -> dict[str, list[float]]:


    if val_dataset is None:
        split_idx = int(len(dataset) * 0.8)
        train_samples = [dataset[i] for i in range(split_idx)]
        val_samples = [dataset[i] for i in range(split_idx, len(dataset))]
    else:
        train_samples = [dataset[i] for i in range(len(dataset))]
        val_samples = [val_dataset[i] for i in range(len(val_dataset))]

    #HuberLoss
    criterion = nn.HuberLoss(delta=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_mae = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        #Training
        model.train()
        total_train_mae = 0.0

        for seq_graphs, target_y in train_samples:
            optimizer.zero_grad()
            pred_y = model(seq_graphs)
            loss = criterion(pred_y, target_y)
            loss.backward()
            optimizer.step()

            mae = torch.mean(torch.abs(pred_y - target_y)).item()
            total_train_mae += mae

        avg_train_mae = total_train_mae / max(len(train_samples), 1)

        #Validation
        model.eval()
        total_val_mae = 0.0

        with torch.no_grad():
            for seq_graphs, target_y in val_samples:
                pred_y = model(seq_graphs)
                mae = torch.mean(torch.abs(pred_y - target_y)).item()
                total_val_mae += mae

        avg_val_mae = total_val_mae / max(len(val_samples), 1)
        scheduler.step(avg_val_mae)

        history["train_loss"].append(avg_train_mae)
        history["val_loss"].append(avg_val_mae)

        saved_msg = ""
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), save_path)
            saved_msg = " [BEST SAVED]"

        if epoch % 5 == 0 or epoch == 1 or saved_msg != "":
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                "Epoch " + str(epoch).zfill(2) + "/" + str(epochs)
                + " | Train MAE: " + "{:.3f}".format(avg_train_mae) + "s"
                + " | Val MAE: " + "{:.3f}".format(avg_val_mae) + "s"
                + " | LR: " + "{:.1e}".format(current_lr)
                + saved_msg
            )

    print("Training finished. Best Val MAE: " + "{:.3f}".format(best_val_mae) + "s")
    print("Best model saved in: " + save_path)

    return history
