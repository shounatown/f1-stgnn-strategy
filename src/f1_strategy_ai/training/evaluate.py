import torch
import torch.nn as nn
from torch.utils.data import Dataset


def evaluate_model(model: nn.Module, dataset: Dataset, weights_path: str = "models/st_gnn.pt",
            ) -> dict[str, float]:

    #evaluate_model: evaluate the trained ST-GNN against a mean baseline on the validation split

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    #80/20
    split_idx = int(len(dataset) * 0.8)
    train_samples = [dataset[i] for i in range(split_idx)]
    val_samples = [dataset[i] for i in range(split_idx, len(dataset))]

    train_targets = torch.cat([y for _, y in train_samples], dim=0)
    baseline_pred = train_targets.mean()

    mae_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()

    gnn_preds, reals, baseline_preds = [], [], []

    with torch.no_grad():
        for seq_graphs, target_y in val_samples:
            pred_y = model(seq_graphs)
            gnn_preds.append(pred_y)
            reals.append(target_y)
            baseline_preds.append(torch.full_like(target_y, baseline_pred))

    all_reals = torch.cat(reals, dim=0)
    all_gnn = torch.cat(gnn_preds, dim=0)
    all_base = torch.cat(baseline_preds, dim=0)

    gnn_mae = mae_fn(all_gnn, all_reals).item()
    gnn_rmse = torch.sqrt(mse_fn(all_gnn, all_reals)).item()

    base_mae = mae_fn(all_base, all_reals).item()
    base_rmse = torch.sqrt(mse_fn(all_base, all_reals)).item()

    improvement = ((base_mae - gnn_mae) / base_mae) * 100

    print(f"Validation Evaluation ({len(val_samples)} sequences):")
    print(f"Baseline - MAE: {base_mae:.3f}s | RMSE: {base_rmse:.3f}s")
    print(f"ST-GNN - MAE: {gnn_mae:.3f}s | RMSE: {gnn_rmse:.3f}s")
    print(f"Improvement over baseline: {improvement:+.2f}%")

    return {
        "gnn_mae": gnn_mae,
        "gnn_rmse": gnn_rmse,
        "base_mae": base_mae,
        "base_rmse": base_rmse,
        "improvement_pct": improvement,
    }