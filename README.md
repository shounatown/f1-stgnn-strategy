# F1 ST-GNN Strategy

Spatio-temporal graph neural network (GAT + LSTM) for Formula 1 lap time prediction and race strategy simulation using FastF1 telemetry.

## Overview

This project models Formula 1 races as dynamic spatio-temporal graphs. At each lap, cars are represented as nodes with 11 telemetry, tyre, and track-condition features, while proximity edges (`< 25m`) capture real-time traffic and slipstream interactions. A Graph Attention Network (GAT) extracts spatial race context, and an LSTM models temporal degradation and pace evolution across laps.

### Key Results
* **Validation MAE:** `1.215 s` (`-0.720 s` vs. baseline)
* **Improvement vs. Baseline:** `+37.19%`

## Project Structure

```text
f1-stgnn-strategy/
├── app/
│   └── dashboard/
│       ├── main.py                        # Streamlit race strategy dashboard
│       └── components/
│           └── track_visualization.py     # Interactive Plotly circuit & traffic animation
├── models/
│   └── st_gnn_2025_multi.pt               # Pre-trained ST-GNN weights
├── scripts/
│   ├── animate_race.py                    # Standalone race animation script
│   ├── build_graphs.py                    # Graph dataset generation
│   ├── train_multi.py                     # Multi-circuit training pipeline
│   └── visualize_graph.py                 # Static lap graph visualization
└── src/
    └── f1_strategy_ai/
        ├── data/                          # FastF1 client & PyG temporal dataset
        ├── graph/                         # Global SessionTime graph builder
        ├── models/                        # Spatial encoder (GAT) & ST-GNN architecture
        └── training/                      # Training loop & evaluation
```

## Quickstart

1. **Clone the repository**
   ```bash
   git clone [https://github.com/shounatown/f1-stgnn-strategy.git](https://github.com/shounatown/f1-stgnn-strategy.git)
   cd f1-stgnn-strategy
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run the Streamlit Dashboard**
   ```bash
   streamlit run app/dashboard/main.py
   ```
