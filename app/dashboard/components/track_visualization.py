import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.data import Data

import torch
from f1_strategy_ai.graph.graph_builder import build_lap_graph
from f1_strategy_ai.models.st_gnn import F1StrategySTGNN

def plot_circuit_graph(session, graph_data: Data, lap_number: int = 20) -> go.Figure:
    #plot_circuit_graph: plots the circuit spatial traffic graph

    ref_lap = session.laps.pick_fastest()
    ref_telemetry = ref_lap.get_telemetry().dropna(subset=["X", "Y"])

    track_x = ref_telemetry["X"] / 10.0
    track_y = ref_telemetry["Y"] / 10.0

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.74, 0.26],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.02,
    )

    #Circuit
    fig.add_trace(
        go.Scatter(
            x=track_x,
            y=track_y,
            mode="lines",
            line=dict(color="#333333", width=4),
            name="Circuit",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    #Traffic edges
    pos = graph_data.pos.numpy()
    edge_index = graph_data.edge_index.numpy()

    edge_x, edge_y = [], []
    for src, dst in zip(edge_index[0], edge_index[1]):
        edge_x.extend([pos[src, 0], pos[dst, 0], None])
        edge_y.extend([pos[src, 1], pos[dst, 1], None])

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(255, 60, 60, 0.7)", width=2),
                name="Traffic (<25m)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    #Nodes
    driver_names = getattr(graph_data, "driver_mapping", [f"Car {i}" for i in range(graph_data.num_nodes)])
    speeds = graph_data.x[:, 0].numpy()

    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers",
            marker=dict(
                size=14,
                color=speeds,
                colorscale="Turbo",
                colorbar=dict(
                    title=dict(text="Velocity (km/h)", font=dict(color="white", size=11)),
                    tickfont=dict(color="white", size=10),
                    orientation="h",
                    x=0.37,
                    y=-0.08,
                    len=0.45,
                    thickness=12,
                ),
                showscale=True,
                line=dict(color="white", width=1.5),
            ),
            text=driver_names,
            name="Driver",
            hovertemplate="<b>Driver:</b> %{text}<br><b>Velocity:</b> %{marker.color:.1f} km/h<extra></extra>",
        ),
        row=1,
        col=1,
    )

    #Positions
    lap_standings = session.laps[session.laps["LapNumber"] == lap_number].sort_values("Position")

    team_map = {}
    color_map = {}
    if hasattr(session, "results") and not session.results.empty:
        for _, r in session.results.iterrows():
            abbr = r["Abbreviation"]
            team_map[abbr] = r.get("TeamName", "-")
            hex_color = str(r.get("TeamColor", "555555")).strip()
            color_map[abbr] = f"#{hex_color}" if not hex_color.startswith("#") else hex_color

    tire_colors = {
        "SOFT": "#FF3333",
        "MEDIUM": "#FFD700",
        "HARD": "#FFFFFF",
        "INTERMEDIATE": "#39B54A",
        "WET": "#00AEEF",
    }

    pos_col, drv_col, team_col, tire_col = [], [], [], []
    team_text_colors, tire_text_colors = [], []

    for _, row in lap_standings.iterrows():
        drv = row["Driver"]
        comp = str(row["Compound"]) if pd.notna(row["Compound"]) else "-"

        pos_col.append(int(row["Position"]) if pd.notna(row["Position"]) else "-")
        drv_col.append(f"<b>{drv}</b>")
        team_col.append(team_map.get(drv, str(row.get("Team", "-"))))
        tire_col.append(f"<b>{comp[0]}</b>" if comp != "-" else "-")

        team_text_colors.append(color_map.get(drv, "#AAAAAA"))
        tire_text_colors.append(tire_colors.get(comp, "#AAAAAA"))

    fig.add_trace(
        go.Table(
            columnwidth=[0.14, 0.24, 0.46, 0.16],
            header=dict(
                values=["<b>POS</b>", "<b>DRIVER</b>", "<b>TEAM</b>", "<b>TYR</b>"],
                fill_color="#15151E",
                align=["center", "left", "left", "center"],
                font=dict(color="#E10600", size=11, family="Arial Black"),
                line_color="#2A2A35",
                height=30,
            ),
            cells=dict(
                values=[pos_col, drv_col, team_col, tire_col],
                fill_color=[["#1B1B24", "#16161D"] * (len(pos_col) // 2 + 1)],
                align=["center", "left", "left", "center"],
                font=dict(
                    color=["white", "white", team_text_colors, tire_text_colors],
                    size=11,
                ),
                line_color="#24242E",
                height=26,
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)

    gp_name = session.event.get("EventName", "Grand Prix")
    year = session.event["EventDate"].year if "EventDate" in session.event else ""

    fig.update_layout(
        title=dict(
            text=f"{gp_name} {year} | LAP {lap_number}".strip(),
            font=dict(color="white", size=16),
        ),
        paper_bgcolor="#0e0e11",
        plot_bgcolor="#0e0e11",
        margin=dict(l=15, r=15, t=55, b=80),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(20,20,25,0.7)",
            x=0.01,
            y=0.99,
        ),
    )

    return fig



def plot_animated_race(session, weights_path: str = "models/st_gnn.pt", start_lap: int = 10,
                       end_lap: int = 25, window_size: int = 5,
                    ) -> go.Figure:

    #plot_animated_race:creates an interactive animated dashboard

    race_name = session.event["EventName"]
    race_year = session.event["EventDate"].year

    #Load model
    model = F1StrategySTGNN()
    model.load_state_dict(torch.load(weights_path))
    model.eval()


    graphs = {}
    for lap in range(max(2, start_lap - window_size + 1), end_lap + 1):
        g = build_lap_graph(session, lap_number=lap)
        if g is not None:
            graphs[lap] = g

    #Build frames
    frames = []
    valid_laps = []

    for lap in range(start_lap, end_lap + 1):
        seq_laps = [lap - i for i in range(window_size - 1, -1, -1)]
        if not all(l in graphs for l in seq_laps):
            continue

        seq_graphs = [graphs[l] for l in seq_laps]
        if len({g.num_nodes for g in seq_graphs}) != 1:
            continue

        #Normalize
        norm_seq = []
        for g in seq_graphs:
            g_norm = g.clone()
            g_norm.x[:, 0] = g_norm.x[:, 0] / 350.0
            g_norm.x[:, 1] = g_norm.x[:, 1] / 100.0
            g_norm.x[:, 3] = g_norm.x[:, 3] / 50.0
            g_norm.x[:, 4] = (g_norm.x[:, 4] - 90.0) / 10.0
            g_norm.x[:, 5] = g_norm.x[:, 5] / 3.0
            g_norm.x[:, 7] = g_norm.x[:, 7] / 60.0
            g_norm.x[:, 8] = g_norm.x[:, 8] / 50.0
            norm_seq.append(g_norm)


        with torch.no_grad():
            preds = (model(norm_seq).squeeze(1) + 90.0).tolist()

        current_g = graphs[lap]
        pos = current_g.pos.numpy()
        drivers = current_g.driver_mapping
        speeds = current_g.x[:, 0].numpy()




        #Edges
        edge_x, edge_y = [], []
        edges = current_g.edge_index.numpy()
        for src, dst in zip(edges[0], edges[1]):
            edge_x.extend([pos[src, 0], pos[dst, 0], None])
            edge_y.extend([pos[src, 1], pos[dst, 1], None])

        rows = sorted(
            zip(drivers, speeds, preds),
            key=lambda r: r[2],
        )
        tbl_drivers = [r[0] for r in rows]
        tbl_speeds = [f"{r[1]:.0f} km/h" for r in rows]
        tbl_preds = [f"{r[2]:.2f} s" for r in rows]

        frames.append(
            go.Frame(
                name=str(lap),
                traces=[1, 2, 3],
                data=[
                    go.Scatter(x=edge_x, y=edge_y),
                    go.Scatter(
                        x=pos[:, 0],
                        y=pos[:, 1],
                        text=drivers,
                        customdata=list(zip(preds, speeds)),
                        marker=dict(color=speeds),
                    ),
                    go.Table(
                        cells=dict(values=[tbl_drivers, tbl_speeds, tbl_preds])
                    ),
                ],
                layout=go.Layout(
                    title_text=f"{race_name} {race_year} | Lap {lap}"
                ),
            )
        )
        valid_laps.append(lap)

    first_frame = frames[0]
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("Track and traffic graph (< 25m)", "Next-lap predictions"),
    )

    # Circuit
    ref_tel = session.laps.pick_fastest().get_telemetry()
    fig.add_trace(
        go.Scatter(
            x=ref_tel["X"] / 10.0,
            y=ref_tel["Y"] / 10.0,
            mode="lines",
            line=dict(color="rgba(150, 150, 150, 0.35)", width=6),
            hoverinfo="none",
            name="Track",
        ),
        row=1,
        col=1,
    )

    #Traffic edges
    fig.add_trace(
        go.Scatter(
            x=first_frame.data[0].x,
            y=first_frame.data[0].y,
            mode="lines",
            line=dict(color="rgba(255, 60, 60, 0.6)", width=1.5),
            hoverinfo="none",
            name="Traffic (<25m)",
        ),
        row=1,
        col=1,
    )

    #Drivers
    fig.add_trace(
        go.Scatter(
            x=first_frame.data[1].x,
            y=first_frame.data[1].y,
            mode="markers",
            text=first_frame.data[1].text,
            marker=dict(
                size=11,
                color=first_frame.data[1].marker.color,
                colorscale="Turbo",
                cmin=50,
                cmax=340,
                showscale=True,
                colorbar=dict(
                    title="km/h",
                    x=0.57,
                    thickness=12,
                    len=0.7,
                ),
                line=dict(width=1.5, color="white"),
            ),
            customdata=first_frame.data[1].customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Speed: %{customdata[1]:.0f} km/h<br>"
                "Pred Next Lap: %{customdata[0]:.2f}s"
                "<extra></extra>"
            ),
            name="Drivers",
        ),
        row=1,
        col=1,
    )

    #Table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Driver</b>", "<b>Speed</b>", "<b>Pred next lap</b>"],
                fill_color="#1F1F2B",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=first_frame.data[2].cells.values,
                fill_color="#15151E",
                font=dict(color="white", size=11),
                align="center",
            ),
        ),
        row=1,
        col=2,
    )

    #Controls
    fig.frames = frames
    fig.update_layout(
        template="plotly_dark",
        title_text=f"{race_name} {race_year} | Lap {lap}",
        height=680,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=-0.05,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.18,
                y=-0.05,
                len=0.75,
                steps=[
                    dict(
                        method="animate",
                        args=[[str(lap)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=f"L{lap}",
                    )
                    for lap in valid_laps
                ],
            )
        ],
    )

    return fig