import os
import sys
import torch
import plotly.graph_objects as go
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

parser = argparse.ArgumentParser(description="Experiment 05 visualization (memory dynamics)")
parser.add_argument("--threshold", help="Visualisation threshold", required=False, default=7.5e-2, type=float)
args = parser.parse_args()

# Load memory state
data = torch.load(f"{script_dir}/data/ex05/05_stage_a_dynamics_memory_state.datarec.pt")
memory = data["memory"].cpu()
target_A = data["target_A"].cpu()

# Scalar density
density_a = (memory * target_A.view(1, 1, 1, 1, -1)).abs().sum(dim=-1)

# Normalize density
heat = density_a / (density_a.max() + 1e-8)
heat = heat / heat.abs().max()

# Prepare heatmap
threshold = args.threshold
mask = heat.abs() > threshold
timeframe, x, y, z = mask.nonzero(as_tuple=True)
values = heat[mask].numpy()
norm_values = (values + 1.0) / 2.0

# Build animation
frames = []

# Coords
x_coords = x.numpy()
y_coords = y.numpy()
z_coords = z.numpy()

# Timescale
timesteps = sorted(set(timeframe.tolist()))

for t in timesteps:
    mask_t = timeframe == t
    x_t = x_coords[mask_t]
    y_t = y_coords[mask_t]
    z_t = z_coords[mask_t]
    val_t = norm_values[mask_t]

    scatter = go.Scatter3d(
        x=x_t,
        y=y_t,
        z=z_t,
        mode="markers",
        marker=dict(
            size=3,
            color=val_t,
            colorscale="Jet",
            cmin=0.0,
            cmax=1.0,
            opacity=0.8,
            colorbar=dict(
                title="Semantic Affinity",
                tickvals=[0.0, 1.0],
                ticktext=["Initial noise", "Zone A"]
            ),
        ),
        name=f"Step {t}"
    )

    frames.append(go.Frame(data=[scatter], name=f"step_{t:03d}"))

# First frame
initial_frame = frames[0].data[0]

# Scene size
_, size_x, size_y, size_z = heat.shape

# Build final animation
fig = go.Figure(
    data=[initial_frame],
    layout=go.Layout(
        title=f"HPM Memory Dynamics — Stage A",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(title="X", range=[0, size_x]),
            yaxis=dict(title="Y", range=[0, size_y]),
            zaxis=dict(title="Z", range=[0, size_z]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
            )]
        )],
        sliders=[dict(
            steps=[
                dict(method="animate", args=[[f"step_{t:03d}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"{t}")
                for t in timesteps
            ],
            transition=dict(duration=0),
            x=0, y=0, len=1.0
        )]
    ),
    frames=frames
)

fig.show()
