import os
import sys
import torch
import plotly.graph_objects as go
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

parser = argparse.ArgumentParser(description="Experiment 06 visualization (final memory states)")
parser.add_argument("--stage", help="Input stage code: a|b|c", required=True)
parser.add_argument("--threshold", help="Visualisation threshold", required=False, default=1.0e-3, type=float)
args = parser.parse_args()

# Load memory state
stage = str(args.stage).strip()
data = torch.load(f"{script_dir}/data/ex06/stage_{stage}_memory_state.datarec.pt")
memory = data["memory"].cpu()
target_A = data["target_A"].cpu()
target_B = data["target_B"].cpu() if data["target_B"] is not None else 0.0

# Scalar density
density_a = (memory * target_A.view(1, 1, 1, -1)).abs().sum(dim=-1)
density_b = (memory * target_B.view(1, 1, 1, -1)).abs().sum(dim=-1) if data["target_B"] is not None else 0.0

# Normalize density
density_a = density_a / (density_a.max() + 1e-8) * -1.0
density_b = density_b / (density_b.max() + 1e-8) if data["target_B"] is not None else 0.0
heat = density_a + density_b
heat = heat / heat.abs().max()

# Prepare heatmap
threshold = args.threshold
mask = heat.abs() > threshold
x, y, z = mask.nonzero(as_tuple=True)
values = heat[mask].numpy()
norm_values = (values + 1.0) / 2.0

# Scene size
size_x, size_y, size_z = heat.shape

# Visualize
fig = go.Figure(data=go.Scatter3d(
    x=x.numpy(),
    y=y.numpy(),
    z=z.numpy(),
    mode='markers',
    marker=dict(
        size=3,
        color=norm_values,
        colorscale="Jet",
        cmin=0.0,
        cmax=1.0,
        opacity=0.8,
        colorbar=dict(
            title="Semantic Affinity",
            tickvals=[0.0, 0.5, 1.0],
            ticktext=["Zone A", "Blend", "Zone B"]
        )
    )
))

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(title='X', range=[0, size_x]),
        yaxis=dict(title='Y', range=[0, size_y]),
        zaxis=dict(title='Z', range=[0, size_z]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
    ),
    title="3D Semantic Divergence in HPM Field",
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
