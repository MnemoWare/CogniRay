import os
import sys
import torch
import plotly.graph_objects as go

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Load loss data
data = torch.load(f"{script_dir}/data/ex06/loss_dynamics.datarec.pt")

loss_A = data["loss_A"].cpu() if data["loss_A"] is not None else None
loss_B = data["loss_B"].cpu() if data["loss_B"] is not None else None

# Plot losses
fig = go.Figure()

if loss_A is not None:
    fig.add_trace(go.Scatter(
        y=loss_A,
        mode="lines",
        name="Loss A",
    ))

if loss_B is not None:
    fig.add_trace(go.Scatter(
        y=loss_B,
        mode="lines",
        name="Loss B",
    ))

fig.update_layout(
    title=f"Loss Dynamics (logarithmic scale)",
    xaxis_title="Step",
    yaxis_title="Loss",
    template="plotly_white",
    yaxis_type="log",
)

fig.show()
