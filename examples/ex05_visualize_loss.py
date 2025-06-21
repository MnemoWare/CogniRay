import os
import sys
import torch
import plotly.graph_objects as go
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

parser = argparse.ArgumentParser(description="Experiment 06 visualization (final memory states)")
parser.add_argument("-m", "--method", help="Input method name: delta|suppresive|associative|reflexive", required=True)
parser.add_argument("-s", "--stage", help="Input stage code: a|b", required=True)
args = parser.parse_args()

# Load loss data
method = str(args.method).strip()
stage = str(args.stage).strip()
data = torch.load(f"{script_dir}/data/ex05/{method}_stage_{stage}_loss_dynamics.datarec.pt")

loss_A = data["loss_A"].cpu() if data["loss_A"] is not None else None
loss_other = data["loss_other"].cpu() if data["loss_other"] is not None else None

# Plot losses
fig = go.Figure()

if loss_A is not None:
    fig.add_trace(go.Scatter(
        y=loss_A,
        mode="lines",
        name="Loss A"
    ))

if loss_other is not None:
    fig.add_trace(go.Scatter(
        y=loss_other,
        mode="lines",
        name="Loss Other"
    ))

fig.update_layout(
    title=f"Loss Dynamics - Method: {method}, Stage: {stage}",
    xaxis_title="Step",
    yaxis_title="Loss",
    template="plotly_white"
)

fig.show()
