# author: Michael Hüppe
# date: 11.11.2024
# project: utils/util_visualizeModel.py
# Function to plot the metrics
import numpy as np
import plotly.graph_objects as go


def plot_metrics(history, metric="loss", renderer=None, epochs=None):
    losses = history.history[metric]
    if not epochs:
        epochs = len(history.history["loss"])
    val_losses = history.history[f"val_{metric}"]
    list_epochs = np.arange(1, epochs + 1)
    metric = " ".join([w.capitalize() for w in metric.split("_")])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list_epochs, y=losses, name=metric, line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=list_epochs, y=val_losses, name=f"Val {metric}", line=dict(color="#FF6600", width=2)))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text=metric)
    fig.update_layout(title=f"Evolution of {metric} across Epochs",
                      title_font=dict(size=20),
                      title_x=0.5,
                      height=500,
                      width=1200)

    fig.show(renderer)
