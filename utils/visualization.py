import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Bold


def plot_category_distribution(df: pd.DataFrame, category_col: str = "category") -> go.Figure:
    counts = df[category_col].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_sequence=PALETTE,
        title="Resume Distribution by Job Category",
        text="Count",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        margin=dict(t=50, b=40),
    )
    return fig


def plot_category_pie(df: pd.DataFrame, category_col: str = "category") -> go.Figure:
    counts = df[category_col].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=PALETTE),
        textinfo="label+percent",
        textposition="outside",
    ))
    fig.update_layout(
        title="Category Breakdown",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        margin=dict(t=50, b=20),
        showlegend=True,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> go.Figure:
    # Normalize for color, keep raw values as text
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted", tickfont=dict(size=11)),
        yaxis=dict(title="Actual", tickfont=dict(size=11), autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif"),
        margin=dict(t=50, b=50, l=80),
    )
    return fig


def plot_model_comparison(results: dict[str, dict]) -> go.Figure:
    """Bar chart comparing multiple model metrics side by side."""
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig = go.Figure()
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            marker_color=PALETTE[i],
        ))

    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis=dict(title="Score", range=[0, 1.15], showgrid=True, gridcolor="#eee"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=80),
    )
    return fig


def plot_confidence_gauge(confidence: float, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        number=dict(suffix="%", font=dict(size=36)),
        title=dict(text=f"Confidence — {label}", font=dict(size=16)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color="#2563EB"),
            steps=[
                dict(range=[0, 50], color="#FEE2E2"),
                dict(range=[50, 75], color="#FEF3C7"),
                dict(range=[75, 100], color="#D1FAE5"),
            ],
            threshold=dict(line=dict(color="#1E40AF", width=4), thickness=0.75, value=confidence * 100),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif"),
        height=280,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def plot_top_keywords(keywords: list[tuple[str, int]], top_n: int = 10) -> go.Figure:
    if not keywords:
        return go.Figure()

    words, counts = zip(*keywords[:top_n])
    fig = go.Figure(go.Bar(
        x=list(counts),
        y=list(words),
        orientation="h",
        marker_color=PALETTE[2],
        text=list(counts),
        textposition="outside",
    ))
    fig.update_layout(
        title="Top Keywords in Resume",
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        margin=dict(t=50, b=30, l=10),
    )
    return fig


def plot_accuracy_by_category(report_dict: dict) -> go.Figure:
    """F1 score per class from sklearn classification report dict."""
    rows = {
        k: v for k, v in report_dict.items()
        if isinstance(v, dict) and "f1-score" in v
    }
    if not rows:
        return go.Figure()

    labels = list(rows.keys())
    f1_scores = [rows[l]["f1-score"] for l in labels]

    fig = px.bar(
        x=labels,
        y=f1_scores,
        color=labels,
        color_discrete_sequence=PALETTE,
        title="F1 Score by Category",
        labels={"x": "Category", "y": "F1 Score"},
        text=[f"{s:.2f}" for s in f1_scores],
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        yaxis=dict(range=[0, 1.2], showgrid=True, gridcolor="#eee"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        margin=dict(t=50, b=40),
    )
    return fig
