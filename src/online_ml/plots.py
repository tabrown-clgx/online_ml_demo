"""Headless figures, imported only by the UI."""

from matplotlib.figure import Figure


def learning_curve(history: list[dict], comparisons: list[tuple[dict, list[dict]]] = ()) -> Figure:
    figure = Figure(figsize=(8, 3.5), layout="constrained")
    ax = figure.subplots()
    has_values = False
    for config, metrics in comparisons:
        points = [m for m in metrics if m["probe"]["balanced_accuracy"] is not None]
        if points:
            ax.plot([m["n_labels"] for m in points],
                    [m["probe"]["balanced_accuracy"] for m in points],
                    alpha=0.4, linestyle="--",
                    label=f"{config.get('strategy', 'manual')} · {config['run_id'][-8:]}")
            has_values = True
    points = [m for m in history if m["probe"]["balanced_accuracy"] is not None]
    if points:
        ax.plot([m["n_labels"] for m in points],
                [m["probe"]["balanced_accuracy"] for m in points],
                marker="o", markersize=3, color="#0d9488", label="Current probe")
        ax.axhline(points[0]["baseline"]["balanced_accuracy"], color="#64748b",
                   linestyle=":", label="Zero-shot baseline")
        has_values = True
    if has_values:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "Add a reviewed evaluation set to see a learning curve",
                transform=ax.transAxes, ha="center", va="center", color="#64748b")
    ax.set(xlabel="Current feedback labels", ylabel="Balanced accuracy", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.15)
    return figure
