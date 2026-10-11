"""
plots.py

Figures for the report. Saved as PNG files (no window is opened, so this
works the same in VS Code, a plain terminal, or a remote session).
"""

import matplotlib

matplotlib.use("Agg")  # file output only, no GUI needed
import matplotlib.pyplot as plt


def plot_pr_curves(curves, path):
    """
    curves: {"label": [11 interpolated precision values], ...}
    Draws the standard 11-point interpolated precision-recall curve.
    """
    recall_levels = [i / 10 for i in range(11)]
    fig, ax = plt.subplots(figsize=(6.5, 5))

    styles = ["o-", "s--", "^:"]
    for (label, values), style in zip(curves.items(), styles):
        ax.plot(recall_levels, values, style, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_title("11-Point Precision-Recall Curve (CISI, 76 judged queries)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_param_heatmap(grid, k1_values, b_values, path):
    """grid: {(k1, b): MAP}. Rows = k1, columns = b."""
    data = [[grid[(k1, b)] for b in b_values] for k1 in k1_values]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(b_values)))
    ax.set_yticks(range(len(k1_values)))
    ax.set_xticklabels([str(b) for b in b_values])
    ax.set_yticklabels([str(k) for k in k1_values])
    ax.set_xlabel("b (length normalisation)")
    ax.set_ylabel("k1 (term-frequency saturation)")
    ax.set_title("BM25 MAP for different k1 / b values")

    lo, hi = min(min(r) for r in data), max(max(r) for r in data)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            dark = (value - lo) / (hi - lo + 1e-12) > 0.6
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if dark else "black", fontsize=8)

    fig.colorbar(im, ax=ax, label="MAP")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
