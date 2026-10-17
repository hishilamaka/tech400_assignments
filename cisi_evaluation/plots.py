"""
plots.py

Figures for the evaluation report. All are saved as PNG files (no window is
opened), so they work the same in VS Code, a terminal, or a remote session.
"""

import matplotlib

matplotlib.use("Agg")  # write files only; no GUI backend needed
import matplotlib.pyplot as plt

# One fixed colour and line style per model, so a model looks the same in
# every figure.
COLORS = {
    "Random": "#9e9e9e",
    "TF-IDF": "#1f77b4",
    "BM25-set": "#ff7f0e",
    "BM25": "#2ca02c",
    "LM-JM 0.1": "#9467bd",
    "LM-JM 0.7": "#d62728",
}
STYLES = {
    "Random": ":", "TF-IDF": "-", "BM25-set": "--",
    "BM25": "-", "LM-JM 0.1": "--", "LM-JM 0.7": "-",
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results, path):
    """11-point interpolated precision-recall curve, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 5.2))
    levels = [i / 10 for i in range(11)]
    for name, result in results.items():
        ax.plot(levels, result["pr_curve"], STYLES[name], color=COLORS[name],
                label=name, linewidth=2, marker="o", markersize=3.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_title("11-point interpolated precision-recall curve (76 queries)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_metrics_vs_k(results, path):
    """How Precision@k, Recall@k and nDCG@k change as the user reads deeper."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("precision", "Precision@k"), ("recall", "Recall@k"), ("ndcg", "nDCG@k")]
    for ax, (key, title) in zip(axes, panels):
        for name, result in results.items():
            values = result["curves"][key]
            ax.plot(range(1, len(values) + 1), values, STYLES[name],
                    color=COLORS[name], label=name, linewidth=2)
        ax.set_xlabel("k (number of results examined)")
        ax.set_title(title)
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean over 76 queries")
    axes[0].legend(fontsize=8)
    _save(fig, path)


def plot_model_comparison(summary, path):
    """
    Bar chart of MAP and nDCG@10 per model with 95% bootstrap confidence
    intervals. Overlapping intervals warn that a difference may be noise.
    summary: {model: {metric: (mean, lower, upper)}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in zip(axes, ["AP", "nDCG@10"], ["MAP", "nDCG@10"]):
        names = list(summary)
        means = [summary[n][metric][0] for n in names]
        errors = [[summary[n][metric][0] - summary[n][metric][1] for n in names],
                  [summary[n][metric][2] - summary[n][metric][0] for n in names]]
        ax.bar(names, means, yerr=errors, capsize=4,
               color=[COLORS[n] for n in names], edgecolor="black", linewidth=0.6)
        # Print each value ABOVE its confidence interval so labels never
        # collide with the error bars (even for the tiny Random bar).
        for i, name in enumerate(names):
            upper = summary[name][metric][2]
            ax.text(i, upper + 0.008, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(summary[n][metric][2] for n in names) * 1.12)
        ax.set_title(f"{title} with 95% bootstrap confidence interval")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_per_query_scatter(per_query_by_model, pairs, path, metric="AP"):
    """
    One point per query, comparing two models. Points above the diagonal are
    queries where the model on the y-axis did better.
    pairs: [(model_x, model_y), ...]
    """
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (x_name, y_name) in zip(axes, pairs):
        qids = sorted(per_query_by_model[x_name])
        xs = [per_query_by_model[x_name][q][metric] for q in qids]
        ys = [per_query_by_model[y_name][q][metric] for q in qids]
        ax.scatter(xs, ys, s=28, alpha=0.7, color=COLORS[y_name], edgecolor="black", linewidth=0.4)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel(f"{x_name}  (average precision)")
        ax.set_ylabel(f"{y_name}  (average precision)")
        ax.set_title(f"Per-query AP: {y_name} vs {x_name}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    _save(fig, path)


def plot_query_properties(features, ap_values, correlations, path, model_name="BM25"):
    """Average precision against query length and against number of relevant docs."""
    qids = sorted(ap_values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    specs = [("n_terms", "number of index terms in the query"),
             ("n_relevant", "number of relevant documents (log scale)")]
    for ax, (key, label) in zip(axes, specs):
        xs = [features[q][key] for q in qids]
        ys = [ap_values[q] for q in qids]
        ax.scatter(xs, ys, s=28, alpha=0.7, color=COLORS[model_name], edgecolor="black", linewidth=0.4)
        if key == "n_relevant":
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel(f"average precision ({model_name})")
        ax.set_title(f"Spearman rho = {correlations[key]:+.2f}")
        ax.grid(alpha=0.3)
    _save(fig, path)


def plot_bm25_heatmaps(grids, k1_values, b_values, path, metrics=("AP", "nDCG@10")):
    """
    Side-by-side heatmaps of two metrics over the (k1, b) grid. The best cell
    of each is outlined, showing whether the metrics agree on the best setting.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5.4))
    for ax, metric in zip(axes, metrics):
        data = [[grids[metric][(k1, b)] for b in b_values] for k1 in k1_values]
        image = ax.imshow(data, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(b_values)))
        ax.set_yticks(range(len(k1_values)))
        ax.set_xticklabels([str(b) for b in b_values])
        ax.set_yticklabels([str(k) for k in k1_values])
        ax.set_xlabel("b (length normalisation)")
        ax.set_ylabel("k1 (term-frequency saturation)")
        ax.set_title(f"BM25 {'MAP' if metric == 'AP' else metric}")

        low = min(min(row) for row in data)
        high = max(max(row) for row in data)
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                dark = (value - low) / (high - low + 1e-12) > 0.6
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")

        # Outline the best cell.
        best = max(grids[metric], key=grids[metric].get)
        bi, bj = k1_values.index(best[0]), b_values.index(best[1])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=2.5))
        fig.colorbar(image, ax=ax)
    _save(fig, path)


def plot_lambda_sweep(sweep, path, metrics=("AP", "nDCG@10", "P@10", "R@20")):
    """Language-model scores as the Jelinek-Mercer weight lambda varies."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lambdas = sorted(sweep)
    for metric, color in zip(metrics, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]):
        ax.plot(lambdas, [sweep[lam][metric] for lam in lambdas], "o-", color=color,
                linewidth=2, label="MAP" if metric == "AP" else metric)
    ax.set_xlabel("lambda (weight on the collection model)")
    ax.set_ylabel("mean score over 76 queries")
    ax.set_title("Language model with Jelinek-Mercer smoothing: effect of lambda")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_length_effect(features, per_query_by_model, short_ids, long_ids, rho, p_value, path):
    """
    Left: for each query, how much better BM25 is than TF-IDF (AP difference)
    against query length. Right: MAP of three models for short vs long queries.
    """
    qids = sorted(per_query_by_model["BM25"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    xs = [features[q]["n_terms"] for q in qids]
    ys = [per_query_by_model["BM25"][q]["AP"] - per_query_by_model["TF-IDF"][q]["AP"] for q in qids]
    ax.scatter(xs, ys, s=30, alpha=0.7, color=COLORS["BM25"], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("number of index terms in the query")
    ax.set_ylabel("AP(BM25) - AP(TF-IDF)")
    ax.set_title(f"Above 0: BM25 better.  Spearman rho = {rho:+.2f}, p = {p_value:.3f}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    models = ["TF-IDF", "BM25", "LM-JM 0.7"]
    width = 0.26
    for i, model in enumerate(models):
        values = []
        for group in (short_ids, long_ids):
            values.append(sum(per_query_by_model[model][q]["AP"] for q in group) / len(group))
        positions = [0 + (i - 1) * width, 1 + (i - 1) * width]
        bars = ax.bar(positions, values, width, color=COLORS[model], edgecolor="black",
                      linewidth=0.6, label=model)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"short queries (n={len(short_ids)})", f"long queries (n={len(long_ids)})"])
    ax.set_ylabel("MAP")
    ax.set_title("MAP for short vs long queries (split at the median length)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
