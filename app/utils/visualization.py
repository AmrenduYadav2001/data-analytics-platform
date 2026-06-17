import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import uuid


# ✅ Global style — applied to all charts
def set_style():
    plt.rcParams.update({
        "figure.dpi":        150,       # high quality
        "savefig.dpi":       150,
        "font.family":       "Arial",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def save_chart(fig, output_dir, prefix):
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path     = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"/static/charts/{filename}"


def generate_charts(file_path, output_dir="app/static/charts"):

    os.makedirs(output_dir, exist_ok=True)
    set_style()

    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"File could not be read: {e}")

    charts     = {}
    numeric_df = df.select_dtypes(include="number")
    cat_df     = df.select_dtypes(include="object")

    # --- 1. Missing Values Chart ---
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(missing.index, missing.values, color=COLORS[:len(missing)], edgecolor="white", linewidth=0.8)
        ax.set_title("Missing Values Per Column")
        ax.set_ylabel("Missing Count")
        ax.set_xlabel("Columns")
        plt.xticks(rotation=45, ha="right")

        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

        charts["missing_values"] = save_chart(fig, output_dir, "missing")

    # --- 2. Correlation Heatmap ---
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        n    = len(corr)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))

        im  = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

        ax.set_xticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.columns, fontsize=9)
        ax.set_title("Correlation Heatmap")

        # Add values inside cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")

        charts["correlation"] = save_chart(fig, output_dir, "corr")

    # --- 3. Histograms with KDE line ---
    if not numeric_df.empty:
        hist_images = []
        for col in numeric_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            data = numeric_df[col].dropna()

            ax.hist(data, bins=25, color=COLORS[0], edgecolor="white",
                    linewidth=0.6, alpha=0.85, density=True)

            # KDE line
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x   = np.linspace(data.min(), data.max(), 200)
                ax.plot(x, kde(x), color="#DD8452", linewidth=2, label="KDE")
                ax.legend()
            except Exception:
                pass

            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            hist_images.append(save_chart(fig, output_dir, f"hist_{col}"))

        charts["histograms"] = hist_images

    # --- 4. Outlier Boxplots ---
    if not numeric_df.empty:
        box_images = []
        for col in numeric_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            data = numeric_df[col].dropna()

            bp = ax.boxplot(data, vert=False, patch_artist=True,
                            boxprops=dict(facecolor=COLORS[1], color=COLORS[1], alpha=0.7),
                            medianprops=dict(color="red", linewidth=2),
                            flierprops=dict(marker="o", markerfacecolor=COLORS[3],
                                            markersize=4, alpha=0.5))

            ax.set_title(f"Outlier Detection — {col}")
            ax.set_xlabel(col)
            ax.set_yticks([])
            box_images.append(save_chart(fig, output_dir, f"box_{col}"))

        charts["boxplots"] = box_images

    # --- 5. Top Categorical Value Counts ---
    if not cat_df.empty:
        cat_images = []
        for col in cat_df.columns[:5]:  # max 5 categorical columns
            top = df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.barh(top.index.astype(str), top.values,
                           color=COLORS[:len(top)], edgecolor="white")
            ax.set_title(f"Top Values — {col}")
            ax.set_xlabel("Count")
            ax.invert_yaxis()

            for bar, val in zip(bars, top.values):
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=9, fontweight="bold")

            cat_images.append(save_chart(fig, output_dir, f"cat_{col}"))
        charts["categorical"] = cat_images

    return charts