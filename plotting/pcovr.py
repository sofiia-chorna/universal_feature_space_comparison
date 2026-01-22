import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, rcParams

MODEL_MAPPING = {
    "PET-MAD": "PET-MAD",
    "MACE": "MACE-MP-03b",
    "UMA": "UMA-S-1P1",
    "DPA": "DPA-3.1",
}

plt.style.use("seaborn-v0_8-paper")
rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 22,
        "legend.handletextpad": 0.2,
        "legend.labelspacing": 0.4,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1,
        "axes.grid": False,
    }
)

styling = {
    "MC3D": {"color": "#0033FF", "marker": "o"},
    "MC3D-random": {"color": "orange", "marker": "s"},
    "MC3D-rattled": {"color": "yellow", "marker": "^"},
    "MC3D-clusters": {"color": "#008000", "marker": "D"},
    "MC2D": {"color": "red", "marker": "v"},
    "MC3D-surfaces": {"color": "#00FF00", "marker": "P"},
    "SHIFTML-molcrys": {"color": "black", "marker": "X"},
    "SHIFTML-molfrags": {"color": "#FF00FF", "marker": "*"},
}
existing_subsets = [
    "MC3D-clusters",
    "MC3D",
    "MC3D-random",
    "MC3D-rattled",
    "MC2D",
    "MC3D-surfaces",
    "SHIFTML-molcrys",
    "SHIFTML-molfrags",
]


def get_triple_row_plot(
    datasets_1, datasets_8, labels, cbar_title="cohesive energy (eV/atom)", save_path=""
):
    N_COLS = len(datasets_1)

    fig = plt.figure(figsize=(14.5, 13), dpi=300, facecolor="white")

    gs = gridspec.GridSpec(
        nrows=5,
        ncols=N_COLS,
        width_ratios=[1] * N_COLS,
        height_ratios=[0.6, 1, 1, 1, 0.1],
        wspace=0.03,
        hspace=0.05,
        figure=fig,
    )

    plot_axes = []
    for c in range(N_COLS):
        ax_cumulant1 = fig.add_subplot(gs[1, c])
        ax_cumulant8 = fig.add_subplot(gs[2, c])
        ax_energy = fig.add_subplot(gs[3, c])
        plot_axes.extend([ax_cumulant1, ax_cumulant8, ax_energy])

    legend_axis = fig.add_subplot(gs[0, :])
    cbar_axis = fig.add_subplot(gs[4, :])

    all_energies_8 = np.concatenate([d[1] for d in datasets_8])
    vmin, vmax = np.min(all_energies_8), np.max(all_energies_8)

    for i, (pcovr, energy, title) in enumerate(datasets_1):
        ax = plot_axes[i * 3]

        if i == 0:
            pcovr_x = -pcovr[:, 0]
            pcovr_y = pcovr[:, 1]
        else:
            pcovr_x = pcovr[:, 0]
            pcovr_y = pcovr[:, 1]

        for subset in existing_subsets:
            mask = np.array([label == subset for label in labels])
            if np.sum(mask) > 0:
                ax.scatter(
                    pcovr_x[mask],
                    pcovr_y[mask],
                    c=styling[subset]["color"],
                    marker=styling[subset]["marker"],
                    s=25,
                    alpha=0.8,
                    linewidth=0.1,
                    edgecolors="black",
                    rasterized=True,
                    label=subset if i == 0 else "",
                    zorder=2,
                )

        mapped_title = MODEL_MAPPING.get(title)
        ax.set_title(mapped_title, pad=10, fontweight="normal")

        if i == 0:
            ax.set_ylabel("PCovR 2", labelpad=1)

        if i == N_COLS - 1:
            ax_right = ax.twinx()
            ax_right.set_ylabel("Mean features", rotation=-90, labelpad=35, fontsize=20)
            ax_right.tick_params(
                which="both",
                bottom=False,
                left=False,
                top=False,
                right=False,
                labelbottom=False,
                labelleft=False,
                labeltop=False,
                labelright=False,
            )

        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, (pcovr, energy, title) in enumerate(datasets_8):
        ax = plot_axes[i * 3 + 1]

        if i == 0:
            pcovr_x = -pcovr[:, 0]
            pcovr_y = pcovr[:, 1]
        else:
            pcovr_x = pcovr[:, 0]
            pcovr_y = pcovr[:, 1]

        for subset in existing_subsets:
            mask = np.array([label == subset for label in labels])
            if np.sum(mask) > 0:
                ax.scatter(
                    pcovr_x[mask],
                    pcovr_y[mask],
                    c=styling[subset]["color"],
                    marker=styling[subset]["marker"],
                    s=25,
                    alpha=0.8,
                    linewidth=0.1,
                    edgecolors="black",
                    rasterized=True,
                    zorder=2,
                )

        if i == 0:
            ax.set_ylabel("PCovR 2", labelpad=1)

        if i == N_COLS - 1:
            ax_right = ax.twinx()
            ax_right.set_ylabel(
                "Cumulant features", rotation=-90, labelpad=35, fontsize=20
            )
            ax_right.tick_params(
                which="both",
                bottom=False,
                left=False,
                top=False,
                right=False,
                labelbottom=False,
                labelleft=False,
                labeltop=False,
                labelright=False,
            )

        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, (pcovr, energy, title) in enumerate(datasets_8):
        ax = plot_axes[i * 3 + 2]

        if i == 0:
            pcovr_x = -pcovr[:, 0]
            pcovr_y = pcovr[:, 1]
        else:
            pcovr_x = pcovr[:, 0]
            pcovr_y = pcovr[:, 1]

        scatter_energy = ax.scatter(
            pcovr_x,
            pcovr_y,
            c=energy,
            s=25,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.85,
            rasterized=True,
            linewidth=0.1,
            edgecolors="black",
        )

        if i == 0:
            ax.set_ylabel("PCovR 2", labelpad=0.5)

        if i == N_COLS - 1:
            ax_right = ax.twinx()
            ax_right.set_ylabel(
                "Cumulant features\n by cohesive energy",
                rotation=-90,
                labelpad=45,
                fontsize=20,
            )
            ax_right.tick_params(
                which="both",
                bottom=False,
                left=False,
                top=False,
                right=False,
                labelbottom=False,
                labelleft=False,
                labeltop=False,
                labelright=False,
            )

        ax.set_xlabel("PCovR 1", labelpad=1)
        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker=styling[subset]["marker"],
            color="w",
            markerfacecolor=styling[subset]["color"],
            markeredgecolor="black",
            markersize=14,
            markeredgewidth=0.4,
            label=subset,
            linewidth=0,
        )
        for subset in existing_subsets
    ]

    legend_axis.axis("off")
    legend = legend_axis.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.65),
        ncol=min(4, len(legend_elements)),
        frameon=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        handletextpad=0.1,
        columnspacing=0.7,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(0.8)

    cbar = fig.colorbar(scatter_energy, cax=cbar_axis, orientation="horizontal")
    cbar.set_label(cbar_title, labelpad=12, fontsize=24)
    cbar_axis.xaxis.set_label_position("bottom")
    cbar_axis.xaxis.tick_bottom()
    cbar.ax.tick_params(labelsize=18, pad=5)

    pos = cbar_axis.get_position()
    cbar_axis.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height * 0.7])

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_pcovr_lips(datasets, labels_lips, save_path=""):
    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": 14,
            "legend.markerscale": 2,
            "legend.title_fontsize": 12,
            "legend.fontsize": 14,
            "legend.handletextpad": 0.6,
            "legend.labelspacing": 1.0,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
        }
    )

    N_MODELS = len(datasets) // 2

    fig = plt.figure(figsize=(14, 9.5), dpi=300)
    fig.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], wspace=0.04, hspace=0.04
    )

    axes = []
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2:
                legend_ax = fig.add_subplot(gs[i, j])
                legend_ax.axis("off")
            else:
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)

    subset_style = {
        "bulk": {"label": "Bulk", "color": "#0033FF", "marker": "o"},
        "surface": {"label": "Surface", "color": "orange", "marker": "s"},
        "surface_water": {"label": "Surface water", "color": "#FF00FF", "marker": "*"},
        "surface_hydrogen": {
            "label": "Surface hydrogen",
            "color": "#00FF00",
            "marker": "P",
        },
    }

    lips_pcovr = np.concatenate([d[0] for d in datasets[N_MODELS:]])

    x_min, x_max = lips_pcovr[:, 0].min(), lips_pcovr[:, 0].max()
    y_min, y_max = lips_pcovr[:, 1].min(), lips_pcovr[:, 1].max()

    MARGIN_FACTOR = 0.8
    x_range, y_range = x_max - x_min, y_max - y_min

    x_margin = MARGIN_FACTOR * x_range
    y_margin = MARGIN_FACTOR * y_range

    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim_normal = (y_min - y_margin, y_max + y_margin)
    y_lim_flipped = (y_max + y_margin, y_min - y_margin)

    def style_axis(ax):
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
            spine.set_visible(True)

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    for i, ax in enumerate(axes):
        if i < N_MODELS:
            mad_features, _, mad_name = datasets[i]
            lips_features, _, _ = datasets[i + N_MODELS]

            ax.scatter(
                mad_features[:, 0],
                mad_features[:, 1],
                color="grey",
                alpha=0.2,
                s=65,
                label="MAD test",
                edgecolors="none",
                rasterized=True,
            )

            for subset, style in subset_style.items():
                mask = np.array(labels_lips) == subset
                ax.scatter(
                    lips_features[mask, 0],
                    lips_features[mask, 1],
                    color=style["color"],
                    marker=style["marker"],
                    s=65,
                    alpha=0.9,
                    edgecolors="black",
                    linewidth=0.3,
                    label=style["label"],
                    rasterized=True,
                )

            mad_name = mad_name.split(" (MAD)")[0]

            ax.text(
                0.5,
                0.03,
                mad_name,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
            )

            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.set_xlim(x_lim)

            if i == 1 or i == 2:
                ax.set_ylim(y_lim_flipped)
            else:
                ax.set_ylim(y_lim_normal)

            style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    legend_ax.legend(
        handles=unique_handles,
        labels=unique_labels,
        ncol=1,
        loc="center",
        frameon=False,
        fontsize=20,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
