import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Polygon, Rectangle

DESIRED_ORDER = ["MACE-MP-0B3", "PET-MAD", "DPA-3.1", "UMA-S-1P1"]

MODEL_MAPPING = {
    "PETMAD": "PET-MAD",
    "MACE": "MACE-MP-0B3",
    "UMA": "UMA-S-1P1",
    "DPA": "DPA-3.1",
}

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 26,
        # 'font.size': 21,
        #'font.size': 16,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.grid": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 2.4,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def plot_bidirectional_heatmaps(
    error_matrix1,
    annot_matrix1,
    error_matrix2,
    annot_matrix2,
    title1="Matrix 1",
    title2="Matrix 2",
    xlabel="target",
    ylabel="source",
    cbar_label="Value",
    save_path=None,
    title=None,
    rotate_ticks=False,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=300)
    plt.subplots_adjust(wspace=0.04)

    cbar_ax = fig.add_axes([0.915, 0.21, 0.03, 0.57])

    heatmap1 = sns.heatmap(
        error_matrix1,
        annot=annot_matrix1,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=1.0,
        cbar=False,
        ax=ax1,
        square=True,
        xticklabels=error_matrix1.columns,
        yticklabels=error_matrix1.index,
    )

    for collection in heatmap1.collections:
        collection.set_edgecolor("black")
        collection.set_linewidth(0.4)

    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    ax1.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
    )

    ax1.set_xlabel(xlabel, labelpad=10)
    ax1.set_ylabel(ylabel)
    ax1.tick_params(axis="both", which="both", length=0, pad=10)

    if rotate_ticks:
        ax1.set_xticklabels(ax1.get_xticklabels(), ha="right", rotation=30)
    else:
        ax1.set_yticklabels(ax1.get_yticklabels(), ha="right", rotation=0)

    ax1.set_title(title1, pad=12)

    heatmap2 = sns.heatmap(
        error_matrix2,
        annot=annot_matrix2,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=1.01,
        cbar_ax=cbar_ax,
        ax=ax2,
        square=True,
        xticklabels=error_matrix2.columns,
        yticklabels=False,
    )

    for collection in heatmap2.collections:
        collection.set_edgecolor("black")
        collection.set_linewidth(0.4)

    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    ax2.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
    )

    ax2.set_xlabel(xlabel, labelpad=10)
    ax2.set_ylabel("")
    ax2.tick_params(axis="both", which="both", length=0, pad=10)

    if rotate_ticks:
        ax2.set_xticklabels(ax2.get_xticklabels(), ha="right", rotation=30)
    else:
        ax2.set_xticklabels(ax2.get_xticklabels(), ha="center", rotation=0)
    ax2.set_title(title2, pad=12)

    cbar = heatmap2.collections[0].colorbar
    cbar.set_label(cbar_label, rotation=270, labelpad=25, fontsize=22)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    if title:
        fig.suptitle(title, fontsize=10, y=0.9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def get_matrices(results, desired_order=DESIRED_ORDER, model_mapping=MODEL_MAPPING):
    all_methods = set(results.keys())
    for inner_dict in results.values():
        all_methods.update(inner_dict.keys())

    methods = sorted(list(all_methods))
    mapped_methods = [model_mapping.get(method) for method in methods]

    error_matrix_a_to_b = pd.DataFrame(
        np.nan, index=mapped_methods, columns=mapped_methods
    )
    annot_matrix_a_to_b = pd.DataFrame(
        "", index=mapped_methods, columns=mapped_methods, dtype="U20"
    )

    error_matrix_b_to_a = pd.DataFrame(
        np.nan, index=mapped_methods, columns=mapped_methods
    )
    annot_matrix_b_to_a = pd.DataFrame(
        "", index=mapped_methods, columns=mapped_methods, dtype="U20"
    )

    for ref_model in results.keys():
        mapped_ref_model = model_mapping.get(ref_model)

        for source_model, data in results[ref_model].items():

            mapped_source_model = model_mapping.get(source_model)

            a_to_b_error = data["A_to_B"]
            error_matrix_a_to_b.loc[mapped_source_model, mapped_ref_model] = (
                a_to_b_error
            )
            annot_matrix_a_to_b.loc[mapped_source_model, mapped_ref_model] = (
                f"{a_to_b_error:.2f}"
            )

            b_to_a_error = data["B_to_A"]
            error_matrix_b_to_a.loc[mapped_ref_model, mapped_source_model] = (
                b_to_a_error
            )
            annot_matrix_b_to_a.loc[mapped_ref_model, mapped_source_model] = (
                f"{b_to_a_error:.2f}"
            )

    final_order = [m for m in desired_order if m in error_matrix_a_to_b.index]

    error_matrix_a_to_b = error_matrix_a_to_b.reindex(
        index=final_order, columns=final_order
    )
    annot_matrix_a_to_b = annot_matrix_a_to_b.reindex(
        index=final_order, columns=final_order
    )

    error_matrix_b_to_a = error_matrix_b_to_a.reindex(
        index=final_order, columns=final_order
    )
    annot_matrix_b_to_a = annot_matrix_b_to_a.reindex(
        index=final_order, columns=final_order
    )

    return error_matrix_a_to_b, annot_matrix_a_to_b


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def get_2xN_matrix(results, order_list, model_mapping=MODEL_MAPPING):
    ll_to_bb_errors = {}
    bb_to_ll_errors = {}

    for bb_key, ll_entries in results.items():
        target_head = bb_key.split("_")[0]
        source_key = f"{target_head}_LL"

        if source_key in ll_entries:
            data = ll_entries[source_key]
            display_head = model_mapping.get(target_head, target_head)

            if display_head in order_list:
                ll_to_bb_errors[display_head] = data["A_to_B"]
                bb_to_ll_errors[display_head] = data["B_to_A"]

    error_data = {
        "LL → BB": pd.Series(ll_to_bb_errors),
        "BB → LL": pd.Series(bb_to_ll_errors),
    }
    error_matrix = pd.DataFrame(error_data).T.reindex(columns=order_list)

    annot_matrix = error_matrix.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    return error_matrix, annot_matrix


def plot_2xN_matrix(
    error_matrix_gfre,
    annot_matrix_gfre,
    error_matrix_lfre,
    annot_matrix_lfre,
    save_path,
    rotate_ticks=False,
):
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 17,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.grid": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig = plt.figure(figsize=(18, 7), dpi=300)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[4, 0.3],
        width_ratios=[1, 1],
        wspace=0.08,
        hspace=0.2,
    )

    ax_gfre = fig.add_subplot(gs[0, 0])
    ax_lfre = fig.add_subplot(gs[0, 1])
    cbar_ax_gfre = fig.add_subplot(gs[1, 0])
    cbar_ax_lfre = fig.add_subplot(gs[1, 1])

    plot_list = [
        (ax_gfre, error_matrix_gfre, annot_matrix_gfre, "GFRE", cbar_ax_gfre, True),
        (ax_lfre, error_matrix_lfre, annot_matrix_lfre, "LFRE", cbar_ax_lfre, False),
    ]

    for ax, err_matrix, ann_matrix, etype, cbar_ax, is_leftmost in plot_list:

        heatmap = sns.heatmap(
            err_matrix,
            annot=ann_matrix,
            fmt="",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=True,
            cbar_ax=cbar_ax,
            cbar_kws={
                "label": f"{etype} Error",
                "orientation": "horizontal",
                "ticks": np.linspace(0, 1, 5).round(1),
            },
            ax=ax,
            square=True,
            annot_kws={"size": 34},
            linewidths=0.4,
            linecolor="black",
        )

        ax.set_title(f"{etype}", fontsize=28, pad=10)
        ax.set_xlabel("head", fontsize=28, labelpad=10)

        if not is_leftmost:
            ax.tick_params(axis="y", length=0)
            plt.setp(ax.get_yticklabels(), visible=False)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        border = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(border)

        if rotate_ticks:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=24)
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=24)

        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=24)
        ax.tick_params(axis="both", which="both", length=0, pad=12)

        cbar = heatmap.collections[0].colorbar
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.5)

        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.xaxis.tick_bottom()
        cbar.set_label("reconstruction error", labelpad=10, fontsize=26)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


class DoubleTriangle:
    def __init__(self, color_upper_right, color_lower_left):
        self.color_upper_right = color_upper_right
        self.color_lower_left = color_lower_left


class DoubleTriangleHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x, y = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height

        triangle_upper_right = mpatches.Polygon(
            [[x, y + h], [x + w, y + h], [x + w, y]],
            facecolor=orig_handle.color_upper_right,
            edgecolor="black",
            lw=1,
            transform=handlebox.get_transform(),
        )

        triangle_lower_left = mpatches.Polygon(
            [[x, y], [x + w, y], [x, y + h]],
            facecolor=orig_handle.color_lower_left,
            edgecolor="black",
            lw=1,
            transform=handlebox.get_transform(),
        )

        handlebox.add_artist(triangle_upper_right)
        handlebox.add_artist(triangle_lower_left)
        return triangle_upper_right, triangle_lower_left


def plot_split_diagonal_heatmaps(
    error_gfre_mad,
    error_gfre_org,
    error_lfre_mad,
    error_lfre_org,
    xlabel="Target",
    ylabel="Source",
    cbar_label="Reconstruction Error",
    save_path=None,
    title=None,
):
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 30,
            "axes.titlesize": 32,
            "axes.labelsize": 26,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 16,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.grid": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 2.4,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    plt.subplots_adjust(wspace=0.03)

    cbar_ax = fig.add_axes([0.915, 0.113, 0.02, 0.6])

    def plot_split_heatmap(ax, mad_matrix, org_matrix, title):
        n = len(mad_matrix)

        sns.heatmap(
            np.zeros((n, n)),
            annot=False,
            fmt="",
            cmap="viridis",
            vmin=0,
            vmax=1.0,
            square=True,
            cbar=False,
            ax=ax,
            xticklabels=mad_matrix.columns,
            yticklabels=mad_matrix.index,
        )

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        for i in range(n):
            for j in range(n):
                x0, y0 = j, i
                x1, y1 = j + 1, i + 1

                mad_val = mad_matrix.iloc[i, j]
                org_val = org_matrix.iloc[i, j]

                norm_mad = mad_val
                norm_org = org_val

                mad_triangle = Polygon(
                    [[x0, y0], [x1, y0], [x1, y1]],
                    closed=True,
                    color=plt.cm.viridis(norm_mad),
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax.add_patch(mad_triangle)

                org_triangle = Polygon(
                    [[x0, y0], [x0, y1], [x1, y1]],
                    closed=True,
                    color=plt.cm.viridis(norm_org),
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax.add_patch(org_triangle)

                ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.8)

                ax.text(
                    x1 - 0.15,
                    y0 + 0.15,
                    f"{mad_val:.2f}",
                    ha="right",
                    va="top",
                    fontsize=32,
                    color="white" if norm_mad < 0.5 else "black",
                )
                ax.text(
                    x0 + 0.15,
                    y1 - 0.15,
                    f"{org_val:.2f}",
                    ha="left",
                    va="bottom",
                    fontsize=32,
                    color="white" if norm_org < 0.5 else "black",
                )

        for k in range(n + 1):
            ax.plot([0, n], [k, k], color="black", linewidth=1.0)
            ax.plot([k, k], [0, n], color="black", linewidth=1.0)

        ax.set_xlabel(xlabel, labelpad=10)
        ax.tick_params(axis="both", which="both", length=0, pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), ha="center", rotation=0)
        ax.set_title(title, pad=12)

    plot_split_heatmap(ax1, error_gfre_mad, error_gfre_org, "GFRE")
    ax1.set_ylabel(ylabel)
    ax1.set_yticklabels(ax1.get_yticklabels(), ha="right", rotation=0)

    plot_split_heatmap(ax2, error_lfre_mad, error_lfre_org, "LFRE")
    ax2.set_ylabel("")
    ax2.set_yticklabels([])

    norm = plt.Normalize(0, 1.0)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=35, fontsize=28)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    plt.legend(
        [DoubleTriangle("black", "white"), DoubleTriangle("white", "black")],
        ["MP-0a (upper)", "OFF23 (lower)"],
        handler_map={DoubleTriangle: DoubleTriangleHandler()},
        loc="upper center",
        bbox_to_anchor=(5.6, 1.34),
        ncol=1,
        frameon=False,
        fancybox=False,
        edgecolor="black",
        fontsize=26,
    )

    if title:
        fig.suptitle(title, fontsize=24, y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")

    plt.show()
