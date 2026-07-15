"""Plotting helpers live here."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ggplot2 line widths are in mm; this is its internal mm -> pt factor
_GG_PT = 72.27 / 25.4  # ≈ 2.845


def set_pubr_theme(
    base_size=12,          # master font size — everything scales off this
    base_family="Arial",   # falls back to sans-serif if not installed
    linewidth_mm=0.5,      # ggpubr axis.line / tick linewidth
    legend="top",          # "top" | "bottom" | "left" | "right" | "none"
    text_color="black",
):
    """Apply a matplotlib/seaborn style mimicking ggpubr::theme_pubr()."""

    # Proportional font sizes (mirrors ggplot2's rel() multipliers)
    axis_title = 1.0 * base_size
    axis_text = 0.8 * base_size
    title = 1.2 * base_size
    legend_txt = 0.8 * base_size
    legend_ttl = 1.0 * base_size

    lw_pt = linewidth_mm * _GG_PT      # spines & ticks
    tick_len = base_size / 4           # ggpubr tick length (pt)

    rc = {
        # --- font ---
        "font.family": base_family,
        "font.size": base_size,
        "text.color": text_color,
        "axes.titlesize": title,
        "axes.labelsize": axis_title,
        "axes.labelcolor": text_color,
        "xtick.labelsize": axis_text,
        "ytick.labelsize": axis_text,
        "legend.fontsize": legend_txt,
        "legend.title_fontsize": legend_ttl,

        # --- background / panel ---
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,

        # --- spines: only left + bottom (= ggpubr axis.line) ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": text_color,
        "axes.linewidth": lw_pt,

        # --- ticks ---
        "xtick.color": text_color,
        "ytick.color": text_color,
        "xtick.labelcolor": text_color,
        "ytick.labelcolor": text_color,
        "xtick.major.width": lw_pt,
        "ytick.major.width": lw_pt,
        "xtick.major.size": tick_len,
        "ytick.major.size": tick_len,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # --- legend (no key background, matches legend.key = blank) ---
        "legend.frameon": False,
        "legend.loc": "upper center" if legend in ("top", "bottom") else "best",

        # --- default line/marker weight for plotted data ---
        "lines.linewidth": linewidth_mm * _GG_PT,
        "patch.linewidth": lw_pt,        # bar/box outlines
        "patch.edgecolor": text_color,
    }

    mpl.rcParams.update(rc)
    return rc


def pubr_legend(ax, position="top", ncol=None, relabel=None, **kwargs):
    """Place a legend outside the axes, ggpubr-style (default: above the plot).

    position: "top" | "bottom" | "left" | "right"
    ncol: legend columns (defaults to one row for top/bottom, one column otherwise)
    relabel: optional {original: display} dict to remap legend entry text; keys
        not present are left unchanged
    Extra kwargs are passed through to ax.legend().
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        # seaborn (e.g. boxplot with hue=) builds its own legend whose handles
        # aren't returned above; pull them off the existing legend instead
        existing = ax.get_legend()
        if existing is not None:
            handles = existing.legend_handles
            labels = [t.get_text() for t in existing.texts]
    n = len(labels)
    if n == 0:
        return None  # nothing to put in a legend

    if relabel:
        labels = [relabel.get(lbl, lbl) for lbl in labels]

    anchors = {
        "top": dict(loc="lower center", bbox_to_anchor=(0.5, 1.02)),
        "bottom": dict(loc="upper center", bbox_to_anchor=(0.5, -0.12)),
        "left": dict(loc="center right", bbox_to_anchor=(-0.12, 0.5)),
        "right": dict(loc="center left", bbox_to_anchor=(1.02, 0.5)),
    }
    if position not in anchors:
        raise ValueError(f"position must be one of {list(anchors)}, got {position!r}")

    if ncol is None:
        ncol = n if position in ("top", "bottom") else 1
    ncol = max(ncol, 1)

    opts = dict(frameon=False, ncol=ncol)
    opts.update(anchors[position])
    opts.update(kwargs)
    return ax.legend(handles, labels, **opts)


def pubr_fill(ax, alpha=0.5, source="face", color_lines=True, linewidth=None, legend=None):
    """Recolour patches so the outline is solid and the fill is a faded version.

    Mimics ggpubr's fill = group / colour = group with alpha on the fill only.
    For each patch (bars, boxes, etc.) the base hue is taken from `source`, the
    outline is set to that hue at full opacity, and the fill to the same hue at
    `alpha`.

    alpha: fill opacity (0–1)
    source: "face" take the base hue from the current fill, "edge" from the
        current outline
    color_lines: for box plots, also recolour the whiskers, caps, and median/mean
        lines to match each box (seaborn draws these as separate Line2D objects
        that otherwise stay the default colour)
    linewidth: outline width for patches and box lines. seaborn sets its own
        width at draw time (often thinner than the axis spines), so this is reset
        here. Defaults to the spine width so outlines match the axes.
    legend: an optional matplotlib Legend whose swatches are restyled the same
        way (translucent fill + solid same-hue outline), since the legend is
        built before this recolour and otherwise keeps the original look. Pass
        ax.get_legend() for an axes-level plot or a figure-level seaborn legend
        (e.g. g._legend). None (default) leaves the legend untouched.
    """
    if source not in ("face", "edge"):
        raise ValueError(f"source must be 'face' or 'edge', got {source!r}")

    if linewidth is None:
        linewidth = mpl.rcParams["axes.linewidth"]   # match the spines

    for patch in ax.patches:
        base = patch.get_facecolor() if source == "face" else patch.get_edgecolor()
        rgb = to_rgba(base)[:3]
        patch.set_edgecolor((*rgb, 1.0))
        patch.set_facecolor((*rgb, alpha))
        patch.set_linewidth(linewidth)

    if color_lines and ax.patches and ax.lines:
        # Match each line (whisker/cap/median/fliers) to its box by x-position.
        # Boxes can have different line counts (e.g. only some have outliers),
        # so positional slicing misaligns — geometry is reliable.
        box_centers = [
            (patch.get_path().vertices[:, 0].mean(), patch) for patch in ax.patches
        ]
        for line in ax.lines:
            xs = line.get_xdata()
            if len(xs) == 0:
                continue
            lx = sum(xs) / len(xs)
            patch = min(box_centers, key=lambda c: abs(c[0] - lx))[1]
            rgb = to_rgba(patch.get_facecolor())[:3]
            line.set_color((*rgb, 1.0))
            line.set_markeredgecolor((*rgb, 1.0))
            line.set_linewidth(linewidth)

    if legend is not None:
        # the legend is built before this recolour, so its swatches keep the
        # original look — restyle each handle to match the faded fill
        for handle in legend.legend_handles:
            if not hasattr(handle, "set_facecolor"):
                continue  # skip non-patch handles (e.g. Line2D entries)
            base = handle.get_facecolor() if source == "face" else handle.get_edgecolor()
            rgb = to_rgba(base)[:3]
            handle.set_edgecolor((*rgb, 1.0))
            handle.set_facecolor((*rgb, alpha))
            handle.set_linewidth(linewidth)
    return ax


def plot_loss_curve(history_dict, file_path):

    """ Plots a training and validation loss curve from a single histry dict from a training run"""

    set_pubr_theme()

    
    fold = str(history_dict.get("loop")[0]).title()
        
    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["train_loss"], label="Train Loss")
    if history_dict.get("val_loss"):
        plt.plot(history_dict["val_loss"], label="Val Loss") 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if history_dict.get("val_loss"):
        plt.suptitle(f"Training vs Validation Loss")
        plt.title(f"{fold} Fold | Test region: {history_dict["test_region"][0]} | Val region: {history_dict["val_region"][0]}")
    else:
        plt.suptitle(f"Training Loss")
        plt.title(f"{fold} Fold | Test region: {history_dict["test_region"][0]}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(file_path), dpi=150)
    plt.show()


def plot_loss_curve_from_df(df, file_path, test_region, val_region = None, fold = None, facet = False):

    """ Plots training and validation loss curves from the dataframe contructed from concatenated history_dicts"""

    test_regions = df["test_regions"].unique()
    if val_region:
        val_regions = df["val_regions"].unique()

    if not test_region or test_region not in test_regions:

    if fold:
        if fold not in ["inner", "outer"]:
            raise ValueError('fold argument must be either "inner" or "outer"')
    if not fold or fold== "outer":
            
        # plot single plot of the test region train loss
            
    elif fold == "inner" and not facet:
        if not val_region or val_region not in val_regions:
            raise ValueError(f"Please specify a val_region from {val_regions}")
            
        #plot a single fig with the test and val region

    else: 
        # plot facet diagram of the val regions for that test region


