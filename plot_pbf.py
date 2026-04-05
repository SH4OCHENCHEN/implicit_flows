import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 180
})

BLUE = "#3B82C4"
BLUE_DARK = "#1D4E89"
BLUE_LIGHT = "#DCEBFA"
RED = "#D94841"
ORANGE = "#E68A2E"
GREEN = "#2E8B57"
GRAY = "#6B7280"
BLACK = "#111827"

def clean_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

def panel_title(ax, text):
    ax.text(0.02, 0.98, text, ha="left", va="top", fontsize=12, color=BLACK, weight="medium")
    ax.plot([0.02, 0.98], [0.935, 0.935], color="#B7C7D9", lw=0.9)

def arrow(ax, p0, p1, color=BLACK, lw=1.2, ms=12, ls="-", alpha=1.0, z=4, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms,
        lw=lw, color=color, linestyle=ls, alpha=alpha, zorder=z,
        capstyle="round", joinstyle="round"
    ))

def label_box(ax, x, y, text, w=0.16, h=0.08, fc="white", ec="#9CA3AF", color=BLACK, fs=9):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=5
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=6)

def gaussian(ax, cx, cy, sx=0.035, sy=0.07, base=0.16, color=BLUE, alpha=0.35, lw=1.2, jagged=False, z=3):
    x = np.linspace(cx - base/2, cx + base/2, 240)
    y = cy + sy * np.exp(-((x - cx) ** 2) / (2 * sx ** 2))
    if jagged:
        y = y + 0.010*np.sin(np.linspace(0, 12*np.pi, len(x))) + 0.004*np.sign(np.sin(np.linspace(0, 27*np.pi, len(x))))
    ax.fill_between(x, cy, y, color=color, alpha=alpha, zorder=z)
    ax.plot(x, y, color=color, lw=lw, zorder=z+0.1)
    return x, y

def scatter_cloud(ax, center, spread=(0.05, 0.08), n=40, color=BLUE, alpha=0.45, seed=0, s=14, z=4):
    rng = np.random.default_rng(seed)
    cx, cy = center
    xs = rng.normal(cx, spread[0], n)
    ys = rng.normal(cy, spread[1], n)
    ax.scatter(xs, ys, s=s, color=color, alpha=alpha, edgecolors="none", zorder=z)
    return xs, ys

def noise_disk(ax, center, r=0.065, text="noise"):
    c = Circle(center, r, facecolor=BLUE, edgecolor=BLUE_DARK, lw=0.8, alpha=0.95, zorder=3)
    ax.add_patch(c)
    ax.text(center[0], center[1], text, ha="center", va="center", color="white", fontsize=10, weight="medium", zorder=4)

def smooth_path(x0, x1, y_base=0.5, amp=0.06, slope=0.0, phase=0.0):
    xx = np.linspace(x0, x1, 250)
    yy = y_base + amp*np.sin(np.linspace(phase, phase+2.2*np.pi, len(xx))) + slope*(xx-x0)
    return xx, yy

def draw_panel_A(ax):
    clean_axis(ax)
    panel_title(ax, "Panel A   Prior flow-based Bellman update")

    ax.text(0.07, 0.84, "current return flow", fontsize=10, color=BLACK)
    noise_disk(ax, (0.10, 0.68), r=0.058, text="noise")
    xx1, yy1 = smooth_path(0.17, 0.56, y_base=0.68, amp=0.03, slope=0.008, phase=0.2)
    ax.plot(xx1, yy1, color=BLUE, lw=2.0, zorder=3)
    arrow(ax, (0.54, yy1[-1]), (0.60, yy1[-1]+0.002), color=BLUE, lw=2.0, ms=12)

    gaussian(ax, 0.78, 0.61, sx=0.05, sy=0.10, base=0.24, color=RED, alpha=0.18, jagged=True, lw=1.0)
    ax.text(0.78, 0.75, "target for current return", ha="center", fontsize=10, color=BLACK)
    ax.text(0.78, 0.55, r"$r + \gamma Z(s',a')$", ha="center", fontsize=10, color=RED)

    ax.text(0.07, 0.42, "next return flow", fontsize=10, color=BLACK)
    noise_disk(ax, (0.10, 0.27), r=0.058, text="noise")
    xx2, yy2 = smooth_path(0.17, 0.52, y_base=0.27, amp=0.028, slope=0.008, phase=1.1)
    ax.plot(xx2, yy2, color=BLUE, lw=2.0, zorder=3)
    arrow(ax, (0.50, yy2[-1]), (0.56, yy2[-1]+0.002), color=BLUE, lw=2.0, ms=12)
    gaussian(ax, 0.68, 0.22, sx=0.037, sy=0.085, base=0.20, color=BLUE, alpha=0.30, lw=1.0)
    ax.text(0.68, 0.35, "fully sampled next return", ha="center", fontsize=10, color=BLACK)
    ax.text(0.68, 0.17, r"$Z(s',a')$", ha="center", fontsize=10, color=BLUE_DARK)

    ax.text(0.60, 0.47, "Bellman target", fontsize=9, color=GRAY, ha="left")
    arrow(ax, (0.69, 0.30), (0.73, 0.52), color=RED, lw=1.2, ms=11)
    arrow(ax, (0.73, 0.52), (0.73, 0.57), color=RED, lw=1.2, ms=11)
    ax.plot([0.70, 0.86], [0.50, 0.50], color=RED, lw=1.0, ls="--", alpha=0.9)
    ax.text(0.83, 0.48, "used as supervision target", fontsize=8.8, color=RED, ha="right")

    for x in [0.71, 0.78, 0.85]:
        arrow(ax, (x, 0.70), (x+0.018, 0.63-0.018*np.sin(12*x)), color=RED, lw=0.8, ms=8, alpha=0.85)
    ax.text(0.88, 0.62, "error\naccumulation", ha="center", va="center", fontsize=8.5, color=RED)

    ax.text(0.49, 0.08,
            "first fully roll out the next-return distribution,\nthen use it as the target to train the current-return flow",
            ha="center", va="center", fontsize=9.5, color=BLACK)

def draw_panel_B(ax):
    clean_axis(ax)
    panel_title(ax, "Panel B   Our pathwise / any-$t$ Bellman update")

    scatter_cloud(ax, (0.13, 0.47), spread=(0.03, 0.10), n=50, color=BLUE, alpha=0.42, seed=1, s=12)
    ax.text(0.13, 0.79, "noise", ha="center", fontsize=10)

    scatter_cloud(ax, (0.87, 0.47), spread=(0.04, 0.12), n=55, color=GREEN, alpha=0.42, seed=2, s=12)
    ax.text(0.87, 0.79, "return", ha="center", fontsize=10)

    xx = np.linspace(0.23, 0.77, 300)
    yy = 0.48 + 0.10*np.exp(-((xx-0.50)/0.18)**2)*np.sin((xx-0.28)*8.0)
    ax.plot(xx, yy, color=BLUE, lw=2.2, zorder=4)
    arrow(ax, (0.75, yy[-1]), (0.81, 0.49), color=BLUE, lw=2.0, ms=12)
    arrow(ax, (0.19, 0.47), (0.23, yy[0]), color=BLUE, lw=2.0, ms=12)

    ax.plot([0.20, 0.80], [0.18, 0.18], color=GRAY, lw=1.0)
    arrow(ax, (0.80, 0.18), (0.84, 0.18), color=GRAY, lw=1.0, ms=12)
    ax.text(0.20, 0.12, r"$t=0$", fontsize=11, style="italic")
    ax.text(0.84, 0.12, r"$t=1$", fontsize=11, style="italic", ha="right")

    xx2 = np.linspace(0.26, 0.74, 220)
    yy2 = 0.35 + 0.06*np.sin((xx2-0.25)*7.2 + 0.8)
    ax.plot(xx2, yy2, color="#8DB6E2", lw=1.6, ls="--", zorder=3)
    ax.text(0.55, 0.31, r"next-return flow $Z_t(s',a')$", fontsize=9, color=BLUE_DARK)

    label_box(ax, 0.53, 0.76,
              r"$\widehat{Z}_t(s,a)\ \leftarrow\ r + \gamma\, \widehat{Z}_t(s',a')$",
              w=0.46, h=0.08, fc="#F8FAFC", ec="#A7B4C2", fs=10)
    arrow(ax, (0.53, 0.72), (0.53, 0.60), color=GRAY, lw=0.9, ms=10)
    ax.text(0.53, 0.85, "distributional Bellman update at each flow step $t$",
            ha="center", fontsize=10)
    ax.text(0.53, 0.08, "pathwise supervision without full next-return rollout",
            ha="center", fontsize=10, style="italic", color=BLACK)

def draw_panel_C(ax):
    clean_axis(ax)
    panel_title(ax, "Panel C   Variance-aware local weighting")

    x0, y0 = 0.13, 0.18
    x1, y1 = 0.90, 0.78
    ax.plot([x0, x1], [y0, y0], color=GRAY, lw=1.0)
    ax.plot([x0, x0], [y0, y1], color=GRAY, lw=1.0)
    arrow(ax, (x1, y0), (0.94, y0), color=GRAY, lw=1.0, ms=12)
    arrow(ax, (x0, y1), (x0, 0.84), color=GRAY, lw=1.0, ms=12)
    ax.text(0.52, 0.08, "return std", ha="center", fontsize=10)
    ax.text(0.05, 0.51, "weight", ha="center", rotation=90, fontsize=10)

    formula = r"$w = 0.5 + \mathrm{sigmoid}\!\left(-\,\mathrm{temp}\cdot \mathrm{return\_std}\right)$"
    label_box(ax, 0.53, 0.90, formula, w=0.66, h=0.08, fc="#F8FAFC", ec="#A7B4C2", fs=9.5)

    xs = np.linspace(0, 1, 300)
    temp = 6.0
    raw = 0.5 + 1/(1 + np.exp(temp * xs))
    ys = (raw - raw.min()) / (raw.max() - raw.min())
    X = x0 + xs * (x1 - x0)
    Y = y0 + ys * (y1 - y0) * 0.95
    ax.plot(X, Y, color=BLUE_DARK, lw=2.0, zorder=3)

    def curve_y(x):
        raw0 = 0.5 + 1/(1 + np.exp(temp * 0))
        raw1 = 0.5 + 1/(1 + np.exp(temp * 1))
        rawx = 0.5 + 1/(1 + np.exp(temp * x))
        return (rawx - raw1) / (raw0 - raw1)

    low_x = 0.18
    high_x = 0.78
    low_y = curve_y(low_x)
    high_y = curve_y(high_x)
    low_pt = (x0 + low_x*(x1-x0), y0 + low_y*(y1-y0)*0.95)
    high_pt = (x0 + high_x*(x1-x0), y0 + high_y*(y1-y0)*0.95)

    ax.scatter([low_pt[0]], [low_pt[1]], s=42, color=GREEN, zorder=5)
    ax.scatter([high_pt[0]], [high_pt[1]], s=42, color=ORANGE, zorder=5)

    gaussian(ax, 0.18, 0.28, sx=0.03, sy=0.09, base=0.18, color=GREEN, alpha=0.35, lw=1.1)
    ax.text(0.18, 0.20, "low variance", ha="center", fontsize=9, color=GREEN)

    gaussian(ax, 0.22, 0.66, sx=0.07, sy=0.08, base=0.28, color=ORANGE, alpha=0.22, lw=1.1)
    ax.text(0.22, 0.58, "high variance", ha="center", fontsize=9, color=ORANGE)

    arrow(ax, (0.25, 0.29), (low_pt[0]-0.02, low_pt[1]-0.01), color=GREEN, lw=1.2, ms=11)
    arrow(ax, (0.31, 0.66), (high_pt[0]-0.02, high_pt[1]), color=ORANGE, lw=1.2, ms=11)

    ax.text(low_pt[0]+0.03, low_pt[1]+0.03, "large weight", color=GREEN, fontsize=9)
    ax.text(high_pt[0]+0.03, high_pt[1]-0.06, "small weight", color=ORANGE, fontsize=9)

def draw_panel_D(ax):
    clean_axis(ax)
    panel_title(ax, "Panel D   Time-dependent support clipping")

    ax.plot([0.10, 0.92], [0.10, 0.10], color=GRAY, lw=1.0)
    arrow(ax, (0.92, 0.10), (0.96, 0.10), color=GRAY, lw=1.0, ms=12)
    ax.text(0.12, 0.13, "0", fontsize=10)
    ax.text(0.92, 0.13, "1", fontsize=10, ha="right")

    x0, x1 = 0.20, 0.78
    yl0, yu0 = 0.40, 0.58
    yl1, yu1 = 0.22, 0.78

    ax.plot([x0, x0], [yl0, yu0], color=BLUE_DARK, lw=2.0)
    ax.plot([x1, x1], [yl1, yu1], color=BLUE_DARK, lw=2.0)
    ax.plot([x0, x1], [yu0, yu1], color=BLUE, lw=1.6)
    ax.plot([x0, x1], [yl0, yl1], color=BLUE, lw=1.6)

    tube = Polygon([[x0, yl0], [x0, yu0], [x1, yu1], [x1, yl1]],
                   closed=True, facecolor=BLUE_LIGHT, edgecolor="none", alpha=0.50, zorder=0)
    ax.add_patch(tube)

    ax.text(x0, 0.83, r"$t=0$", ha="center", fontsize=12, style="italic")
    ax.text(x1, 0.83, r"$t=1$", ha="center", fontsize=12, style="italic")
    ax.text(0.07, 0.49, "noise\nsupport", fontsize=10, va="center")
    ax.text(0.81, 0.50, "data\nsupport", fontsize=10, va="center")

    for x in np.linspace(x0+0.07, x1-0.07, 4):
        low = yl0 + (yl1-yl0)*(x-x0)/(x1-x0)
        up = yu0 + (yu1-yu0)*(x-x0)/(x1-x0)
        ax.plot([x, x], [low, up], color=BLUE, lw=0.8, ls=(0, (6, 4)), alpha=0.8)

    pts = np.array([
        [0.22, 0.51],
        [0.31, 0.48],
        [0.43, 0.33],
        [0.54, 0.28],
        [0.66, 0.46],
    ])
    ax.plot(pts[:, 0], pts[:, 1], color=BLUE_DARK, lw=2.0)
    for p in pts:
        ax.add_patch(Circle((p[0], p[1]), 0.012, facecolor=BLUE_DARK, edgecolor="white", lw=0.5, zorder=5))

    p1 = np.array([0.70, 0.51])
    p2 = np.array([0.73, 0.63])
    p3 = np.array([0.76, 0.56])
    ax.plot([pts[-1, 0], p1[0]], [pts[-1, 1], p1[1]], color=BLUE_DARK, lw=2.0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=RED, lw=1.8, ls="--")
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=RED, lw=1.2, ls="--")
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], color=ORANGE, lw=1.8)
    ax.plot([p3[0], 0.84], [p3[1], 0.54], color=BLUE_DARK, lw=2.0)
    for p in [tuple(p1), tuple(p3), (0.84, 0.54)]:
        ax.add_patch(Circle(p, 0.012, facecolor=BLUE_DARK, edgecolor="white", lw=0.5, zorder=5))

    ax.text(0.54, 0.15, "feasible support tube", ha="center", fontsize=9)
    ax.text(0.73, 0.66, "overshoot", color=RED, fontsize=9)
    ax.text(0.52, 0.04, "clip at each flow step $t$", ha="center", fontsize=10, style="italic")

fig = plt.figure(figsize=(15.6, 7.8), facecolor="white")
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.92], height_ratios=[1.0, 1.0], wspace=0.08, hspace=0.18)

axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[:, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 2])

draw_panel_A(axA)
draw_panel_B(axB)
draw_panel_C(axC)
draw_panel_D(axD)

fig.suptitle("From full-sample Bellman supervision to pathwise Bellman supervision",
             y=0.99, fontsize=17, color=BLACK, weight="medium")

fig.savefig("pathwise_bellman_method_figure_v3.png", bbox_inches="tight", facecolor="white")
fig.savefig("pathwise_bellman_method_figure_v3.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("pathwise_bellman_method_figure_v3.svg", bbox_inches="tight", facecolor="white")
plt.show()