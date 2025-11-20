import numpy as np

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

mpl.rcParams["savefig.transparent"] = True


c = 1.0 / np.sqrt(2.0)
e = 1e-2
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))

# all plots
for n in range(3):
    ax[n].set_aspect(1.0)
    ax[n].set_xlim(-1.1, 1.1)
    ax[n].set_ylim(-1.1, 1.1)
    ax[n].set_xticks(np.linspace(-1.0, 1.0, 5))
    ax[n].set_yticks(np.linspace(-1.0, 1.0, 5))
    ax[n].tick_params(axis="x", labelsize=14)
    ax[n].tick_params(axis="y", labelsize=14)
    ax[n].set_xlabel("axis 1", fontsize=16)
    ax[n].set_ylabel("axis 0", fontsize=16)


# scico
ax[0].set_title("scico", fontsize=18)
plist = [
    patches.FancyArrowPatch((-1.0, 0.0), (-0.5, 0.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((-c, -c), (-c / 2.0, -c / 2.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch(
        (
            0.0,
            -1.0,
        ),
        (0.0, -0.5),
        arrowstyle=style,
        color="r",
    ),
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=180, theta2=-45.0, color="b", lw=2, ls="dotted"),
    patches.FancyArrowPatch((c - e, -c - e), (c + e, -c + e), arrowstyle=style, color="b"),
]
for p in plist:
    ax[0].add_patch(p)

ax[0].text(-0.88, 0.02, r"$\theta=0$", color="r", fontsize=16)
ax[0].text(-3 * c / 4 - 0.01, -3 * c / 4 - 0.1, r"$\theta=\frac{\pi}{4}$", color="r", fontsize=16)
ax[0].text(0.03, -0.8, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=16)

ax[0].plot((1.0, 1.0), (-0.375, 0.375), color="orange", lw=2)
ax[0].arrow(
    0.94,
    0.375,
    0.0,
    -0.75,
    color="orange",
    lw=1.0,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax[0].text(0.7, 0.0, r"$\theta=0$", color="orange", ha="left", fontsize=16)
ax[0].plot((-0.375, 0.375), (1.0, 1.0), color="orange", lw=2)
ax[0].arrow(
    -0.375,
    0.94,
    0.75,
    0.0,
    color="orange",
    lw=1.0,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax[0].text(0.0, 0.82, r"$\theta=\frac{\pi}{2}$", color="orange", ha="center", fontsize=16)


# astra
ax[1].set_title("astra", fontsize=18)
plist = [
    patches.FancyArrowPatch((0.0, -1.0), (0.0, -0.5), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((c, -c), (c / 2.0, -c / 2.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((1.0, 0.0), (0.5, 0.0), arrowstyle=style, color="r"),
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=-90, theta2=45.0, color="b", lw=2, ls="dotted"),
    patches.FancyArrowPatch((c + e, c - e), (c - e, c + e), arrowstyle=style, color="b"),
]
for p in plist:
    ax[1].add_patch(p)

ax[1].text(0.02, -0.75, r"$\theta=0$", color="r", fontsize=16)
ax[1].text(3 * c / 4 + 0.01, -3 * c / 4 + 0.01, r"$\theta=\frac{\pi}{4}$", color="r", fontsize=16)
ax[1].text(0.65, 0.05, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=16)

ax[1].plot((-0.375, 0.375), (1.0, 1.0), color="orange", lw=2)
ax[1].arrow(
    -0.375,
    0.94,
    0.75,
    0.0,
    color="orange",
    lw=1.0,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax[1].text(0.0, 0.82, r"$\theta=0$", color="orange", ha="center", fontsize=16)
ax[1].plot((-1.0, -1.0), (-0.375, 0.375), color="orange", lw=2)
ax[1].arrow(
    -0.94,
    -0.375,
    0.0,
    0.75,
    color="orange",
    lw=1.0,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax[1].text(-0.9, 0.0, r"$\theta=\frac{\pi}{2}$", color="orange", ha="left", fontsize=16)


# svmbir
ax[2].set_title("svmbir", fontsize=18)
plist = [
    patches.FancyArrowPatch((-1.0, 0.0), (-0.5, 0.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((-c, c), (-c / 2.0, c / 2.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch(
        (
            0.0,
            1.0,
        ),
        (0.0, 0.5),
        arrowstyle=style,
        color="r",
    ),
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=45, theta2=180, color="b", lw=2, ls="dotted"),
    patches.FancyArrowPatch((c - e, c + e), (c + e, c - e), arrowstyle=style, color="b"),
]
for p in plist:
    ax[2].add_patch(p)
ax[2].text(-0.88, 0.02, r"$\theta=0$", color="r", fontsize=16)
ax[2].text(-3 * c / 4 + 0.01, 3 * c / 4 + 0.01, r"$\theta=\frac{\pi}{4}$", color="r", fontsize=16)
ax[2].text(0.03, 0.75, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=16)

ax[2].plot((1.0, 1.0), (-0.375, 0.375), color="orange", lw=2)
ax[2].arrow(
    0.94,
    0.375,
    0.0,
    -0.75,
    color="orange",
    lw=1.0,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax[2].text(0.7, 0.0, r"$\theta=0$", color="orange", ha="left", fontsize=16)

ax[2].plot((-0.375, 0.375), (-1.0, -1.0), color="orange", lw=2)
ax[2].arrow(
    0.375,
    -0.94,
    -0.75,
    0.0,
    color="orange",
    lw=1.0,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax[2].text(0.0, -0.82, r"$\theta=\frac{\pi}{2}$", color="orange", ha="center", fontsize=16)


fig.tight_layout()
fig.show()
