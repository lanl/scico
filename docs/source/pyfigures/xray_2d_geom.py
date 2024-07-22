import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt

c = 1.0 / np.sqrt(2.0)
e = 1e-2
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
for n in range(3):
    ax[n].set_aspect(1.0)
    ax[n].set_xlim(-1.1, 1.1)
    ax[n].set_ylim(-1.1, 1.1)
    ax[n].set_xticks(np.linspace(-1.0, 1.0, 5))
    ax[n].set_yticks(np.linspace(-1.0, 1.0, 5))
    ax[n].tick_params(axis="x", labelsize=12)
    ax[n].tick_params(axis="y", labelsize=12)
    ax[n].set_xlabel("axis 1", fontsize=14)
    ax[n].set_ylabel("axis 0", fontsize=14)

# scico
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
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=180, theta2=-45.0, color="b", ls="dotted"),
    patches.FancyArrowPatch((c - e, -c - e), (c + e, -c + e), arrowstyle=style, color="b"),
]
for p in plist:
    ax[0].add_patch(p)
ax[0].text(-0.88, 0.02, r"$\theta=0$", color="r", fontsize=14)
ax[0].text(-3 * c / 4 - 0.01, -3 * c / 4 - 0.1, r"$\theta=\frac{\pi}{4}$", color="r", fontsize=14)
ax[0].text(0.03, -0.8, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=14)
ax[0].set_title("scico", fontsize=14)

# astra
plist = [
    patches.FancyArrowPatch((0.0, -1.0), (0.0, -0.5), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((c, -c), (c / 2.0, -c / 2.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((1.0, 0.0), (0.5, 0.0), arrowstyle=style, color="r"),
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=-90, theta2=45.0, color="b", ls="dotted"),
    patches.FancyArrowPatch((c + e, c - e), (c - e, c + e), arrowstyle=style, color="b"),
]
for p in plist:
    ax[1].add_patch(p)
ax[1].text(0.02, -0.75, r"$\theta=0$", color="r", fontsize=14)
ax[1].text(3 * c / 4 + 0.01, -3 * c / 4 + 0.01, r"$\theta=\frac{\pi}{4}$", color="r", fontsize=14)
ax[1].text(0.65, 0.05, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=14)
ax[1].set_title("astra", fontsize=14)

# svmbir
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
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=45, theta2=180, color="b", ls="dotted"),
    patches.FancyArrowPatch((c - e, c + e), (c + e, c - e), arrowstyle=style, color="b"),
]
for p in plist:
    ax[2].add_patch(p)
ax[2].text(-0.88, 0.02, r"$\theta=0$", color="r", fontsize=14)
ax[2].text(-3 * c / 4 + 0.01, 3 * c / 4 + 0.01, r"$\theta=\frac{\pi}{4}$", color="r", fontsize=14)
ax[2].text(0.03, 0.75, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=14)
ax[2].set_title("svmbir", fontsize=14)

fig.tight_layout()
fig.show()
