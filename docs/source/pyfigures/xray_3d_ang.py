import numpy as np

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

mpl.rcParams["savefig.transparent"] = True


c = 1.0 / np.sqrt(2.0)
e = 1e-2
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.set_aspect(1.0)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xticks(np.linspace(-1.0, 1.0, 5))
ax.set_yticks(np.linspace(-1.0, 1.0, 5))
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel("$x$", fontsize=14)
ax.set_ylabel("$y$", fontsize=14)

plist = [
    patches.FancyArrowPatch((0.0, -1.0), (0.0, -0.5), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((c, -c), (c / 2.0, -c / 2.0), arrowstyle=style, color="r"),
    patches.FancyArrowPatch((1.0, 0.0), (0.5, 0.0), arrowstyle=style, color="r"),
    patches.Arc((0.0, 0.0), 2.0, 2.0, theta1=-90, theta2=45.0, color="b", lw=2, ls="dotted"),
    patches.FancyArrowPatch((c + e, c - e), (c - e, c + e), arrowstyle=style, color="b"),
]
for p in plist:
    ax.add_patch(p)
ax.text(0.02, -0.75, r"$\theta=0$", color="r", fontsize=14)
ax.text(
    3 * c / 4 + 0.01,
    -3 * c / 4 + 0.01,
    r"$\theta=\frac{\pi}{4}$",
    color="r",
    fontsize=14,
)
ax.text(0.65, 0.05, r"$\theta=\frac{\pi}{2}$", color="r", fontsize=14)

ax.plot((-0.375, 0.375), (1.0, 1.0), color="orange", lw=2)
ax.arrow(
    -0.375,
    0.94,
    0.75,
    0.0,
    color="orange",
    lw=0.5,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax.text(0.0, 0.82, r"$\theta=0$", color="orange", ha="center", fontsize=14)

ax.plot((-1.0, -1.0), (-0.375, 0.375), color="orange", lw=2)
ax.arrow(
    -0.94,
    -0.375,
    0.0,
    0.75,
    color="orange",
    lw=0.5,
    ls="--",
    head_width=0.03,
    length_includes_head=True,
)
ax.text(-0.9, 0.0, r"$\theta=\frac{\pi}{2}$", color="orange", ha="left", fontsize=14)

fig.tight_layout()
fig.show()
