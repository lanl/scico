import numpy as np

import scico.linop as scl
from scico import plot

input_shape = (21, 21)
centre = (np.array(input_shape) - 1) / 2
end = np.array(input_shape) - centre
g0, g1 = np.mgrid[-centre[0] : end[0], -centre[1] : end[1]]

pg = scl.PolarGradient(input_shape=input_shape)

ang = pg.coord[0]
rad = pg.coord[1]

clr = (np.arctan2(ang[1], ang[0]) + np.pi) / (2 * np.pi)

plot.plt.rcParams["image.cmap"] = "plasma"
plot.plt.rcParams["savefig.transparent"] = True

fig, ax = plot.plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
ax[0].quiver(g0, g1, ang[0], ang[1], clr)
ax[0].set_title("Angular local coordinate axis", fontsize=16)
ax[0].set_xlabel("$x$", fontsize=14)
ax[0].set_ylabel("$y$", fontsize=14)
ax[0].tick_params(labelsize=14)
ax[0].xaxis.set_ticks((-10, -5, 0, 5, 10))
ax[0].yaxis.set_ticks((-10, -5, 0, 5, 10))
ax[1].quiver(g0, g1, rad[0], rad[1], clr)
ax[1].set_title("Radial local coordinate axis", fontsize=16)
ax[1].set_xlabel("$x$", fontsize=14)
ax[1].set_ylabel("$y$", fontsize=14)
ax[1].tick_params(labelsize=14)
ax[1].xaxis.set_ticks((-10, -5, 0, 5, 10))
ax[1].yaxis.set_ticks((-10, -5, 0, 5, 10))
fig.tight_layout()
fig.show()
