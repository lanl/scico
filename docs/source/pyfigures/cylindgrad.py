import numpy as np

import scico.linop as scl
from scico import plot

input_shape = (7, 7, 7)
centre = (np.array(input_shape) - 1) / 2
end = np.array(input_shape) - centre
g0, g1, g2 = np.mgrid[-centre[0] : end[0], -centre[1] : end[1], -centre[2] : end[2]]

cg = scl.CylindricalGradient(input_shape=input_shape)

ang = cg.coord[0]
rad = cg.coord[1]
axi = cg.coord[2]

theta = np.arctan2(g0, g1)
clr = theta
# See https://stackoverflow.com/a/49888126
clr = (clr.ravel() - clr.min()) / np.ptp(clr)
clr = np.concatenate((clr, np.repeat(clr, 2)))
clr = plot.plt.cm.plasma(clr)

plot.plt.rcParams["savefig.transparent"] = True

fig = plot.plt.figure(figsize=(20, 6))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.quiver(g0, g1, g2, ang[0], ang[1], ang[2], colors=clr, length=0.9)
ax.set_title("Angular local coordinate axis", fontsize=18)
ax.set_xlabel("$x$", fontsize=15)
ax.set_ylabel("$y$", fontsize=15)
ax.set_zlabel("$z$", fontsize=15)
ax.tick_params(labelsize=15)
ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.quiver(g0, g1, g2, rad[0], rad[1], rad[2], colors=clr, length=0.9)
ax.set_title("Radial local coordinate axis", fontsize=18)
ax.set_xlabel("$x$", fontsize=15)
ax.set_ylabel("$y$", fontsize=15)
ax.set_zlabel("$z$", fontsize=15)
ax.tick_params(labelsize=15)
ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.quiver(g0, g1, g2, axi[0], axi[1], axi[2], colors=clr[0], length=0.9)
ax.set_title("Axial local coordinate axis", fontsize=18)
ax.set_xlabel("$x$", fontsize=15)
ax.set_ylabel("$y$", fontsize=15)
ax.set_zlabel("$z$", fontsize=15)
ax.tick_params(labelsize=15)
fig.tight_layout()
fig.show()
