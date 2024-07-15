import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

mpl.rcParams["savefig.transparent"] = True


# See https://github.com/matplotlib/matplotlib/issues/21688
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


# Define vector components
𝜃 = 10 * np.pi / 180.0  # angle in x-y plane (azimuth angle)
𝛼 = 70 * np.pi / 180.0  # angle with z axis (zenith angle)
𝛥p, 𝛥d = 0.3, 1.0
d = (-𝛥d * np.sin(𝛼) * np.sin(𝜃), 𝛥d * np.sin(𝛼) * np.cos(𝜃), 𝛥d * np.cos(𝛼))
u = (𝛥p * np.cos(𝜃), 𝛥p * np.sin(𝜃), 0.0)
v = (𝛥p * np.cos(𝛼) * np.sin(𝜃), -𝛥p * np.cos(𝛼) * np.cos(𝜃), 𝛥p * np.sin(𝛼))

# Location of text labels
d_txtpos = np.array(d) + np.array([0, 0, -0.12])
u_txtpos = np.array(d) + np.array(u) + np.array([0, 0, -0.1])
v_txtpos = np.array(d) + np.array(v) + np.array([0, 0, 0.03])


arrowstyle = "-|>,head_width=2.5,head_length=9"

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Set view
ax.set_aspect("equal")
ax.elev = 40
ax.azim = -60
ax.set_box_aspect(None, zoom=1.8)
ax.set_xlim((-10.5, 10.5))
ax.set_ylim((-10.5, 10.5))
ax.set_zlim((-10.5, 10.5))

# Disable shaded 3d axis grids
ax.set_axis_off()

# Draw central x,y,z axes and labels
axis_crds = np.array([[-10, 10], [0, 0], [0, 0]])
axis_lbls = ("$x$", "$y$", "$z$")
for k in range(3):
    crd = np.roll(axis_crds, k, axis=0)
    ax.add_artist(
        Arrow3D(
            *crd.tolist(),
            lw=1.5,
            ls="--",
            arrowstyle=arrowstyle,
            color="black",
        )
    )
    ax.text(*(1.05 * crd[:, 1]).tolist(), axis_lbls[k], fontsize=12)

wx = 4
wy = 3
wz = 2
bx = np.array([-wx, wx, wx, wx, -wx, -wx, -wx])
by = np.array([-wy, -wy, wy, wy, wy, -wy, -wy])
bz = np.array([-wz, -wz, -wz, wz, wz, wz, -wz])
ax.plot(bx, by, bz, lw=2, color="blue")
ax.plot(bx[0:3], by[0:3], -bz[0:3], lw=2, color="blue")
bx = np.array([wx, wx])
by = np.array([-wy, -wy])
bz = np.array([-wz, wz])
ax.plot(bx, by, bz, lw=2, color="blue")
bx = np.array([-wx, -wx, wx])
by = np.array([-wy, wy, wy])
bz = np.array([-wz, -wz, -wz])
ax.plot(bx, by, bz, lw=2, ls="--", color="blue")
bx = np.array([-wx, -wx])
by = np.array([wy, wy])
bz = np.array([-wz, wz])
ax.plot(bx, by, bz, lw=2, ls="--", color="blue")

fig.tight_layout()
fig.show()
