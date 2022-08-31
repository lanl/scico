import numpy as np

import imageio
from xdesign import Foam, discrete_phantom

from scico.linop.radon_svmbir import TomographicProjector


# make a simple foam and corresponding sinogram
def make_ct_test():
    np.random.seed(1234)  # initialize numpy seed
    N = 512  # phantom size
    x_fm = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
    x_gt = x_fm / np.max(x_fm)
    x_gt = np.clip(x_gt, 0, 1.0)

    # CT projection
    n_projection = 45  # number of projections
    angles = np.linspace(
        0,
        np.pi,
        n_projection,
        endpoint=False,
    )  # evenly spaced projection angles
    A = TomographicProjector(x_gt.shape, angles, N)  # Radon transform operator
    y = A @ x_gt  # sinogram
    return x_gt, y


x_gt_ct, y_ct = make_ct_test()

""" user-facing functions """

def load_y_ct():
    return y_ct


def load_gt_ct():
    return x_gt_ct
