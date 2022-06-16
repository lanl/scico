import numpy as np

import imageio
from xdesign import Foam, discrete_phantom

from scico.linop.radon_svmbir import TomographicProjector

Tx = 50  # how far to shift
N = 350  # image size

# prep the ground truth, x
def make_x():
    x = imageio.imread("cells.jpg")  # image is CC0, can use without attribution
    x = x.sum(axis=-1)[90 : 90 + N, : N + Tx]
    x = x / x.max()
    return x.astype(np.float32)


x = make_x()

# prep the illumination, g
def make_w():
    I, J = np.meshgrid(range(N), range(N), indexing="ij")
    c = (N / 3, N / 4)
    w = np.exp(-((I - c[0]) ** 2 + (J - c[1]) ** 2) / (N * 10))
    w = w + (1 - (0.1 * I + 0.7 * J) / N)
    w = w / w.max()
    return w.astype(np.float32)


w = make_w()


def make_ys():
    y1 = x[:, :N] + w
    y2 = x[:, Tx : Tx + N] + w
    return y1, y2


y1, y2 = make_ys()


def make_x_test():
    x = np.zeros((N, N + Tx))
    x[N // 4 : 3 * N // 4, N // 6 : 5 * N // 6] = 1.0
    return x


x_test = make_x_test()

# make a simple illumination pattern
def make_w_test():
    w, _ = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    return w


w_test = make_w_test()


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


def load_y1():
    return y1


def load_ys():
    return y1, y2


def find_offset(y1, y2):
    return Tx


def load_test_solution():
    return x_test, w_test


def load_y_ct():
    return y_ct


def load_gt_ct():
    return x_gt_ct
