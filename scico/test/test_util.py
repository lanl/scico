import socket
import urllib.error as urlerror
import warnings

import numpy as np

import jax
from jax.interpreters.xla import DeviceArray

import pytest

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.util import (
    ContextTimer,
    Timer,
    check_for_tracer,
    ensure_on_device,
    indexed_shape,
    is_nested,
    parse_axes,
    slice_length,
    url_get,
)


def test_ensure_on_device():
    # Used to restore the warnings after the context is used
    with warnings.catch_warnings():
        # Ignores warning raised by ensure_on_device
        warnings.filterwarnings(action="ignore", category=UserWarning)

        NP = np.ones(2)
        SNP = snp.ones(2)
        BA = BlockArray.array([NP, SNP])
        NP_, SNP_, BA_ = ensure_on_device(NP, SNP, BA)

        assert isinstance(NP_, DeviceArray)

        assert isinstance(SNP_, DeviceArray)
        assert SNP.unsafe_buffer_pointer() == SNP_.unsafe_buffer_pointer()

        assert isinstance(BA_, BlockArray)
        assert BA._data.unsafe_buffer_pointer() == BA_._data.unsafe_buffer_pointer()

        np.testing.assert_raises(TypeError, ensure_on_device, [1, 1, 1])

        NP_ = ensure_on_device(NP)
        assert isinstance(NP_, DeviceArray)


# See https://stackoverflow.com/a/33117579
def _internet_connected(host="8.8.8.8", port=53, timeout=3):
    """Check if internet connection available.

    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        return False


@pytest.mark.skipif(not _internet_connected(), reason="No internet connection")
def test_url_get():
    url = "https://github.com/lanl/scico/blob/main/README.rst"
    assert not url_get(url).getvalue().find(b"SCICO") == -1

    url = "about:blank"
    np.testing.assert_raises(urlerror.URLError, url_get, url)

    url = "https://github.com/lanl/scico/blob/main/README.rst"
    np.testing.assert_raises(ValueError, url_get, url, -1)


def test_parse_axes():
    axes = None
    np.testing.assert_raises(ValueError, parse_axes, axes)

    axes = None
    assert parse_axes(axes, np.shape([[1, 1], [1, 1]])) == [0, 1]

    axes = None
    assert parse_axes(axes, np.shape([[1, 1], [1, 1]]), default=[0]) == [0]

    axes = [1, 2]
    assert parse_axes(axes) == axes

    axes = 1
    assert parse_axes(axes) == (1,)

    axes = "axes"
    np.testing.assert_raises(ValueError, parse_axes, axes)

    axes = 2
    np.testing.assert_raises(ValueError, parse_axes, axes, np.shape([1]))

    axes = (1, 2, 2)
    np.testing.assert_raises(ValueError, parse_axes, axes)


@pytest.mark.parametrize("length", (4, 5, 8, 16, 17))
@pytest.mark.parametrize("start", (None, 0, 1, 2, 3))
@pytest.mark.parametrize("stop", (None, 0, 1, 2, -2, -1))
@pytest.mark.parametrize("stride", (None, 1, 2, 3))
def test_slice_length(length, start, stop, stride):
    x = np.zeros(length)
    slc = slice(start, stop, stride)
    assert x[slc].size == slice_length(length, slc)


@pytest.mark.parametrize("length", (4, 5))
@pytest.mark.parametrize("slc", (0, 1, -4, Ellipsis))
def test_slice_length_other(length, slc):
    x = np.zeros(length)
    assert x[slc].size == slice_length(length, slc)


@pytest.mark.parametrize("shape", ((8, 8, 1), (7, 1, 6, 5)))
@pytest.mark.parametrize(
    "slc",
    (
        np.s_[0:5],
        np.s_[:, 0:4],
        np.s_[2:, :, :-2],
        np.s_[..., 2:],
        np.s_[..., 2:, :],
        np.s_[1:, ..., 2:],
    ),
)
def test_indexed_shape(shape, slc):
    x = np.zeros(shape)
    assert x[slc].shape == indexed_shape(shape, slc)


def test_check_for_tracer():
    # Using examples from Jax documentation

    A = snp.ones((5, 5))
    x = snp.ones((10, 5))

    @check_for_tracer
    def norm(X):
        X = X - X.mean(0)
        return X / X.std(0)

    with pytest.raises(TypeError):
        check_norm = jax.jit(norm)
        check_norm(x)

    vv = check_for_tracer(lambda x: A @ x)
    with pytest.raises(TypeError):
        mv = jax.vmap(vv)
        mv(x)


def test_is_nested():
    # list
    assert is_nested([1, 2, 3]) == False

    # tuple
    assert is_nested((1, 2, 3)) == False

    # list of lists
    assert is_nested([[1, 2], [4, 5], [3]]) == True

    # list of lists + scalar
    assert is_nested([[1, 2], 3]) == True

    # list of tuple + scalar
    assert is_nested([(1, 2), 3]) == True

    # tuple of tuple + scalar
    assert is_nested(((1, 2), 3)) == True

    # tuple of lists + scalar
    assert is_nested(([1, 2], 3)) == True


def test_timer_basic():
    t = Timer()
    t.start()
    t0 = t.elapsed()
    t.stop()
    t1 = t.elapsed()
    assert t0 >= 0.0
    assert t1 >= t0
    assert len(t.__str__()) > 0
    assert len(t.labels()) > 0


def test_timer_multi():
    t = Timer("a")
    t.start(["a", "b"])
    t0 = t.elapsed("a")
    t.stop("a")
    t.stop("b")
    t.stop(["a", "b"])
    assert t.elapsed("a") >= 0.0
    assert t.elapsed("b") >= 0.0
    assert t.elapsed("a", total=False) == 0.0


def test_timer_reset():
    t = Timer("a")
    t.start(["a", "b"])
    t.reset("a")
    assert t.elapsed("a") == 0.0
    t.reset("all")
    assert t.elapsed("b") == 0.0


def test_ctxttimer_basic():
    t = Timer()
    with ContextTimer(t):
        t0 = t.elapsed()
    assert t.elapsed() >= 0.0


def test_ctxttimer_stopstart():
    t = Timer()
    t.start()
    with ContextTimer(t, action="StopStart"):
        t0 = t.elapsed()
    t.stop()
    assert t.elapsed() >= 0.0
