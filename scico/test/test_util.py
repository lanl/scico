import socket
import urllib.error as urlerror

import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.util import (
    ContextTimer,
    Timer,
    check_for_tracer,
    partial,
    rgetattr,
    rsetattr,
    url_get,
)


def test_rattr():
    class A:
        class B:
            c = 0

        b = B()

    a = A()
    rsetattr(a, "b.c", 1)
    assert rgetattr(a, "b.c") == 1

    assert rgetattr(a, "c.d", 10) == 10

    with pytest.raises(AttributeError):
        assert rgetattr(a, "c.d")


def test_partial_pos():
    def func(a, b, c, d):
        return a + 2 * b + 4 * c + 8 * d

    pfunc = partial(func, (0, 2), 0, 0)
    assert pfunc(1, 0) == 2 and pfunc(0, 1) == 8


def test_partial_kw():
    def func(a=1, b=1, c=1, d=1):
        return a + 2 * b + 4 * c + 8 * d

    pfunc = partial(func, (), a=0, c=0)
    assert pfunc(b=1, d=0) == 2 and pfunc(b=0, d=1) == 8


def test_partial_pos_and_kw():
    def func(a, b, c=1, d=1):
        return a + 2 * b + 4 * c + 8 * d

    pfunc = partial(func, (0,), 0, c=0)
    assert pfunc(1, d=0) == 2 and pfunc(0, d=1) == 8


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
