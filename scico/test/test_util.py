import socket
import urllib.error as urlerror

import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.util import ContextTimer, Timer, check_for_tracer, url_get


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
