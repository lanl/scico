import re
from inspect import getmembers, isfunction

# Similar processing for scico.scipy
import scico.scipy

ssp_func = getmembers(scico.scipy.special, isfunction)
for _, f in ssp_func:
    if f.__module__[0:11] == "scico.scipy" or f.__module__[0:14] == "jax._src.scipy":
        # Rewrite module name so that function is included in docs
        f.__module__ = "scico.scipy.special"
        # Attempt to fix incorrect cross-reference
        f.__doc__ = re.sub(
            r"^:func:`([\w_]+)` wrapped to operate",
            r":obj:`jax.scipy.special.\1` wrapped to operate",
            str(f.__doc__),
            flags=re.M,
        )
        modname = "scipy.special"
        f.__doc__ = re.sub(
            r"^LAX-backend implementation of :func:`([\w_]+)`.",
            r"LAX-backend implementation of :obj:`%s.\1`." % modname,
            str(f.__doc__),
            flags=re.M,
        )
        # Remove cross-reference to numpydoc style references section
        f.__doc__ = re.sub(r"(^|\ )\[(\d+)\]_", "", f.__doc__, flags=re.M)
        # Remove entire numpydoc references section
        f.__doc__ = re.sub(r"References\n----------\n.*\n", "", f.__doc__, flags=re.DOTALL)
        # Remove problematic citation
        f.__doc__ = re.sub(r"See \[dlmf\]_ for details.", "", f.__doc__, re.M)
        f.__doc__ = re.sub(r"\[dlmf\]_", "NIST DLMF", f.__doc__, re.M)

# Fix indentation problems
if hasattr(scico.scipy.special, "sph_harm"):
    scico.scipy.special.sph_harm.__doc__ = re.sub(
        "^Computes the", "  Computes the", scico.scipy.special.sph_harm.__doc__, flags=re.M
    )
