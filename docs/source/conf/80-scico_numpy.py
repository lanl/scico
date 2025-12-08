import re
from inspect import getmembers, isfunction

# Rewrite module names for certain functions imported into scico.numpy so that they are
# included in the docs for that module. While a bit messy to do so here rather than in a
# function run via app.connect, it is necessary (for some yet to be identified reason)
# to do it here to ensure that the relevant API docs include a table of functions.
import scico.numpy

for module in (scico.numpy, scico.numpy.fft, scico.numpy.linalg, scico.numpy.testing):
    for _, f in getmembers(module, isfunction):
        # Rewrite module name so that function is included in docs
        f.__module__ = module.__name__
        f.__doc__ = re.sub(
            r"^:func:`([\w_]+)` wrapped to operate",
            r":obj:`jax.numpy.\1` wrapped to operate",
            str(f.__doc__),
            flags=re.M,
        )
        modname = ".".join(module.__name__.split(".")[1:])
        f.__doc__ = re.sub(
            r"^LAX-backend implementation of :func:`([\w_]+)`.",
            r"LAX-backend implementation of :obj:`%s.\1`." % modname,
            str(f.__doc__),
            flags=re.M,
        )
        # Improve formatting of jax.numpy warning
        f.__doc__ = re.sub(
            r"^\*\*\* This function is not yet implemented by jax.numpy, and will "
            r"raise NotImplementedError \*\*\*",
            "**WARNING**: This function is not yet implemented by jax.numpy, "
            " and will raise :exc:`NotImplementedError`.",
            f.__doc__,
            flags=re.M,
        )
        # Remove cross-references to section NEP35
        f.__doc__ = re.sub(":ref:`NEP 35 <NEP35>`", "NEP 35", f.__doc__, re.M)
        # Remove cross-reference to numpydoc style references section
        f.__doc__ = re.sub(r" \[(\d+)\]_", "", f.__doc__, flags=re.M)
        # Remove entire numpydoc references section
        f.__doc__ = re.sub(r"References\n----------\n.*\n", "", f.__doc__, flags=re.DOTALL)


# Fix various docstring formatting errors
scico.numpy.testing.break_cycles.__doc__ = re.sub(
    "calling gc.collect$",
    "calling gc.collect.\n\n",
    scico.numpy.testing.break_cycles.__doc__,
    flags=re.M,
)
scico.numpy.testing.break_cycles.__doc__ = re.sub(
    r" __del__\) inside", r"__del__\) inside", scico.numpy.testing.break_cycles.__doc__, flags=re.M
)
scico.numpy.testing.assert_raises_regex.__doc__ = re.sub(
    r"\*args,\n.*\*\*kwargs",
    "*args, **kwargs",
    scico.numpy.testing.assert_raises_regex.__doc__,
    flags=re.M,
)
scico.numpy.BlockArray.global_shards.__doc__ = re.sub(
    r"`Shard`s", r"`Shard`\ s", scico.numpy.BlockArray.global_shards.__doc__, flags=re.M
)
