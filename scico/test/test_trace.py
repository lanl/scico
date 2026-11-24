import inspect
import re
import sys
import types

import numpy as np

import jax.numpy as jnp

import pytest

import scico.linop
import scico.numpy as snp
import scico.trace
from scico.trace import (
    _get_hash,
    _in_module,
    _is_scico_module,
    _is_scico_object,
    _is_submodule,
    _submodule_name,
    _trace_arg_repr,
    apply_decorator,
    call_trace,
    register_variable,
)

# Trace output is wrapped in colorama escape sequences when colorama is
# installed, and is not when it is absent. Strip them so that the assertions
# below hold either way.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

TOY = "trace_toy"


def plain(text):
    """Remove ANSI colour escape sequences from `text`."""
    return _ANSI.sub("", text)


def traced(capsys):
    """Return the stripped lines of trace output captured so far."""
    # call_trace writes to stderr, not stdout.
    return [plain(ln) for ln in capsys.readouterr().err.splitlines() if ln.strip()]


@pytest.fixture(autouse=True)
def restore_trace_state():
    """Save and restore global state mutated by the tracing machinery.

    The trace level, the registered-variable table, and the display flags
    are attributes of the `call_trace` function itself, and there is no
    mechanism for resetting them. Without this, a test that failed part
    way through a traced call would leave `trace_level` non-zero and break
    the indentation expected by every later test. The synthetic modules
    are registered in :data:`sys.modules` and are removed here.
    """
    names = (
        "trace_level",
        "instance_hash",
        "show_return_value",
        "show_jax_device",
        "show_jax_sharding",
    )
    saved = {name: getattr(call_trace, name) for name in names}
    call_trace.instance_hash = {}
    yield
    for name, value in saved.items():
        setattr(call_trace, name, value)
    for name in [n for n in sys.modules if n == TOY or n.startswith(TOY)]:
        del sys.modules[name]


TOY_SOURCE = """
class Gadget:
    def method(self, v):
        return v + 1

    def quiet(self):
        return None

    @staticmethod
    def stat(a, b):
        return a * b

    @classmethod
    def make(cls, v):
        return (cls.__name__, v)

    @property
    def prop(self):
        return 7


class Widget(Gadget):
    pass


def helper(x):
    return x - 1


def outer(x):
    return helper(x) * 10
"""


def toy_module(name=TOY):
    """Construct a throwaway module for exercising `apply_decorator`.

    The public entry point, `trace_scico_calls`, wraps methods in place
    across eight scico submodules and provides no way to undo that, so
    calling it here would leave the package traced for the remainder of
    the pytest session. A synthetic module drives the same traversal
    without touching scico.

    The module is registered in :data:`sys.modules` because
    :func:`inspect._findclass`, which the wrapper uses to recognise a
    method call and to name the defining class, resolves the class by
    looking its module up there.
    """
    mod = types.ModuleType(name)
    exec(compile(TOY_SOURCE, f"<{name}>", "exec"), mod.__dict__)
    # apply_decorator only considers objects whose __module__ matches the
    # name of the module, which exec does not arrange by itself.
    for obj in (mod.Gadget, mod.Widget, mod.helper, mod.outer):
        obj.__module__ = name
    for value in vars(mod.Gadget).values():
        func = getattr(value, "__func__", getattr(value, "fget", value))
        if isinstance(func, types.FunctionType):
            func.__module__ = name
    sys.modules[name] = mod
    return mod


@pytest.fixture
def toy():
    """A synthetic module with `call_trace` applied as scico applies it."""
    mod = toy_module()
    apply_decorator(mod, call_trace, recursive=False)
    return mod


class TestGetHash:
    def test_hashable(self):
        assert _get_hash(7) == hash(7)
        assert _get_hash("abc") == hash("abc")

    def test_ndarray_uses_memory_address(self):
        # The reference must be kept alive: the address is the hash, and a
        # freed buffer's address can be reused by a later allocation.
        x = np.ones(3)
        assert _get_hash(x) == x.ctypes.data

    def test_unhashable(self):
        # A jax array is not hashable, which is why register_variable
        # rejects one.
        assert _get_hash(jnp.ones(3)) is None
        # A list sets __hash__ to None rather than omitting it.
        assert _get_hash([1, 2]) is None


class TestRegisterVariable:
    def test_ndarray(self):
        x = np.ones((2, 2))
        register_variable(x, "xray")
        assert plain(_trace_arg_repr(x)) == "xray"

    def test_object(self):
        obj = Exception("not an array, but hashable")
        register_variable(obj, "thing")
        assert plain(_trace_arg_repr(obj)) == "thing"

    def test_jax_array_rejected(self):
        with pytest.raises(ValueError):
            register_variable(jnp.ones(3), "cannot")


class TestTraceArgRepr:
    @pytest.mark.parametrize(
        "val,expected",
        (
            (None, "None"),
            (3, "3"),
            (2.5, "2.5"),
            ("s", "s"),
            ((1, 2), "(1, 2)"),
            (np.dtype(np.float32), "numpy.float32"),
            (np.array(5.0), "5.0"),
        ),
    )
    def test_simple(self, val, expected):
        assert plain(_trace_arg_repr(val)) == expected

    def test_class(self):
        assert _trace_arg_repr(int) == "builtins.int"

    @pytest.mark.parametrize("make", (np.zeros, snp.zeros))
    def test_array_shape(self, make):
        assert plain(_trace_arg_repr(make((2, 3)))) == "Array(2, 3)"

    def test_long_tuple_is_not_expanded(self):
        # Only tuples of fewer than six scalars are shown element by element.
        assert plain(_trace_arg_repr(tuple(range(6)))) == "[tuple]"

    def test_unregistered_object(self):
        class Opaque:
            pass

        assert plain(_trace_arg_repr(Opaque())) == "[Opaque]"


class TestCallTrace:
    def test_function(self, toy, capsys):
        assert toy.helper(5) == 4
        assert traced(capsys) == [f">> {TOY}.helper(5)", "<< 4"]

    def test_no_return_line_when_none(self, toy, capsys):
        assert toy.Gadget().quiet() is None
        # A None return value is not reported.
        assert traced(capsys) == [f">> {TOY}.Gadget.quiet()"]

    def test_show_return_value_flag(self, toy, capsys):
        call_trace.show_return_value = False
        toy.helper(5)
        assert traced(capsys) == [f">> {TOY}.helper(5)"]

    def test_nesting_is_indented(self, toy, capsys):
        assert toy.outer(5) == 40
        assert traced(capsys) == [
            f">> {TOY}.outer(5)",
            f">>   {TOY}.helper(5)",
            "<<   4",
            "<< 40",
        ]

    def test_trace_level_restored_after_call(self, toy):
        before = call_trace.trace_level
        toy.helper(5)
        assert call_trace.trace_level == before

    def test_wrapping_is_idempotent(self):
        once = call_trace(toy_module().helper)
        assert call_trace(once) is once

    def test_metadata_preserved(self):
        mod = toy_module()
        original = mod.helper
        assert call_trace(original).__name__ == original.__name__

    def test_method_elides_self(self, toy, capsys):
        assert toy.Gadget().method(1) == 2
        assert traced(capsys)[0] == f">> {TOY}.Gadget.method(1)"

    def test_method_shows_defining_and_self_class(self, toy, capsys):
        assert toy.Widget().method(1) == 2
        # method is defined by Gadget but called on a Widget.
        assert traced(capsys)[0] == f">> {TOY}.Gadget[Widget].method(1)"

    def test_registered_self_is_named(self, toy, capsys):
        g = toy.Gadget()
        register_variable(g, "gee")
        assert g.method(1) == 2
        assert traced(capsys)[0] == ">> gee.method(1)"

    def test_keyword_arguments(self, toy, capsys):
        assert toy.Gadget().method(v=1) == 2
        assert traced(capsys)[0] == f">> {TOY}.Gadget.method(v=1)"


class TestStaticMethodDispatch:
    """Cover the heuristic in `_call_wrapped_function`.

    Wrapping replaces a `staticmethod` with a plain function on the class,
    so a call through an instance passes the instance as the first
    argument while a call through the class does not. The wrapper tells
    them apart by comparing the class that defines the method with the
    type of the first argument, stripping the argument only when they are
    the same class.
    """

    def test_via_class(self, toy, capsys):
        assert toy.Gadget.stat(3, 4) == 12
        assert traced(capsys)[0] == f">> {TOY}.Gadget.stat(3, 4)"

    def test_via_instance(self, toy, capsys):
        assert toy.Gadget().stat(3, 4) == 12
        assert traced(capsys)[0] == f">> {TOY}.Gadget.stat(3, 4)"

    def test_via_subclass_instance(self, toy):
        # The comparison is against the defining class, so an instance of a
        # subclass is not recognised and is passed through to the method as
        # a surplus argument. This is the failure that the comment beside
        # the heuristic anticipates, recorded here rather than endorsed.
        with pytest.raises(TypeError):
            toy.Widget().stat(3, 4)


class TestApplyDecorator:
    def test_wraps_functions_and_methods(self):
        seen = apply_decorator(toy_module(), call_trace, recursive=False)
        assert seen[f"{TOY}.helper"] == 1
        assert seen[f"{TOY}.Gadget.method"] == 1
        assert seen[f"{TOY}.Gadget.stat"] == 1

    def test_leaves_classmethods_and_properties_alone(self, toy):
        # getmembers reports a classmethod as a bound method and a property
        # as a property, neither of which matches the FunctionType filter
        # that apply_decorator uses, so neither is wrapped.
        for name in ("make", "prop"):
            obj = inspect.getattr_static(toy.Gadget, name)
            inner = getattr(obj, "__func__", getattr(obj, "fget", obj))
            assert not hasattr(inner, "_call_trace_wrap")
        assert toy.Gadget.make(2) == ("Gadget", 2)
        assert toy.Gadget().prop == 7

    def test_skip(self):
        mod = toy_module()
        seen = apply_decorator(mod, call_trace, recursive=False, skip=["helper"])
        assert f"{TOY}.helper" not in seen
        assert not hasattr(mod.helper, "_call_trace_wrap")

    def test_each_function_wrapped_once(self):
        seen = apply_decorator(toy_module(), call_trace, recursive=False)
        # Widget inherits method and stat from Gadget, and they are recorded
        # under the class that defines them, so they are not wrapped twice.
        assert set(seen.values()) == {1}

    def test_seen_prevents_rewrapping(self):
        mod = toy_module()
        seen = apply_decorator(mod, call_trace, recursive=False)
        first = mod.helper
        apply_decorator(mod, call_trace, recursive=False, seen=seen)
        assert mod.helper is first

    def test_recursive(self):
        parent = toy_module()
        child = toy_module(f"{TOY}.child")
        parent.child = child
        seen = apply_decorator(parent, call_trace, recursive=True)
        assert seen[f"{TOY}.helper"] == 1
        assert seen[f"{TOY}.child.helper"] == 1

    def test_not_recursive(self):
        parent = toy_module()
        child = toy_module(f"{TOY}.child")
        parent.child = child
        seen = apply_decorator(parent, call_trace, recursive=False)
        assert f"{TOY}.child.helper" not in seen


class TestModulePredicates:
    def test_is_scico_object(self):
        assert _is_scico_object(scico.linop.Identity)
        assert _is_scico_object(scico.trace.call_trace)
        # snp.zeros is re-exported from jax and is not defined in scico.
        assert not _is_scico_object(snp.zeros)
        assert not _is_scico_object(np.zeros)

    def test_is_scico_module(self):
        assert _is_scico_module(snp)
        assert not _is_scico_module(np)
        assert not _is_scico_module(toy_module())

    def test_in_module(self):
        mod = toy_module()
        assert _in_module(mod, mod.helper)
        assert not _in_module(mod, np.zeros)

    def test_is_submodule(self):
        assert _is_submodule(toy_module(), toy_module(f"{TOY}.child"))
        assert not _is_submodule(toy_module(f"{TOY}.child"), toy_module())

    def test_submodule_name(self):
        assert _submodule_name(toy_module(), toy_module(f"{TOY}.child")) == "child"
        assert _submodule_name(toy_module(f"{TOY}.child"), toy_module()) == ""
