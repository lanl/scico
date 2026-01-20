#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
SCICO Call Tracing
==================

This example demonstrates the call tracing functionality provided by the
[trace](../_autosummary/scico.trace.rst) module. It is based on the
[non-negative BPDN example](sparsecode_nn_admm.rst).
"""

import numpy as np

import jax

import scico.numpy as snp
from scico import functional, linop, loss, metric
from scico.optimize.admm import ADMM, MatrixSubproblemSolver
from scico.trace import register_variable, trace_scico_calls
from scico.util import device_info

"""
Initialize tracing. JIT must be disabled for correct tracing.

The call tracing mechanism prints the name, arguments, and return values
of functions/methods as they are called. Module and class names are
printed in light red, function and method names in dark red, arguments
and return values in light blue, and the names of registered variables
in light yellow. When a method defined in a class is called for an object
of a derived class type, the class of that object is printed in light
magenta, in square brackets. Function names and return values are
distinguished by initial ">>" and "<<" characters respectively.
"""
jax.config.update("jax_disable_jit", True)
trace_scico_calls()


"""
Create random dictionary, reference random sparse representation, and
test signal consisting of the synthesis of the reference sparse
representation.
"""
m = 32  # signal size
n = 128  # dictionary size
s = 10  # sparsity level

np.random.seed(1)
D = np.random.randn(m, n).astype(np.float32)
D = D / np.linalg.norm(D, axis=0, keepdims=True)  # normalize dictionary

xt = np.zeros(n, dtype=np.float32)  # true signal
idx = np.random.randint(low=0, high=n, size=s)  # support of xt
xt[idx] = np.random.rand(s)
y = D @ xt + 5e-2 * np.random.randn(m)  # synthetic signal

xt = snp.array(xt)  # convert to jax array
y = snp.array(y)  # convert to jax array


"""
Register a variable so that it can be referenced by name in the call trace.
Any hashable object and numpy arrays may be registered, but JAX arrays
cannot.
"""
register_variable(D, "D")


"""
Set up the forward operator and ADMM solver object.
"""
lmbda = 1e-1
A = linop.MatrixOperator(D)
register_variable(A, "A")
f = loss.SquaredL2Loss(y=y, A=A)
g_list = [lmbda * functional.L1Norm(), functional.NonNegativeIndicator()]
C_list = [linop.Identity((n)), linop.Identity((n))]
rho_list = [1.0, 1.0]
maxiter = 1  # number of ADMM iterations (set to small value to simplify trace output)

register_variable(f, "f")
register_variable(g_list[0], "g_list[0]")
register_variable(g_list[1], "g_list[1]")
register_variable(C_list[0], "C_list[0]")
register_variable(C_list[1], "C_list[1]")

solver = ADMM(
    f=f,
    g_list=g_list,
    C_list=C_list,
    rho_list=rho_list,
    x0=A.adj(y),
    maxiter=maxiter,
    subproblem_solver=MatrixSubproblemSolver(),
    itstat_options={"display": True, "period": 5},
)

register_variable(solver, "solver")


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x = solver.solve()
mse = metric.mse(xt, x)
