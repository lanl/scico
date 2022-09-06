"""
# Example: Nonuniform Illumination
This example demonstrates defining a linear operator and solving a least squares problem in SCICO.
"""

"""
## Setup
To set up your environment, run the cell below.

If you get a popup with 'Warning: This notebook was not authored by Google.', select 'Run anyway'.
You should see console outputs appearing.
The install may take several minutes;
when it is finished, you should see `==done with install==`.

"""
!pip install git+https://github.com/lanl/scico@cristina-mike/tutorial
!git clone -b cristina-mike/tutorial https://github.com/lanl/scico-data.git
print('==done with install==')

"""
## Introduction
You set up a new microscope in your lab and take a brightfield image, which we'll call $y_1$.

Run the next cell to see $y_1$.
"""
%cd /content/scico-data/notebooks/tutorial

import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "gray"  # set default colormap

from tutorial_funcs import load_y1

y1 = load_y1()

print(f"The shape of y1 is {y1.shape}")

fig, ax = plt.subplots()
ax.imshow(y1)
ax.set_title("$y_1$")
fig.show()

"""
The image looks good, except for a distracting bright spot in the upper-left corner.
Based on your knowledge of this microscope,
you suspect that this spot comes from an additive nonuniform illumation,
$$y_1 = x_1 + w,$$
where $x_1$ is the unknown image and $w$ is the unknown illumination.

You want to estimate $x_1$ and $w$ from $y_1$, but the problem as is is hopelessly underdetermined:
$350x350$ measurements and $2x350x350$ unknowns.
You have the idea to move the slide to the left and take another image, $y_2$.

Run the next cell to see $y_1$ and $y_2$.
"""

from tutorial_funcs import load_ys

y1, y2 = load_ys()

fig, ax = plt.subplots()
ax.imshow(y1)
ax.set_title("$y_1$")

fig, ax = plt.subplots()
ax.imshow(y2)
ax.set_title("$y_2$")
fig.show()

"""
In reality, we would need to write code to find the offset between $y_1$ and $y_2$.
Here, we'll assume that step is already done.
Run the next cell to find the offset.
"""

from tutorial_funcs import find_offset

offset = find_offset(y1, y2)

print(f"y2 is y1 shifted to the left by {offset} pixels")

"""
You are done with part 1. Please report back in the Webex chat: **done with part 1**.

While you wait for others to finish, you could think about how you would recover $x_1$ and $w$ with the tools you know.

🛑 **PAUSE HERE** 🛑
"""

"""
## Defining a forward model (NumPy version)


We now have the forward model
$$y_1 = x_1 + w$$
$$y_2 = x_2 + w.$$
This is not immediately useful: $2x350x350$ meausurements and $3x350x350$ unknowns.
However, we know that $x_1$ and $x_2$ are parts of a larger image, which we'll call $x$.
In terms of $x$, we have
$$y_1 = C_1x + w,$$
$$y_2 = C_2x + w,$$
where $x$ is $350x400$ and
where $C_1$ and $C_2$ represent two different (known) cropping operations.
"""

"""
**In the cell below, implement this forward model in NumPy.** You can test your forward model by running the cell below that.
"""


# startq
def forward(x, w):
    y1 = ...  # your code here
    y2 = ...

    return y1, y2


# starta
def forward(x, w):
    N_cols = w.shape[1]
    y1 = x[:, :N_cols] + w
    y2 = x[:, offset : N_cols + offset] + w
    return y1, y2


# endqa

""""""

from tutorial_funcs import load_test_solution

x_test, w_test = load_test_solution()

# run the forward model
y1_test, y2_test = forward(x_test, w_test)

(
    fig,
    ax,
) = (
    plt.subplots()
)  # NOTE: not using a single plot so that it is more clear x_test is wider than others
ax.imshow(x_test)
ax.set_title("x_test")
fig.show()

fig, ax = plt.subplots()
ax.imshow(w_test)
ax.set_title("w_test")
fig.show()

fig, ax = plt.subplots()
ax.imshow(y1_test)
ax.set_title("y1_test")

fig, ax = plt.subplots()
ax.imshow(y2_test)
ax.set_title("y2_test")
fig.show()


"""
You are done with part 2. Please report back in the Webex chat: **done with part 2**.

While you wait for others to finish, you could think about measurements systems in your research in terms of forward models.

🛑 **PAUSE HERE** 🛑
"""


"""
## Defining a fowarding model (SCICO version)
SCICO is designed to solve problems of the form $y = Aq$,
where $A$ is the forward model,
$q$ contains the image(s) you want to recover,
and $y$ contains the measurements.
In this section, we'll convert our problem into this form.

A compact way to write our system of equations is
$$ \underbrace{\begin{bmatrix}y_1 \ y_2 \end{bmatrix}}_{y}
=
\underbrace{\begin{bmatrix}C_1 & I   \\ C_2 & I  \end{bmatrix}}_{A}
\underbrace{\begin{bmatrix}x \\ w \end{bmatrix}}_{q} $$

$$y = A q.$$

"""

"""
To represent this equation in SCICO code, we need to (1) convert $y_1$ and $y_2$ into SCICO arrays; (2) stack  $y_1$ and $y_2$ to form $y$; and (3) write the code for $A$.
"""

"""
For step 1, we'll convert $y_1$ and $y_2$ into SCICO arrays and stack them together. Note how we use snp instead of np; this is necessary when we want to use SCICO's operators and solvers.
"""

from tutorial_funcs import load_ys

import scico.numpy as snp

y1, y2 = load_ys()

y1 = snp.array(y1)  # convert to scico arrays
y2 = snp.array(y2)

"""
For step 2 (stacking $y_1$ and $y_2$), note that `scico.numpy` provides most of the functionality of `numpy`.
(See https://scico.readthedocs.io/en/latest/_autosummary/scico.numpy.html#module-scico.numpy)
**Find the appropriate `scico.numpy` function and use it to stack $y_1$ and $y_2$ into a single array.**
"""

# startq
y = ...
# starta
y = snp.stack((y1, y2))
# endqa

"""

For step 3 (define the linear operator $A$) we will first construct a `forward` function.
The challenge is that the function needs to take a single argument, $q$,
whereas our `forward` function from before currently takes $x$ and $w$ separately. Let's first redefine `forward` as a single argument function.
"""

# startq
def forward(q):
    x = q[0]
    w = q[1]
    y1 = ...  # your code goes here
    y2 = ...
    y = ...
    return y


# starta
def forward(q):
    x = q[0]
    w = q[1]

    N_cols = w.shape[1]
    y1 = x[:, :N_cols] + w
    y2 = x[:, offset : N_cols + offset] + w
    return snp.stack((y1, y2))


# endqa

"""
Now let's test `forward`.
To do that, we need to stack `x` and `w`. However, your solution from before won't work. **Explain why.**
"""

# startq

"""
We can't stack `x` and `w` because...
"""

# starta
"""
We can't stack `x` and `w` because $x$ and $w$ have difference shapes.
"""

# endqa

"""
We could use padding to allow $x$ and $w$ to stack.
Instead, we can make use of SCICO's representation flexibility afforded via `BlockArray`.
`BlockArray`s are lists of arrays intendend to extend NumPy functionality
(really JAX functionality) over groups of arrays with different shapes. We will use SCICO's `BlockArray`
to stack `x` and `w` together.
"""
from tutorial_funcs import load_test_solution

x_test, w_test = load_test_solution()

q_test = snp.blockarray((x_test, w_test))

"""
The shape of a `BlockArray` is simply a tuple of the shapes of all the components. Run the next cell to check the shape of the constructued `BlockArray`.
"""

q_test.shape

"""
Now we can test the `BlockArray` version of the forward model.
"""

# run the forward model
y_test = forward(q_test)

# plot
fig, ax = plt.subplots()
ax.imshow(x_test)
ax.set_title("x_test")
fig.show()

fig, ax = plt.subplots()
ax.imshow(w_test)
ax.set_title("w_test")
fig.show()

fig, ax = plt.subplots()
ax.imshow(y_test[0])
ax.set_title("y_test[0]")

fig, ax = plt.subplots()
ax.imshow(y_test[1])
ax.set_title("y_test[1]")
fig.show()

"""
Finally,
we can use `forward` to construct the linear operator $A$.
**Read the [documentation on defining a linear operator](https://scico.readthedocs.io/en/latest/operator.html#defining-a-new-linear-operator) and define $A$ in the next cell.**
"""
import scico.linop

# startq
A = scico.linop.LinearOperator(...)
# starta
A = scico.linop.LinearOperator(input_shape=q_test.shape, eval_fn=forward)
# endqa

"""
Once we've defined $A$ as a `scico.LinearOperator`,
we can treat it just like a NumPy matrix, including using operators like `A @ q`, `A.T`, `A + A`, and `2*A`.
Run the next cell to see an example of how we could compute $y_1$ and $y_2$
from our previous example.
"""

y_test = A @ q_test

fig, ax = plt.subplots()
ax.imshow(y_test[0])
ax.set_title("y_test[0]")

fig, ax = plt.subplots()
ax.imshow(y_test[1])
ax.set_title(y_test[1])
fig.show()

"""
You are done with part 3. Please report back in the Webex chat: **done with part 3**.

While you wait for others to finish, you could think about what $A^T$ should do and try to plot it in the cell below.
"""

# this cell intentionally left blank

"""
🛑 **PAUSE HERE** 🛑
"""

"""
# Solving the Nonuniform Illumination Problem: SCICO CG Solver
To solve this problem with SCICO,
we can put it into the form
$$ \min_q \| y - Aq \|_2^2,$$
where $y$ is an array, $q$ is a `BlockArray`,
and $A$ is a `scico.LinearOperator`.
"""

"""
Since now we have all the components in place, we  are ready to solve. Because this is a least squares problem, a good approach is to equate the gradient to zero
and solve with the conjugate gradient (CG) method. The gradient is

$$A^T A q - A^T y.$$

Note that the gradient computation requires $A^T$.
One of the advantages of using SCICO is that it uses JAX autograd functionality for computing these expressions avoiding the need to implement them explicitly. We can test the $A^T$ operator available with the definition of the `LinearOperator`.
"""

ATy = A.T @ y

print("AT_eval.shape: ", ATy.shape)

fig, ax = plt.subplots()
ax.imshow(ATy[0])
ax.set_title("ATy[0]")

fig, ax = plt.subplots()
ax.imshow(ATy[1])
ax.set_title("ATy[1]")
fig.show()


"""
Take a look at the documentation for `scico.solver.cg`: https://scico.readthedocs.io/en/latest/_autosummary/scico.solver.html#scico.solver.cg and try to use it to find the $q$ that makes the gradient zero.
"""

import scico.solver

# startq
q_hat, info = scico.solver.cg(...)
# starta
q_hat, info = scico.solver.cg(A.T @ A, A.T @ y, snp.zeros(A.input_shape))
# endqa
print(info)

"""
Run the cell below to see your results!
"""

x_hat = q_hat[0]
w_hat = q_hat[1]

# plot
fig, ax = plt.subplots()
ax.imshow(y1)
ax.set_title("measured $y_1$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(y2)
ax.set_title("measured $y_2$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(x_hat)
ax.set_title("recovered $x$")

fig, ax = plt.subplots()
ax.imshow(w_hat)
ax.set_title("recovered $w$")
fig.show()

"""
You are done with part 4. Please report back in the Webex chat: **done with part 4**.

While you wait for others to finish, you could think about what you like and dislike about the solution you got.
How could you improve it?
How do you know if it is correct?

🛑 **PAUSE HERE** 🛑
"""

"""
## Refining the solution with regularization

If we closely inspect the recovered `x` solution, we can see that it exhibits
some artifacts. In several cases we can improve the quality of results by seeking for a regularized least squares solution

$$ \min_q \| y - Aq \|_2^2 + \, r(q),$$

where $r$ is an appropriate regularization.
"""
"""
In this case,
we look for a solution with a smooth `w`. This can be expressed as the minimization of the $L_2$ norm of its gradients in $x$ and $y$ directions

$$ \min_q \| y - Aq \|_2^2 + \alpha\| D q \|_2^2,$$

where $Dq = D_w w$, and $D_w$ computes finite differences.
SCICO provides an operator to compute these gradients.
**Look through the [list of SCICO operators](https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.html#module-scico.linop) to find the appropriate one
and instantiate it.**
"""

# startq
Dw = scico.linop.FindTheCorrectOperator(input_shape=w_test.shape)
# starta
Dw = scico.linop.FiniteDifference(input_shape=w_test.shape)
# endqa

"""
Run the next cell to see $D$ in action.
"""
Dwy1 = Dw @ y1


fig, ax = plt.subplots()
ax.imshow(Dwy1[1])
ax.set_title(r"$\nabla_x y_1$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(Dwy1[0])
ax.set_title(r"$\nabla_y y_1$")

"""
Note, however, that we want to regularize for $w$ being smooth and this is only a part of $q$, the complete unknown we are solving for.
We need an operator that can index into $q$ (this operation is called slicing in Python);
SCICO has an operator for that!
**Look through the [list of SCICO operators](https://scico.readthedocs.io/en/latest/_autosummary/scico.linop.html#module-scico.linop) to find the appropriate one
and instantiate it.**
"""

# startq
S = scico.linop.FindTheCorrectOperator(..., input_shape=q_test.shape)
# starta
S = scico.linop.Slice(idx=1, input_shape=q_test.shape)
# endqa

"""
Let's test this new operator.
"""

w_test = S @ q_test

fig, ax = plt.subplots()
ax.imshow(w_test)
fig.show()

"""
**In the next cell, combine `Dw` and `S` to make D.** Remember that SCICO supports matrix-like operations on operators.
"""
# startq
D = ...
# starta
D = Dw @ S
# endqa

"""
We are now ready to solve.
We can still use the conjugate gradient (CG) method, we just need to compute the gradient including the regularization

$$A^T A q + \alpha D^T D q - A^T y.$$

**In the next cell, fill in the CG solver.**
"""

alpha = 1.0
# startq
q_hat, info = scico.solver.cg(...)
# starta
q_hat, info = scico.solver.cg(A.T @ A + alpha * D.T @ D, A.T @ y, snp.zeros(A.input_shape))
# endqa
print(info)

"""
Run the cell below to see your results!
"""
x_hat = q_hat[0]
w_hat = q_hat[1]

# plot
fig, ax = plt.subplots()
ax.imshow(y1)
ax.set_title("measured $y_1$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(y2)
ax.set_title("measured $y_2$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(x_hat)
ax.set_title("recovered $x$")

fig, ax = plt.subplots()
ax.imshow(w_hat)
ax.set_title("recovered $w$")
fig.show()

"""
You are done with part 5. Please report back in the Webex chat: **done with part 5**.

While you wait for others to finish, you could think about what effect $\alpha$ should have on your solution.
Try changing it.
What trade off does it control?
How can you pick the best value?

🛑 **PAUSE HERE** 🛑
"""

"""
## Refining the Solution: SCICO Functionals, Losses, Optimizers

If we closely inspect our recovered `x` we can see that it has negative values,
which might be nonphysical.
"""

print("Range of solution")
print("Minimum x: ", x_hat.min())
print("Maximum x: ", x_hat.max())
print("Minimum w: ", w_hat.min())
print("Maximum w: ", w_hat.max())

"""
We can add another regularization term to enforce that $x$ and $w$ are nonnegative

$$\min_q \| y - Aq \|_2^2, + \, r_1(q) + \, r_2(q)$$

$r_1$ is the previously defined smoothness regularization and $r_2$ is the nonnegative constraint.
"""

"""
The nonnegative constraint is implemented in SCICO as a nonnegative indicator functional.
Since this is a nonsmooth functional, it is necessary to use a different solver for this
formulation. In this case, we will use the Proximal Gradient Method `PGM`
(see https://scico.readthedocs.io/en/latest/_autosummary/scico.optimize.html#scico.optimize.PGM)

The `PGM` optimizer solves problems of the form

$$\min_x f(x) + g(x),$$

wher $f$ is a smooth function and $g$ is a function with a proximal map. Therefore, we need to cast our problem
 following the PGM structure. We do it as follows

 $$f(q) = \| y - Aq \|_2^2 + \alpha \| D q \|_2^2,$$
 $$g(q) = \iota_{\mathbb{NN}}(q),$$

with $\iota_{\mathbb{NN}}$ representing a nonnegative indicator function. In this way, we group the derivable (smooth) part
 of optimization problem in $f$, using a regularization hyper-parameter $\alpha > 0$ to trade-off fidelity vs.
 smoothness and keep the non-smooth part in $g$.
"""

"""
We show next how we can achieve this using SCICO losses and functionals.
"""

from scico import functional, loss

"""
First, we express the fidelity term $\| y - Aq \|_2^2,$ using a `scico.loss`.
**Use the [loss docs](https://scico.readthedocs.io/en/latest/_autosummary/scico.loss.html#module-scico.loss) to find the appropriate way to express this data fidelity term in SCICO.**
"""
# y and A included here for readability but are defined as before
y = snp.stack((y1, y2))
A = scico.linop.LinearOperator(input_shape=q_test.shape, eval_fn=forward)

# startq
f_data = loss.FindTheCorrectLoss(...)  # your code here
# starta
f_data = loss.SquaredL2Loss(y=y, A=A)
# endqa

"""
**Define a loss to quantify the smoothness of $w$, $\| D w \|_2^2$**.
"""
# startq
f_smoothness = loss.FindTheCorrectLoss(...)  # your code here
# starta
f_smoothness = loss.SquaredL2Loss(y=snp.zeros(D.output_shape), A=D)
# endqa

"""
To form $f$ from above, we need to make the sum of `f_data` and `f_smoothness`.
Luckily, SCICO supports algebra on losses, just like it does on operators.
**Make $f$ in the cell below.**
"""

alpha = 1.0
# startq
f = ...  # your code here
# starta
f = f_data + alpha * f_smoothness
# endqa

"""
Now we define the nonnegative indicator function, $g$.
**Use the [functional docs](https://scico.readthedocs.io/en/latest/_autosummary/scico.functional.html#module-scico.functional) to find the appropriate way to express this data fidelity term in SCICO.**
"""

# startq
g = functional.FindTheCorrectFunctional(...)  # your code here
# starta
g = functional.NonNegativeIndicator()
# endqa

"""
We set up a PGM optimizer to compute the regularized solution.
"""

from scico.optimize.pgm import AcceleratedPGM

L0 = 1e2  # Initial inverse of stepsize
maxiter = 200  # Maximum iterations to compute
x0 = snp.zeros(q_test.shape)  # Initial estimate of the solution

solver = AcceleratedPGM(
    f=f,
    g=g,
    L0=L0,
    x0=x0,
    maxiter=maxiter,
    itstat_options={"display": True},
)

"""
Run the next cell to see an example of running PGM to compute $y_1$ and $y_2$
from our previous setup.
"""
q_hat = solver.solve()

"""
Run the cell below to see your results!
"""
x_hat = q_hat[0]
w_hat = q_hat[1]

# plot
fig, ax = plt.subplots()
ax.imshow(y1)
ax.set_title("measured $y_1$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(y2)
ax.set_title("measured $y_2$")
fig.show()

fig, ax = plt.subplots()
ax.imshow(x_hat)
ax.set_title("recovered $x$")

fig, ax = plt.subplots()
ax.imshow(w_hat)
ax.set_title("recovered $w$")
fig.show()

"""
Run the cell below to check if the nonnegativity constraint worked.
"""

print("Range of solution")
print("Minimum: ", x_hat.min())
print("Maximum: ", x_hat.max())
print("Minimum: ", w_hat.min())
print("Maximum: ", w_hat.max())

"""
## Conclusion
This tutorial has shown how to set up and solve a simple least squares imaging problem in
SCICO as well as regularized least squares formulations. In doing so, it has demonstrated
a diverse set of classes provided by SCICO such as operators, functionals, losses and
solvers which make expressing regularized optimization problems more straightforward.
"""

"""
You are done with the tutorial! Please report back in the Webex chat: **done with the tutorial**.

While you wait for others to finish, you could check out the [PGM documentation](https://scico.readthedocs.io/en/latest/_autosummary/scico.optimize.html#scico.optimize.PGM) and [PGM step size documentation](https://scico.readthedocs.io/en/latest/_autosummary/scico.optimize.pgm.html) to understand more of the arguments to `AcceleratedPGM`.
"""
