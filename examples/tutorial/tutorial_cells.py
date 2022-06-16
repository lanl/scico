"""
# Example: Nonuniform Illumination
This example demonstrates defining a linear operator and solving a least squares problem in SCICO.

## Introduction
You set up a new microscope in your lab and take a brightfield image, which we'll call $y_1$.

Run the next cell to see $y_1$.
"""

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
