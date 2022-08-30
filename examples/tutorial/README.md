# SCICO interactive tutorial
This folder contains the source required to build the jupyter notebooks that we use
for the the interactive SCICO tutorial.

## Instructions for building go here
* `pip install grip`
* `>> make`

## Formatting tutorial_XXX.py files
* Use markdown syntax:
  * inline math: \$ delimiter
  * equation: \$\$ delimiter
  * heading: \## Heading
* Use this structure for question and answer pairs
```python
# startq
def f(x):
    return ...
# starta
def f(x):
    return x**2
# endqa
```