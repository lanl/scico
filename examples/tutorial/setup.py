"""
# Setup commands for the SCICO tutorial
This notebook includes the commands you need to run to set up your Colab runtime
so it is ready for the interactive SCICO tutorial.

Run the cell below.

If you get a popup with 'Warning: This notebook was not authored by Google.', select 'Run anyway'.
You should see console outputs appearing.
The install will take several minutes;
when it is finished, you should see `Resolving deltas: 100% (187/187), done.`.
"""

!pip install -q condacolab
import condacolab

condacolab.install()

!pip install git+https://github.com/lanl/scico@cristina/more-flax
!pip install xdesign
!conda install -c astra-toolbox astra-toolbox

!git clone -b cristina-mike/tutorial https://github.com/lanl/scico-data.git

"""
To test your installation, run the cell below.

If you get no errors and a figure of some cells appears,
you installation succeeded.
If you get errors or see nothing, contact the instructors:
`cgarciac@lanl.gov` and `mccann@lanl.gov`.

"""
%cd /content/scico-data/notebooks/tutorial

import matplotlib.pyplot as plt
from tutorial_funcs import load_y1

import scico
from scico import plot

plot.config_notebook_plotting()  # set up plotting

y1 = load_y1()
fig, ax = plt.subplots()
ax.imshow(y1)
ax.set_title("$y_1$")
fig.show()
