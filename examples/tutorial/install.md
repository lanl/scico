[comment]: <> (pandoc --variable linkcolor=blue  -t html5 install.md -o test.pdf)

# SCICO tutorial installation instructions
Follow these instructions to get ready to follow the SCICO interactive tutorial.
If you have any trouble, email Cristina (cgarciac@lanl.gov) and Mike (mccann@lanl.gov) for help.

1. Get working installations of conda ([link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) and git ([link](https://git-scm.com/downloads)).
Many systems will already have these installed; you can check by openning a terminal and typing<br>
```conda --version```
<br>and<br>
```git --version```.

2. Clone the SCICO repository and checkout correct version for the tutorial.<br>
```git clone https://github.com/lanl/scico scico-tutorial```<br>
```cd scico-tutorial```<br>
```git checkout origin/mike/BA-example-bug```<br>
(It is fine to ignore warnings about "detached HEAD".)

3. Make a conda environment and install SCICO dependencies and SCICO itself.
If convenient, disconnect from any company VPN before running these steps.<br>
```conda create -y -n scico-tutorial python=3.8```<br>
```conda activate scico-tutorial```<br>
```pip install -r requirements.txt```<br>
```pip install -e .```<br>

4. Install JupyterLab<br>
```pip install jupyterlab```

5. Run the tutorial notebook. A browser window should open automatically showing the tutorial. If it does not,
follow the instructions that appear in your terminal.<br>
<move `install_test.ipynb` into the `scico-tutorial` directory> <br>
```jupyter notebook install_test.ipynb```

6. Run the first cell of the notebook (shift + return).
If you see the message "Your SCICO installation seems to be working!"
you are ready for the tutorial.
