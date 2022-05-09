FROM jupyter/base-notebook

# Install git
RUN conda update -n base conda
RUN conda install -c anaconda git
RUN conda install -c conda-forge gcc

# Clone into latest scico and scico-data release
RUN mkdir scico_repo
WORKDIR ./scico_repo
RUN git clone --depth 1 --branch v0.0.2 --recurse-submodules https://github.com/lanl/scico.git

# Install scico and dependencies on base jupyter notebook
RUN pip install --upgrade pip
RUN pip3 install -r scico/requirements.txt
RUN pip install xdesign
RUN pip install -r scico/dev_requirements.txt
RUN pip install -e scico/.
RUN pip uninstall jax -y
RUN pip uninstall jaxlib -y
RUN conda install -c conda-forge jaxlib
RUN conda install -c conda-forge jax
RUN conda install -c astra-toolbox astra-toolbox

# Switch working directory to notebooks
WORKDIR ./scico/data/notebooks
