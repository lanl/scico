#! /bin/bash

# This script installs a conda environment with all required and
# optional scico dependencies. The user is assumed to have write
# permission for the conda installation. It should function correctly
# under both Linux and OSX, but note that there are some additional
# complications in using a conda installed matplotlib under OSX
#   https://matplotlib.org/faq/osx_framework.html
# that are not addressed, and that installation of jaxlib with GPU
# capabilities is not supported under OSX. Note also that additional
# utilities realpath and gsed (gnu sed), available from MacPorts, are
# required to run this script under OSX.
#
# Run with -h flag for usage information
set -e  # exit when any command fails

SCRIPT=$(basename $0)
USAGE=$(cat <<-EOF
Usage: $SCRIPT [-h] [-y] [-g] [-p python_version] [-c cuda_version]
       [-e env_name] [-j jaxlib_url]
          [-h] Display usage information
          [-y] Do not ask for confirmation
          [-g] Install jaxlib with GPU support
          [-c] CUDA version without decimals (e.g. specify 11.0 as 110)
          [-p python_version] Specify Python version (e.g. 3.8)
          [-e env_name] Specify conda environment name
          [-j jaxlib_url] Specify URL for jaxlib pip install
EOF
)

JAXURL=https://storage.googleapis.com/jax-releases/jax_releases.html
AGREE=no
GPU=no
CUVER=""
JLVER="0.1.71"
PYVER="3.8"
ENVNM=py$(echo $PYVER | sed -e 's/\.//g')

# Project requirements files
REQUIRE=$(cat <<-EOF
requirements.txt
dev_requirements.txt
docs/docs_requirements.txt
examples/examples_requirements.txt
examples/notebooks_requirements.txt
EOF
)
# Requirements that cannot be installed via conda (i.e. have to use pip)
NOCONDA=$(cat <<-EOF
bm3d faculty-sphinx-theme py2jn colour_demosaicing ray svmbir
EOF
)


OPTIND=1
while getopts ":hygc:p:e:j:" opt; do
    case $opt in
	c|p|e|j) if [ -z "$OPTARG" ] || [ "${OPTARG:0:1}" = "-" ] ; then
		     echo "Error: option -$opt requires an argument" >&2
		     echo "$USAGE" >&2
		     exit 1
		 fi
		 ;;&
	h) echo "$USAGE"; exit 0;;
	y) AGREE=yes;;
	g) GPU=yes;;
	c) CUVER=$OPTARG;;
	p) PYVER=$OPTARG;;
	e) ENVNM=$OPTARG;;
	j) JAXURL=$OPTARG;;
	:) echo "Error: option -$OPTARG requires an argument" >&2
	   echo "$USAGE" >&2
	   exit 1
	   ;;
	\?) echo "Error: invalid option -$OPTARG" >&2
	    echo "$USAGE" >&2
	    exit 1
	    ;;
  esac
done

shift $((OPTIND-1))
if [ ! $# -eq 0 ] ; then
    echo "Error: no positional arguments" >&2
    echo "$USAGE" >&2
    exit 1
fi

if [ ! "$(which conda 2>/dev/null)" ]; then
    echo "Error: conda command required but not found" >&2
    exit 2
fi

# Not available on BSD systems such as OSX: install via MacPorts etc.
if [ ! "$(which realpath 2>/dev/null)" ]; then
    echo "Error: realpath command required but not found" >&2
    exit 3
fi

OS=$(uname -a | cut -d ' ' -f 1)
case "$OS" in
    Linux)    SOURCEURL=$URLROOT$INSTLINUX; SED=sed;;
    Darwin)   SOURCEURL=$URLROOT$INSTMACOSX; SED=gsed;;
    *)        echo "Error: unsupported operating system $OS" >&2; exit 4;;
esac
if [ "$OS" == "Darwin" -a "$GPU" == yes ]; then
    echo "Error: GPU-enabled jaxlib installation not supported under OSX" >&2
    exit 5
fi
if [ "$OS" == "Darwin" ]; then
    if [ ! "$(which gsed 2>/dev/null)" ]; then
	echo "Error: gsed command required but not found" >&2
	exit 6
    fi
fi

if [ "$GPU" == "yes" ] && [ "$CUVER" == "" ]; then
    if [ "$(which nvcc)" == "" ]; then
	echo "Error: GPU-enabled jaxlib requested but CUDA version not"\
	     "specified and could not be automatically determined" >&2
	exit 7
    else
	CUVER=$(nvcc --version | grep -o 'release [0-9][0-9]*\.[[0-9][0-9]*' \
                              | sed -e 's/release //' -e 's/\.//')
    fi
fi

CONDAHOME=$(conda info --base)
ENVDIR=$CONDAHOME/envs/$ENVNM
if [ -d "$ENVDIR" ]; then
    echo "Error: environment $ENVNM already exists"
    exit 8
fi

if [ "$AGREE" == "no" ]; then
    RSTR="Confirm creation of conda environment $ENVNM with Python $PYVER"
    RSTR="$RSTR [y/N] "
    read -r -p "$RSTR" CNFRM
    if [ "$CNFRM" != 'y' ] && [ "$CNFRM" != 'Y' ]; then
	echo "Cancelling environment creation"
	exit 9
    fi
else
    echo "Creating conda environment $ENVNM with Python $PYVER"
fi

if [ "$AGREE" == "yes" ]; then
    CONDA_FLAGS="-y"
else
    CONDA_FLAGS=""
fi

# Construct merged list of all requirements
if [ "$OS" == "Darwin" ]; then
    ALLREQUIRE=$(mktemp -t condaenv)
else
    ALLREQUIRE=$(mktemp -t condaenv_XXXXXX.txt)
fi
REPOPATH=$(realpath $(dirname $0))
for req in $REQUIRE; do
    pthreq="$REPOPATH/../../$req"
    cat $pthreq >> $ALLREQUIRE
done

# Construct filtered list of requirements: sort, remove duplicates, and
# remove requirements that require special handling
if [ "$OS" == "Darwin" ]; then
    FLTREQUIRE=$(mktemp -t condaenv)
else
    FLTREQUIRE=$(mktemp -t condaenv_XXXXXX.txt)
fi
# Filter the list of requirements; sed patterns are for
#  1st: escape >,<,| characters with a backslash
#  2nd: remove recursive include (-r) lines and packages that require
#       special handling, e.g. jaxlib
sort $ALLREQUIRE | uniq | $SED -E 's/(>|<|\|)/\\\1/g' \
    | $SED -E '/^-r.*|^jaxlib.*|^jax.*|^astra-toolbox.*/d' > $FLTREQUIRE
# Remove requirements that cannot be installed via conda
for nc in $NOCONDA; do
    $SED -i "/^$nc.*\$/d" $FLTREQUIRE
done
# Get list of requirements to be installed via conda
CONDAREQ=$(cat $FLTREQUIRE | xargs)

# Update conda, create new environment, and activate it
conda update $CONDA_FLAGS -n base conda
conda create $CONDA_FLAGS -n $ENVNM python=$PYVER

# See https://stackoverflow.com/a/56155771/1666357
eval "$(conda shell.bash hook)"  # required to avoid errors re: `conda init`
conda activate $ENVNM  # Q: why not `source activate`? A: not always in the path

# Add conda-forge as a backup channel
conda config --env --append channels conda-forge

# Install required conda packages (and extra useful packages)
conda install $CONDA_FLAGS $CONDAREQ ipython

# Utility ffmpeg is required by imageio for reading mp4 video files
# it can also be installed via the system package manager, .e.g.
#    sudo apt install ffmpeg
if [ "$(which ffmpeg)" = '' ]; then
    conda install $CONDA_FLAGS ffmpeg
fi

# Install astra-toolbox
conda install $CONDA_FLAGS -c astra-toolbox/label/dev astra-toolbox

# Install jaxlib and jax, with jaxlib version depending on requested
# GPU support
if [ "$GPU" == "yes" ]; then
    pip install --upgrade jax jaxlib==$JLVER+cuda$CUVER -f $JAXURL
    retval=$?
    if [ $retval -ne 0 ]; then
        echo "Error: jaxlib installation failed"
	exit 10
    fi
else
    pip install --upgrade jax jaxlib
    echo "Jax installed without GPU support. To avoid warning messages,"
    echo "add the following to your .bashrc or .bash_aliases file"
    echo "  export JAX_PLATFORM_NAME=cpu"
fi

# Install other packages that require installation via pip
pip install $NOCONDA


echo "Activate the conda environment with the command"
echo "  conda activate $ENVNM"
echo "The environment can be deactivated with the command"
echo "  conda deactivate"

exit 0
