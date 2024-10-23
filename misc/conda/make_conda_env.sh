#!/usr/bin/env bash

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

if [ "$(cut -d '.' -f 1 <<< "$BASH_VERSION")" -lt "4" ]; then
    echo "Error: this script requires bash version 4 or later" >&2
    exit 1
fi


SCRIPT=$(basename $0)
REPOPATH=$(realpath $(dirname $0))
USAGE=$(cat <<-EOF
Usage: $SCRIPT [-h] [-y] [-g] [-p python_version] [-e env_name]
          [-h] Display usage information
          [-v] Verbose operation
          [-t] Display actions that would be taken but do nothing
          [-y] Do not ask for confirmation
          [-p python_version] Specify Python version (e.g. 3.12)
          [-e env_name] Specify conda environment name
EOF
)

AGREE=no
VERBOSE=no
TEST=no
PYVER="3.12"
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
flax orbax-checkpoint bm3d bm4d py2jn colour_demosaicing hyperopt ray[tune,train]
EOF
)


OPTIND=1
while getopts ":hvtyp:e:" opt; do
    case $opt in
	p|e) if [ -z "$OPTARG" ] || [ "${OPTARG:0:1}" = "-" ] ; then
		     echo "Error: option -$opt requires an argument" >&2
		     echo "$USAGE" >&2
		     exit 2
		 fi
		 ;;&
	h) echo "$USAGE"; exit 0;;
	t) VERBOSE=yes;TEST=yes;;
	v) VERBOSE=yes;;
	y) AGREE=yes;;
	p) PYVER=$OPTARG;;
	e) ENVNM=$OPTARG;;
	:) echo "Error: option -$OPTARG requires an argument" >&2
           echo "$USAGE" >&2
           exit 2
           ;;
	\?) echo "Error: invalid option -$OPTARG" >&2
            echo "$USAGE" >&2
            exit 2
            ;;
    esac
done

shift $((OPTIND-1))
if [ ! $# -eq 0 ] ; then
    echo "Error: no positional arguments" >&2
    echo "$USAGE" >&2
    exit 2
fi

if [ ! "$(which conda 2>/dev/null)" ]; then
    echo "Error: conda command required but not found" >&2
    exit 3
fi

# Not available on BSD systems such as OSX: install via MacPorts etc.
if [ ! "$(which realpath 2>/dev/null)" ]; then
    echo "Error: realpath command required but not found" >&2
    exit 4
fi

# Ensure that a C compiler is available; required for installing svmbir
# On debian/ubuntu linux systems, install package build-essential
if [ -z "$CC" ] && [ ! "$(which gcc 2>/dev/null)" ]; then
    echo "Error: gcc command not found and CC environment variable not set"
    echo "       set CC to the path of your C compiler, or install gcc."
    echo "       On debian/ubuntu, you may need to do"
    echo "           sudo apt install build-essential"
    exit 5
fi

OS=$(uname -a | cut -d ' ' -f 1)
case "$OS" in
    Linux)    SOURCEURL=$URLROOT$INSTLINUX; SED="sed";;
    Darwin)   SOURCEURL=$URLROOT$INSTMACOSX; SED="gsed";;
    *)        echo "Error: unsupported operating system $OS" >&2; exit 6;;
esac
if [ "$OS" == "Darwin" ] && [ "$GPU" == yes ]; then
    echo "Error: GPU-enabled jaxlib installation not supported under OSX" >&2
    exit 7
fi
if [ "$OS" == "Darwin" ]; then
    if [ ! "$(which gsed 2>/dev/null)" ]; then
	echo "Error: gsed command required but not found" >&2
	exit 8
    fi
fi

JLVER=$($SED -n 's/^jaxlib>=.*<=\([0-9\.]*\).*/\1/p' \
	     $REPOPATH/../../requirements.txt)
JXVER=$($SED -n 's/^jax>=.*<=\([0-9\.]*\).*/\1/p' \
	     $REPOPATH/../../requirements.txt)

# Construct merged list of all requirements
if [ "$OS" == "Darwin" ]; then
    ALLREQUIRE=$(/usr/bin/mktemp -t condaenv)
else
    ALLREQUIRE=$(mktemp -t condaenv_XXXXXX.txt)
fi
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
#  2nd: remove comments in requirements file
#  3rd: remove recursive include (-r) lines and packages that require
#       special handling, e.g. jaxlib
sort $ALLREQUIRE | uniq | $SED -E 's/(>|<|\|)/\\\1/g' \
    | $SED -E 's/\#.*$//g' \
    | $SED -E '/^-r.*|^jaxlib.*|^jax.*/d' > $FLTREQUIRE
# Remove requirements that cannot be installed via conda
PIPREQ=""
for nc in $NOCONDA; do
    # Escape [ and ] for use in regex
    nc=$(echo $nc | $SED -E 's/(\[|\])/\\\1/g')
    # Add package to pip package list
    PIPREQ="$PIPREQ "$(grep "$nc" $FLTREQUIRE | $SED 's/\\//g')
    # Remove package $nc from conda package list
    $SED -i "/^$nc.*\$/d" $FLTREQUIRE
done
# Get list of requirements to be installed via conda
CONDAREQ=$(cat $FLTREQUIRE | xargs)

if [ "$VERBOSE" == "yes" ]; then
    echo "Create python $PYVER environment $ENVNM in conda installation"
    echo "    $CONDAHOME"
    echo "Packages to be installed via conda:"
    echo "    $CONDAREQ" | fmt -w 79
    echo "Packages to be installed via pip:"
    echo "    jaxlib==$JLVER jax==$JXVER $PIPREQ" | fmt -w 79
    if [ "$TEST" == "yes" ]; then
	exit 0
    fi
fi

CONDAHOME=$(conda info --base)
ENVDIR=$CONDAHOME/envs/$ENVNM
if [ -d "$ENVDIR" ]; then
    echo "Error: environment $ENVNM already exists"
    exit 9
fi

if [ "$AGREE" == "no" ]; then
    RSTR="Confirm creation of conda environment $ENVNM with Python $PYVER"
    RSTR="$RSTR [y/N] "
    read -r -p "$RSTR" CNFRM
    if [ "$CNFRM" != 'y' ] && [ "$CNFRM" != 'Y' ]; then
	echo "Cancelling environment creation"
	exit 10
    fi
else
    echo "Creating conda environment $ENVNM with Python $PYVER"
fi

if [ "$AGREE" == "yes" ]; then
    CONDA_FLAGS="-y"
else
    CONDA_FLAGS=""
fi


# Update conda, create new environment, and activate it
conda update $CONDA_FLAGS -n base conda
conda create $CONDA_FLAGS -n $ENVNM python=$PYVER

# See https://stackoverflow.com/a/56155771/1666357
eval "$(conda shell.bash hook)"  # required to avoid errors re: `conda init`
conda activate $ENVNM  # Q: why not `source activate`? A: not always in the path

# Add conda-forge channel
conda config --append channels conda-forge

# Install required conda packages (and extra useful packages)
conda install $CONDA_FLAGS $CONDAREQ ipython

# Utility ffmpeg is required by imageio for reading mp4 video files
# it can also be installed via the system package manager, .e.g.
#    sudo apt install ffmpeg
if [ "$(which ffmpeg)" = '' ]; then
    conda install $CONDA_FLAGS ffmpeg
fi

# Install jaxlib and jax
pip install --upgrade jaxlib==$JLVER jax==$JXVER

# Install other packages that require installation via pip
pip install $PIPREQ

# Warn if libopenblas-dev not installed on debian/ubuntu
if [ "$(which dpkg 2>/dev/null)" ]; then
    if [ ! "$(dpkg -s libopenblas-dev 2>/dev/null)" ]; then
	echo "Warning (debian/ubuntu): package libopenblas-dev,"
	echo "which is required by bm3d, does not appear to be"
	echo "installed; install using the command"
	echo "   sudo apt install libopenblas-dev"
    fi
fi

echo
echo "Activate the conda environment with the command"
echo "  conda activate $ENVNM"
echo "The environment can be deactivated with the command"
echo "  conda deactivate"
echo
echo "JAX installed without GPU support. To enable GPU support, install a"
echo "version of jaxlib with CUDA support following the instructions at"
echo "   https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu"
echo "In most cases this just requires the command"
echo "   pip install -U \"jax[cuda12]\""
echo
echo "ASTRA Toolbox installed without GPU support if this script was"
echo "run on a host without CUDA drivers installed. To enable GPU support,"
echo "uninstall and then reinstall the astra-toolbox conda package on a"
echo "host with CUDA drivers installed."

exit 0
