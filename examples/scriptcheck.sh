#!/usr/bin/env bash

# Basic test of example script functionality by running them all with
# optimization algorithms configured to use only a small number of iterations.
# Currently only supported under Linux.

SCRIPT=$(basename $0)
SCRIPTPATH=$(realpath $(dirname $0))
USAGE=$(cat <<-EOF
Usage: $SCRIPT [-h] [-d]
          [-h] Display usage information
          [-e] Display excerpt of error message on failure
          [-d] Skip tests involving additional data downloads
          [-t] Skip tests related to learned model training
          [-g] Skip tests that need a GPU
EOF
)

OPTIND=1
DISPLAY_ERROR=0
SKIP_DOWNLOAD=0
SKIP_TRAINING=0
SKIP_GPU=0
while getopts ":hedtg" opt; do
    case $opt in
    h) echo "$USAGE"; exit 0;;
    e) DISPLAY_ERROR=1;;
    d) SKIP_DOWNLOAD=1;;
    t) SKIP_TRAINING=1;;
    g) SKIP_GPU=1;;
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
    exit 2
fi

# Set environment variables and paths. This script is assumed to be run
# from its root directory.
export PYTHONPATH=$SCRIPTPATH/..
export PYTHONIOENCODING=utf-8
export MPLBACKEND=agg
export PYTHONWARNINGS=ignore:Matplotlib:UserWarning
d='/tmp/scriptcheck_'$$
mkdir -p $d
retval=0

# On SIGINT clean up temporary script directory and exit.
function cleanupexit {
    rm $d/*.py
    rmdir $d
    exit 2
}
trap cleanupexit SIGINT

# Define regex strings.
re1="s/'maxiter' ?: ?[0-9]+/'maxiter': 2/g; "
re2="s/^maxiter ?= ?[0-9]+/maxiter = 2/g; "
re3="s/^N ?= ?[0-9]+/N = 32/g; "
re4="s/num_samples= ?[0-9]+/num_samples = 2/g; "
re5='s/\"cpu\": ?[0-9]+/\"cpu\": 1/g; '
re6="s/^downsampling_rate ?= ?[0-9]+/downsampling_rate = 12/g; "
re7="s/input\(/#input\(/g; "
re8="s/fig.show\(/#fig.show\(/g"

# Iterate over all scripts.
for f in $SCRIPTPATH/scripts/*.py; do

    printf "%-50s " $(basename $f)

    # Skip problem cases.
    if [ $SKIP_DOWNLOAD -eq 1 ] && grep -q '_microscopy' <<< $f; then
        printf "%s\n" skipped
        continue
    fi
    if [ $SKIP_TRAINING -eq 1 ]; then
    if grep -q '_datagen' <<< $f || grep -q '_train' <<< $f; then
        printf "%s\n" skipped
        continue
        fi
    fi
    if [ $SKIP_GPU -eq 1 ] && grep -q '_astra_3d' <<< $f; then
        printf "%s\n" skipped
        continue
    fi
    if [ $SKIP_GPU -eq 1 ] && grep -q 'ct_projector_comparison_3d' <<< $f; then
        printf "%s\n" skipped
        continue
    fi

    # Create temporary copy of script with all algorithm maxiter values set
    # to small number and final input statements commented out.
    g=$d/$(basename $f)
    sed -E -e "$re1$re2$re3$re4$re5$re6$re7$re8" $f > $g

    # Run temporary script and print status message.
    if output=$(timeout 180s python $g 2>&1); then
        printf "%s\n" succeeded
    else
        printf "%s\n" FAILED
        retval=1
    if [ $DISPLAY_ERROR -eq 1 ]; then
       echo "$output" | tail -8 | sed -e 's/^/    /'
    fi
    fi

    # Remove temporary script.
    rm -f $g

done

# Remove temporary script directory.
rmdir $d

exit $retval
