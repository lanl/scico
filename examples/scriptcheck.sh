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
EOF
)

OPTIND=1
DISPLAY_ERROR=0
SKIP_DOWNLOAD=0
while getopts ":hed" opt; do
    case $opt in
	h) echo "$USAGE"; exit 0;;
	e) DISPLAY_ERROR=1;;
	d) SKIP_DOWNLOAD=1;;
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

# Check for presence of Xvfb tool which is used to avoid plots being displayed.
if [ ! "$(which Xvfb 2>/dev/null)" ]; then
    msg="Warning: required tool Xvfb not found: functionality will be degraded"
    echo $msg >&2
    pid=0
else
    Xvfb :20 -screen 0 800x600x16 > /dev/null 2>&1 &
    pid=$!
    export DISPLAY=:20.0
fi

# Set environment variables and paths. This script is assumed to be run
# from its root directory.
export PYTHONPATH=$SCRIPTPATH/..
export PYTHONIOENCODING=utf-8
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

    # Create temporary copy of script with all algorithm maxiter values set
    # to small number and final input statements commented out.
    g=$d/$(basename $f)
    sed -E -e "$re1$re2$re3$re4$re5$re6$re7$re8" $f > $g

    # Run temporary script and print status message.
    if output=$(timeout 60s python $g 2>&1); then
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

# Kill Xvfb process if it was started.
if [ $pid != 0 ]; then
  kill $pid
fi

exit $retval
