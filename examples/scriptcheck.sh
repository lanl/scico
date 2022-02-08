#! /bin/bash

# Basic test of example script functionality by running them all with
# optimization algorithms configured to use only a small number of iterations.
# Currently only supported under Linux.

# Check for presence of Xvfb tool which is used to avoid plots being displayed.
if [ ! "$(which Xvfb 2>/dev/null)" ]; then
    msg="Warning: required tool Xvfb not found: functionality will be degraded"
    echo $msg >&2
    pid=0
else
    Xvfb :11 -screen 0 800x600x16 > /dev/null 2>&1 &
    pid=$!
    export DISPLAY=:10.0
fi

# Set environment variables and paths. This script is assumed to be run
# from its root directory.
export PYTHONPATH=$(cd .. && pwd)
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
re1="s/'maxiter' ?: ?[0-9]+/'maxiter': 3/g; "
re2="s/maxiter ?= ?[0-9]+/maxiter = 3/g; "
re3="s/input\(+/#input\(/g"

# Iterate over all scripts.
for f in scripts/*.py; do

    # Create temporary copy of script with all algorithm maxiter values set
    # to small number and final input statements commented out.
    g=$d/$(basename $f)
    sed -E -e "$re1$re2$re3" $f > $g

    # Run temporary script.
    python $g > /dev/null 2>&1

    # Run temporary script and print status message.
    if python $g > /dev/null 2>&1; then
        printf "%-50s %s\n" $f succeeded
    else
        printf "%-50s %s\n" $f FAILED
        retval=1
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
