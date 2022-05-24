#!/usr/bin/env bash

# This script runs each scico unit test module and lists them all in order
# of decreasing run time. It must be run from the repository root directory.

#if [ $(uname -s) != "Linux" ]; then
#    echo "This script is only supported under Linux OS"
#    exit 1
#fi

tmp=/tmp/pytest_time.$$
rm -f $tmp
for f in $(find scico/test -name "test_*.py"); do
    tstr=$(/usr/bin/time -p pytest -qqq --disable-warnings $f 2>&1 | tail -4)
    # Warning does not work in OSX bash
    if grep -q "Command exited with non-zero status" <<<"$tstr"; then
	echo "WARNING: test failure in $f" >&2
    fi
    t=$(grep "^real" <<<"$tstr" | grep -o -E "[0-9\.]*$")
    printf "%6.2f  %s\n" $t $f >> $tmp
done
sort -r -n $tmp
rm $tmp

exit 0
