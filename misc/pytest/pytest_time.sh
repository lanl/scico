#!/usr/bin/env bash

# This script runs each scico unit test module and lists them all in order
# of decreasing run time. It must be run from the repository root directory.

tmp=/tmp/pytest_time.$$
rm -f $tmp
for f in $(find scico/test/ -name "test_*.py"); do
    t=$(/usr/bin/time -f "%e" pytest -qqq --disable-warnings $f 2>&1\
	    | grep -o -E "[0-9\.]*$")
    printf "%6.2f  %s\n" $t $f >> $tmp
done
sort -r -n $tmp
rm $tmp

exit 0
