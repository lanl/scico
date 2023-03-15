#! /bin/bash

# Determine whether files are acceptable for commit into main scico repo

size_threshold=65536

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
OS=$(uname -a | cut -d ' ' -f 1)

for f in $@; do
    echo $f
    case "$OS" in
        Linux)  size=$(stat --format "%s" $f);;
        Darwin) size=$(stat -f "%z" $f);;
        *)      echo "Error: unsupported operating system $OS" >&2; exit 1;;
    esac
    if [ $size -gt $size_threshold ]; then
        echo "file exceeds maximum allowable size of $size_threshold bytes"
        echo "raw data and ipynb files should go in scico-data"
        exit 2
    fi
    charset=$(file -b --mime $f | sed -e 's/.*charset=//')
    if [ ! -L "$f" ] && [ "$charset" = "binary" ]; then
        echo "binary files cannot be commited to the repository"
        echo "raw data and ipynb files should go in scico-data"
        exit 3
    fi
    basename=$(basename -- "$f")
    ext="${basename##*.}"
    if [ "$ext" = "ipynb" ]; then
        echo "ipynb files cannot be commited to the repository"
        echo "raw data and ipynb files should go in scico-data"
        exit 4
    fi
done

IFS=$SAVEIFS

exit 0
