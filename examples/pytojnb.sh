#! /bin/bash

# Convert a Python script into a Jupyter notebook
# Run as
#    pytojnb.sh <input_py_file> <output_jnb_file>

src=$1
dst=$2

tmp=$(mktemp /tmp/pytojnb.XXXXXX)
trap "rm -f $tmp" 0 2 3 15

# sed usage from:
#   https://stackoverflow.com/questions/37909388

# Remove trailing input
sed '${/^input(/d;}' $src > $tmp
# Remove header comment
sed -i '1,/^$/d' $tmp
# Remove r from r"""
sed -i 's/^r"""$/"""/' $tmp
# Insert notebook plot config after last import
sed -E -i '/^(from|import)[^\n]*/,$!b;//{x;//p;g};//!H;$!d;x;s//&\nplot.config_notebook_plotting()/' $tmp
# Convert citations to nbsphinx-recognized format
sed -E -i 's/:cite:`([^`]+)`/<cite data-cite="\1"\/>/g' $tmp

# Convert modified script to notebook
python -m py2jn $tmp $dst

exit 0
