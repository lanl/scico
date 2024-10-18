#!/usr/bin/env bash

# This script installs miniconda3 in the specified path
#
# Run with -h flag for usage information

URLROOT=https://repo.continuum.io/miniconda/
INSTLINUX=Miniconda3-latest-Linux-x86_64.sh
INSTMACOSX=Miniconda3-latest-MacOSX-x86_64.sh

SCRIPT=$(basename $0)
USAGE=$(cat <<-EOF
Usage: $SCRIPT [-h] [-y] install_path
          [-h] Display usage information
          [-y] Do not ask for confirmation
EOF
)
AGREE=no

OPTIND=1
while getopts ":hy" opt; do
  case $opt in
    h) echo "$USAGE"; exit 0;;
    y) AGREE=yes;;
    \?) echo "Error: invalid option -$OPTARG" >&2
	echo "$USAGE" >&2
	exit 1
	;;
  esac
done

shift $((OPTIND-1))
if [ ! $# -eq 1 ] ; then
    echo "Error: one positional argument required" >&2
    echo "$USAGE" >&2
    exit 1
fi

OS=$(uname -a | cut -d ' ' -f 1)
case "$OS" in
    Linux)    SOURCEURL=$URLROOT$INSTLINUX;;
    Darwin)   SOURCEURL=$URLROOT$INSTMACOSX;;
    *)        echo "Error: unsupported operating system $OS" >&2; exit 2;;
esac

if [ ! "$(which wget 2>/dev/null)" ]; then
    has_wget=0
else
    has_wget=1
fi

if [ ! "$(which curl 2>/dev/null)" ]; then
    has_curl=0
else
    has_curl=1
fi

if [ $has_curl -eq 0 ] && [ $has_wget -eq 0 ]; then
    echo "Error: neither curl nor wget found; at least one required" >&2
    exit 3
fi

INSTALLROOT=$1
if [ ! -d "$INSTALLROOT" ] || [ ! -w "$INSTALLROOT" ]; then
    echo "Error: installation root path \"$INSTALLROOT\" is not a directory "\
	 "or is not writable"  >&2
    exit 4
fi

CONDAHOME=$INSTALLROOT/miniconda3
if [ -d "$CONDAHOME" ]; then
    echo "Error: miniconda3 installation directory $CONDAHOME already exists"\
	 >&2
    exit 5
fi

if [ "$AGREE" == "no" ]; then
    read -r -p "Confirm conda installation in root path $INSTALLROOT [y/N] "\
	 CNFRM
    if [ "$CNFRM" != 'y' ] && [ "$CNFRM" != 'Y' ]; then
	echo "Cancelling installation"
	exit 6
    fi
fi

# Get miniconda bash archive and install it
if [ $has_wget -eq 1 ]; then
    wget $SOURCEURL -O /tmp/miniconda.sh
elif [ $has_curl -eq 1 ]; then
    curl -L $SOURCEURL -o /tmp/miniconda.sh
fi

bash /tmp/miniconda.sh -b -p $CONDAHOME
rm -f /tmp/miniconda.sh

# Initial conda setup
export PATH="$CONDAHOME/bin:$PATH"
hash -r
conda config --set always_yes yes
conda update -q conda
conda info -a

echo "Add the following to your .bashrc or .bash_aliases file"
echo "  export CONDAHOME=$CONDAHOME"
echo "  export PATH=\$PATH:\$CONDAHOME/bin"

exit 0
