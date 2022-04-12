import platform

from packaging.version import parse


def osx_ver_geq_than(verstr):
    """Determine relative platform OSX version.

    Determine whether platform has OSX version that is as recent as or
    more recent than verstr. Returns ``False`` if the OS is not OSX.
    """
    if platform.system() != "Darwin":
        return False
    osxver = platform.mac_ver()[0]
    return parse(osxver) >= parse(verstr)
