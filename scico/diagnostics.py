# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Diagnostic information for iterative solvers."""

import re
import warnings
from collections import OrderedDict, namedtuple
from typing import List, Optional, Tuple, Union

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class IterationStats:
    """Display and record iterative algorithms statistics.

    Display and record statistics related to convergence of iterative
    algorithms.
    """

    def __init__(
        self,
        fields: OrderedDict,
        ident: Optional[dict] = None,
        display: bool = False,
        period: int = 1,
        overwrite: bool = True,
        colsep: int = 2,
    ):
        """
        The `fields` parameter represents an OrderedDict (to ensure that
        field order is retained) specifying field names for each value to
        be inserted and a corresponding format string for when it is
        displayed. When inserted values are printed in tabular form, the
        field lengths are taken as the maxima of the header string
        lengths and the field lengths embedded in the format strings (if
        specified). For best results, the field lengths should be
        manually specified based on knowledge of the ranges of values
        that may be encountered. For example, for a '%e' format string,
        the specified field length should be at least the precision (e.g.
        '%.2e' specifies a precision of 2 places) plus 6 when only
        positive values may encountered, and plus 7 when negative values
        may be encountered.

        Args:
            fields: A dictionary associating field names with format
                strings for displaying the corresponding values.
            ident: A dictionary associating field names.
               with corresponding valid identifiers for use within the
               namedtuple used to record results. Defaults to ``None``.
            display: Flag indicating whether results should be printed
                to stdout. Defaults to ``False``.
            period: Only display one result in every cycle of length
                `period`.
            overwrite: If ``True``, display all results, but each one
                 overwrites the next, except for one result per cycle.
            colsep: Number of spaces seperating fields in displayed
                tables. Defaults to 2.

        Raises:
            TypeError: If the ``fields`` parameter is not a dict.
        """

        # Parameter fields must be specified as an OrderedDict to ensure
        # that field order is retained
        if not isinstance(fields, dict):
            raise TypeError("Parameter fields must be an instance of dict")
        # Subsampling rate of results that are to be displayed
        self.period = period
        # Flag indicating whether to display and overwrite, or not display at all
        self.overwrite = overwrite
        # Number of spaces seperating fields in displayed tables
        self.colsep = colsep
        # Main list of inserted values
        self.iterations = []
        # Total length of header string in displayed tables
        self.headlength = 0
        # List of field names
        self.fieldname = []
        # List of field format strings
        self.fieldformat = []
        # List of lengths of each field in displayed tables
        self.fieldlength = []
        # Names of fields in namedtuple used to record iteration values
        self.tuplefields = []
        # Compile regex for decomposing format strings
        fmre = re.compile(r"%(\+?-?)((?:\d+)?)(\.?)((?:\d+)?)([a-z])")
        # Iterate over field names
        for name in fields:
            # Get format string and decompose it using compiled regex
            fmt = fields[name]
            fmtmatch = fmre.match(fmt)
            if not fmtmatch:
                raise ValueError(f'Format string "{fmt}" could not be parsed')
            fmflg, fmlen, fmdot, fmprc, fmtyp = fmtmatch.groups()
            flen = len(fmt % 0)
            # Warn if actual formatted length longer than specified field
            # length, e.g. as in "%4e"
            if fmlen != "" and flen > int(fmlen):
                warnings.warn(
                    f'Actual length {flen} of format "{fmt}" for field '
                    f'"{name}" is longer than specified value {fmlen}',
                    stacklevel=2,
                )
            # If the actual formatted length is less than that of the header
            # string, insert a field length specifier to increase the
            # length to that of the header string
            if flen < len(name):
                fmt = f"%{fmflg}{len(name)}{fmdot}{fmprc}{fmtyp}"
                flen = len(name)
            self.fieldname.append(name)
            self.fieldformat.append(fmt)
            self.fieldlength.append(flen)
            self.headlength += flen + colsep

            # If a distinct identifier is specified for this field, use it
            # as the namedtuple identifier, otherwise compute it from the
            # field name
            if ident is not None and name in ident:
                self.tuplefields.append(ident[name])
            else:
                # See https://stackoverflow.com/a/3305731
                tfnm = re.sub(r"\W+|^(?=\d)", "_", name)
                if tfnm[0] == "_":
                    tfnm = tfnm[1:]
                self.tuplefields.append(tfnm)

        # Decrement head length to account for final colsep added
        self.headlength -= colsep

        # Construct namedtuple used to record values
        self.IterTuple = namedtuple("IterationStatsTuple", self.tuplefields)

        # Set up table header string display if requested
        self.display = display
        self.disphdr = None
        if display:
            self.disphdr = (
                (" " * colsep).join(
                    ["%-*s" % (fl, fn) for fl, fn in zip(self.fieldlength, self.fieldname)]
                )
                + "\n"
                + "-" * self.headlength
            )

    def insert(self, values: Union[List, Tuple]):
        """Insert a list of values for a single iteration.

        Args:
            values: Statistics for a single iteration.
        """

        self.iterations.append(self.IterTuple(*values))

        if self.display:
            if self.disphdr is not None:
                print(self.disphdr)
                self.disphdr = None
            if self.overwrite:
                if (len(self.iterations) - 1) % self.period == 0:
                    end = "\n"
                else:
                    end = "\r"
                print((" " * self.colsep).join(self.fieldformat) % values, end=end)
            else:
                if (len(self.iterations) - 1) % self.period == 0:
                    print((" " * self.colsep).join(self.fieldformat) % values)

    def end(self):
        """Mark end of iterations.

        This method should be called at the end of a set of iterations.
        Its only function is to ensure that the displayed output is left
        in an appropriate state when overwriting is active with a display
        period other than unity.
        """
        if self.overwrite and self.period > 1 and (len(self.iterations) - 1) % self.period:
            print()

    def history(self, transpose: bool = False):
        """Retrieve record of all inserted iterations.

        Args:
            transpose: Flag indicating whether results should be returned
                in "transposed" form, i.e. as a namedtuple of lists
                rather than a list of namedtuples. Default: False.

        Returns:
            list of namedtuple or namedtuple of lists: Record of all
            inserted iterations.
        """

        if transpose:
            return self.IterTuple(
                *[
                    [self.iterations[m][n] for m in range(len(self.iterations))]
                    for n in range(len(self.iterations[0]))
                ]
            )
        return self.iterations
