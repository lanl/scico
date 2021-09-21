# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Diagnostic information for iterative solvers."""

import re
from collections import OrderedDict, namedtuple
from typing import List, Optional, Tuple, Union

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class IterationStats:
    """Display and record statistics related to convergence of iterative algorithms"""

    def __init__(
        self,
        fields: OrderedDict,
        ident: Optional[dict] = None,
        display: bool = False,
        colsep: int = 2,
    ):
        """
        The `fields` parameter represents an OrderedDict (to ensure that
        field order is retained) specifying field names for each value to
        be inserted and a corresponding format string for when it is
        displayed. When inserted values are printed in tabular form, the
        field lengths are taken as the maxima of the header string lengths
        and the field lengths embedded in the format strings (if specified).
        For best results, the field lengths should be manually specified based
        on knowledge of the ranges of values that may be encountered. For
        example, for a '%e' format string, the specified field length should
        be at least the precision (e.g. '%.2e' specifies a precision of 2
        places) plus 6 when only positive values may encountered, and plus 7
        when negative values may be encountered.

        Args:
            fields: A dictionary associating field names with format strings for
                displaying the corresponding values
            ident: A dictionary associating field names
               with corresponding valid identifiers for use within the namedtuple used to
               record results.  Defaults to None
            display : Flag indicating whether results should be printed to stdout.
                Defaults to ``False``.
            colsep : Number of spaces seperating fields in displayed tables.
                Defaults to 2.

        Raises:
            TypeError: Description
        """

        # Parameter fields must be specified as an OrderedDict to ensure
        # that field order is retained
        if not isinstance(fields, dict):
            raise TypeError("Parameter fields must be an instance of dict")
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
        flre = re.compile(r"%(-)?(\d+)(.*)")
        # Iterate over field names
        for name in fields:
            # Get format string and decompose it using compiled regex
            fmt = fields[name]
            rem = flre.match(fmt)
            if rem is None:
                # If format string does not contain a field length specifier,
                # the field length is determined by the length of the header
                # string
                fln = len(name)
                fmt = "%%%d" % fln + fmt[1:]
            else:
                # If the format string does contain a field length specifier,
                # the field length is the maximum of specified field length
                # and the length of the header string
                fln = max(len(name), int(rem.group(2)))
                sgn = rem.group(1)
                if sgn is None:
                    sgn = ""
                fmt = "%" + sgn + ("%d" % fln) + rem.group(3)
            self.fieldname.append(name)
            self.fieldformat.append(fmt)
            self.fieldlength.append(fln)
            self.headlength += fln + colsep

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
        """
        Insert a list of values for a single iteration

        Args:
            values : Statistics for a single iteration
        """

        self.iterations.append(self.IterTuple(*values))

        if self.display:
            if self.disphdr is not None:
                print(self.disphdr)
                self.disphdr = None
            print((" " * self.colsep).join(self.fieldformat) % values)

    def history(self, transpose: bool = False):
        """
        Retrieve record of all inserted iterations

        Args:
            transpose: Flag indicating whether results
                should be returned in "transposed" form, i.e. as a namedtuple of lists
                rather than a list of namedtuples.  Default: False

        Returns:
            list of namedtuple or namedtuple of lists: Record of all inserted iterations
        """

        if transpose:
            return self.IterTuple(
                *[
                    [self.iterations[m][n] for m in range(len(self.iterations))]
                    for n in range(len(self.iterations[0]))
                ]
            )
        else:
            return self.iterations
