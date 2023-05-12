# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Simplified interfaces to :doc:`ray <ray:index>`."""


try:
    from ray import get, put
    from ray.air.session import report
except ImportError:
    raise ImportError("Could not import ray; please install it.")
