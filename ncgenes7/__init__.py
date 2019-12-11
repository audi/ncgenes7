# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
nucleotide implementations for nucleus7 package
"""
import os

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

from ncgenes7 import augmentations
from ncgenes7 import callbacks
from ncgenes7 import data_feeders
from ncgenes7 import data_readers
from ncgenes7 import kpis
from ncgenes7 import losses
from ncgenes7 import metrics
from ncgenes7 import plugins
from ncgenes7 import postprocessors

# pylint: disable=invalid-name
project_root_dir = os.path.abspath(os.path.dirname(__file__))
version_file_name = os.path.join(project_root_dir, "VERSION")
__version__ = open(version_file_name, "r").read().strip()

del os
del project_root_dir
del version_file_name
