# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for file list operations
"""

import logging
import os


def maybe_create_sub_dirs(parent_dir: str, subdir: str):
    """
    Create folders starting from parent_dir to subdir

    If parent_dir does not exists, will not create anything

    Parameters
    ----------
    parent_dir
        directory to create subdirectories inside
    subdir
        name of subdirectory, possibly nested, to create
    """
    logger = logging.getLogger(__name__)
    if not os.path.isdir(parent_dir):
        logger.debug(
            'Parent directory %s does not exist so no subdirectories '
            'will be generated!!!', parent_dir)
        return
    full_dir = os.path.join(parent_dir, subdir)
    try:
        os.makedirs(full_dir)
        logger.info('Create directory %s', full_dir)
    except OSError:
        logger.debug('Directory %s already exists', full_dir)
        return
