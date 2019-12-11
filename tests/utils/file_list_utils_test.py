# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.utils import file_list_utils


class TestFIleListUtils(parameterized.TestCase, tf.test.TestCase):

    def test_maybe_create_sub_dirs(self):
        parent_dir = self.get_temp_dir()
        subdir = "a/b/c"
        file_list_utils.maybe_create_sub_dirs(parent_dir, subdir)
        self.assertTrue(os.path.exists(os.path.join(parent_dir, subdir)))
        file_list_utils.maybe_create_sub_dirs(parent_dir, subdir)
