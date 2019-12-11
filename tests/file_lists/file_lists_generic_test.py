# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from ncgenes7.file_lists.generic import MatchSubStringFileList


class TestMatchSubStringFileList(parameterized.TestCase):

    @parameterized.parameters(
        {"split_char": "-", "match_indices": -1,
         "match_pattern_must": "d"},
        {"split_char": "_", "match_indices": [0, 1],
         "match_pattern_must": "filename_1"},
        {"split_char": "_", "match_indices": [0, -2],
         "match_pattern_must": "filename_111"},
        {"split_char": "_", "match_indices": [0, -2],
         "match_pattern_must": "d2::filename_111",
         "directory_matcher_depth": 1}
    )
    def test_match_fn(self, split_char, match_indices, match_pattern_must,
                      directory_matcher_depth=-1):
        fname = ("prefix_"
                 + split_char.join(["filename", "1", "111", "d"])
                 + "_suffix")
        if directory_matcher_depth >= 0:
            fname = "d1/d2/d3/" + fname
        file_list = MatchSubStringFileList(
            file_names={"key": ""},
            match_suffixes={"key": "_suffix"},
            match_prefixes={"key": "prefix_"},
            split_char=split_char,
            match_indices=match_indices,
            directory_matcher_depth=directory_matcher_depth)
        self.assertEqual(match_pattern_must,
                         file_list.match_fn(fname, "key"))
