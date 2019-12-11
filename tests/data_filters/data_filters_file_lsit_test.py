# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized

from ncgenes7.data_filters.file_list import FilterFnamesInSet


class TestFilterFnamesInSet(parameterized.TestCase):

    def setUp(self) -> None:
        self.include = ["a", "b", "c", "d"]
        self.exclude = ["d", "f", "e"]

    @parameterized.parameters(
        {"include": True, "exclude": True, "compare_basenames": False},
        {"include": True, "exclude": True},
        {"include": True, "exclude": True},
        {"include": True, "exclude": False},
        {"include": False, "exclude": True})
    def test_predicate(self, include, exclude, compare_basenames=True):
        if not include and not exclude:
            with self.assertRaises(ValueError):
                _ = FilterFnamesInSet()
            return

        include_list = self.include
        exclude_list = self.exclude
        if compare_basenames:
            include_list = [os.path.join("path/before", each_item)
                            for each_item in include_list]
            exclude_list = [os.path.join("path/before", each_item)
                            for each_item in exclude_list]

        fnames_filter = FilterFnamesInSet(
            include=include_list if include else None,
            exclude=exclude_list if exclude else None,
            compare_basenames=compare_basenames,
            predicate_keys_mapping={"fnames": "fnames"}).build()
        self.assertTrue(fnames_filter.predicate("a"))
        if include:
            self.assertFalse(fnames_filter.predicate("g"))
            if compare_basenames:
                self.assertTrue(fnames_filter.predicate("other/a"))
            else:
                self.assertFalse(fnames_filter.predicate("other/a"))
        if exclude:
            self.assertFalse(fnames_filter.predicate("f"))
        else:
            self.assertTrue(fnames_filter.predicate("d"))
            if compare_basenames:
                self.assertTrue(fnames_filter.predicate("other/d"))
            else:
                self.assertFalse(fnames_filter.predicate("other/d"))
