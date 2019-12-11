# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Generic file lists
"""
import os
from typing import List
from typing import Union

import nucleus7 as nc7


class MatchSubStringFileList(nc7.data.FileListExtendedMatch):
    """
    File list that first subtracts suffix and prefix and then can split
    the rest using a split_char and select splits on indices and joins them
    with same split_char to build the matching prefix

    Parameters
    ----------
    split_char
        character in the file name to split with
    match_indices
        indices of the split name to use for matching
    directory_matcher_depth
        if is >= 0, then directory of this depth starting from the file name
        will be used, e.g. d2/d1/d0/fname will result to d1 will be added
        to match pattern if directory_matcher_depth = 0
    """

    def __init__(self, *,
                 split_char: str = "_",
                 match_indices: Union[int, List[int]] = -1,
                 directory_matcher_depth: int = -1,
                 **file_list_kwargs):
        super().__init__(**file_list_kwargs)
        if isinstance(match_indices, list) and len(match_indices) < 2:
            msg = ("match_indices must have length > 1 or be an int! "
                   "(provided: {})".format(match_indices))
            raise ValueError(msg)

        self.split_char = split_char
        self.match_indices = match_indices
        self.directory_matcher_depth = directory_matcher_depth

    def match_fn(self, path: str, key: str) -> str:
        match_pattern_base = super().match_fn(path, key)
        pattern_split = match_pattern_base.split(self.split_char)
        if isinstance(self.match_indices, int):
            match_pattern = pattern_split[self.match_indices]
        else:
            match_pattern = "_".join(
                [pattern_split[each_ind] for each_ind in self.match_indices])
        if self.directory_matcher_depth >= 0:
            directory_path = path.split(
                os.path.sep)[-(self.directory_matcher_depth + 2)]
            match_pattern = "::".join([directory_path, match_pattern])
        return match_pattern
