# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Filters for file lists
"""
import os
from typing import Optional
from typing import Union

import nucleus7 as nc7
from nucleus7.utils import io_utils


class FilterFnamesInSet(nc7.data.DataFilter):
    """
    Filters file names that are inside or outside of particular sets.

    Parameters
    ----------
    include
        can be a list or a json file with list or txt file with each line
        representing file names to include;
        which file names should be included; must match exactly the file names
        that are passed to filter;
        file names that match this will be passed through, otherwise they will
        be filtered out
    exclude
        can be a list or a json file with list or txt file with each line
        representing file names to exclude;
        which file names should be excluded; must match exactly the file names
        that are passed to filter;
        file names that match this will be filtered out
    compare_basenames
        if the basenames must be compared; otherwise absolute file
        names are compared and so must be inside of the include / exclude files

    Raises
    ------
    ValueError
        if none of include or exclude was provided
    ValueError
        if predicate_keys_mapping does not contain 'keys' value or there are
        more than 1 mapping
    """

    def __init__(self,
                 include: Optional[Union[str, list, tuple]] = None,
                 exclude: Optional[Union[str, list, tuple]] = None,
                 compare_basenames: bool = True,
                 **filter_kwargs):
        super().__init__(**filter_kwargs)
        if not include and not exclude:
            raise ValueError(
                "Provide items to include or to exclude to use a filter!")
        if not set(self.predicate_keys_mapping.values()).issubset(
                {"fnames", "_"}):
            raise ValueError("Provide mapping to 'fnames' key! and remap "
                             "to '_' all others!")
        self.include = include
        self.exclude = exclude
        self.compare_basenames = compare_basenames

    def build(self):
        super().build()
        include = _read_from_json_or_txt(self.include) or []
        exclude = _read_from_json_or_txt(self.exclude) or []
        if self.compare_basenames:
            include = (os.path.basename(each_name) for each_name in include)
            exclude = (os.path.basename(each_name) for each_name in exclude)
        self.include = set(include)
        self.exclude = set(exclude)
        return self

    def predicate(self, fnames) -> bool:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if self.compare_basenames:
            fnames = os.path.basename(fnames)
        if self.include and fnames not in self.include:
            return False
        if self.exclude and fnames in self.exclude:
            return False
        return True


def _read_from_json_or_txt(fname_or_data):
    if not isinstance(fname_or_data, str):
        return fname_or_data
    if not os.path.exists(fname_or_data):
        raise FileNotFoundError("file {} not found!".format(
            os.path.realpath(fname_or_data)))

    if os.path.splitext(fname_or_data)[-1] == ".txt":
        with open(fname_or_data, "r") as file:
            data = file.read().splitlines()
        return data

    data = io_utils.maybe_load_json(fname_or_data)
    return data
