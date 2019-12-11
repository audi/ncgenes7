# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Generic IO utils
"""
from typing import Optional

from nucleus7.utils import io_utils


def build_class_labels_to_names_mapping(
        num_classes: Optional[int] = None,
        class_names_to_labels_mapping: Optional[dict] = None,
        class_id_offset: int = 0):
    """
    Build mapping from class id to its names and vice-verse from that mapping or
    from num_classes only. If no mapping was provided, class names are
    "class_{class_id}".

    Parameters
    ----------
    num_classes
        number of classes
    class_names_to_labels_mapping
        file name or mapping itself; mapping should be in format
        {"class name": {"class_id": 1}, "other class_name": {"class_id": 2}},
        where class_id is an unique integer; if multiple class names have same
        class id, then the last name inside of json file will be used as class
        name
    class_id_offset
        offset of class id

    Returns
    -------
    num_classes
        number of classes
    class_names_to_labels_mapping
        mapping in format
        {"class name": {"class_id": 1}, "other class_name": {"class_id": 2}}
    class_labels_to_names_mapping
        mapping from class id to class name
    """
    assert (num_classes is not None
            or class_names_to_labels_mapping is not None
            ), "Provide num_classes or class_names_to_labels_mapping!"

    if not class_names_to_labels_mapping:
        class_labels_to_names_mapping = {
            class_label: "class_{}".format(class_label)
            for class_label in range(class_id_offset,
                                     num_classes + class_id_offset)}
        return (num_classes, class_names_to_labels_mapping,
                class_labels_to_names_mapping)

    class_names_to_labels_mapping = io_utils.maybe_load_json(
        class_names_to_labels_mapping, as_ordereddict=True)
    class_labels_to_names_mapping = {
        v['class_id']: k for k, v in
        reversed(class_names_to_labels_mapping.items())}
    num_classes = len(class_labels_to_names_mapping)

    return (num_classes, class_names_to_labels_mapping,
            class_labels_to_names_mapping)
