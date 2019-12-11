# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
KPI implementation for semantic segmentation
"""

from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import nucleus7 as nc7
from nucleus7.core import AdditiveBuffer
from nucleus7.core import BufferProcessor
import numpy as np

from ncgenes7.utils import io_utils


class SemanticSegmentationMeanIOUKPI(nc7.kpi.KPIAccumulator):
    """
    Calculates intersection over union for semantic segmentation from
    confusion matrix classwise and also mean.

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

    Attributes
    ----------
    incoming_keys
        * predictions : predicted classes
        * labels :  ground truth labels
    generated_keys
        * meanIoU : mean intersection over union
    """
    incoming_keys = [
        "confusion_matrix",
    ]
    generated_keys = [
        "meanIoU",
    ]
    dynamic_generated_keys = True

    def __init__(
            self,
            num_classes: int,
            class_names_to_labels_mapping: Optional[Union[str, dict]] = None,
            **kpi_plugin_kwargs):

        buffer = AdditiveBuffer()
        buffer_processor = BufferProcessor(buffer=buffer)
        super().__init__(buffer_processor=buffer_processor, **kpi_plugin_kwargs)
        self.num_classes = num_classes
        self.class_names_to_labels_mapping = class_names_to_labels_mapping
        self._class_labels_to_names_mapping = None

    def build(self):
        super().build()
        self._build_class_labels_to_names_mapping()
        return self

    def process(self, confusion_matrix) -> dict:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        confusion_matrix = np.sum(confusion_matrix, 0)
        not_empty_classes, empty_classes = (
            _empty_classes_in_confusion_matrix(confusion_matrix))
        if empty_classes.shape[0] > 0:
            warnings.warn("Following classes were not present neither in "
                          "predictions nor in groundtruth: "
                          "{}".format(empty_classes))
        iou_each_class = _iou_for_each_class_np(confusion_matrix)
        mean_iou = np.mean(iou_each_class[not_empty_classes])
        kpi = dict()
        kpi["meanIoU"] = mean_iou
        for each_class in range(self.num_classes):
            class_name = self._class_labels_to_names_mapping[each_class]
            kpi_key_name = "iou-classwise-{}".format(class_name)
            if each_class in empty_classes:
                kpi[kpi_key_name] = np.NAN
            else:
                kpi[kpi_key_name] = iou_each_class[each_class]
        return kpi

    def _build_class_labels_to_names_mapping(self):
        (self.num_classes, self.class_names_to_labels_mapping,
         self._class_labels_to_names_mapping
         ) = io_utils.build_class_labels_to_names_mapping(
             self.num_classes, self.class_names_to_labels_mapping,
             class_id_offset=0)


def _iou_for_each_class_np(segmentation_confusion_matrix: np.ndarray
                           ) -> np.ndarray:
    true_positives = np.diagonal(segmentation_confusion_matrix)
    confusion_matrix_add_rows = np.sum(segmentation_confusion_matrix, axis=0)
    confusion_matrix_add_cols = np.sum(segmentation_confusion_matrix, axis=1)
    union = (confusion_matrix_add_rows
             + confusion_matrix_add_cols
             - true_positives)
    # avoid NaNs
    union = np.where(np.equal(union, 0), np.ones_like(union), union)
    iou_each_class = np.divide(true_positives, union)
    return iou_each_class


def _empty_classes_in_confusion_matrix(
        confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = confusion_matrix.shape[0]
    empty_rows = set(np.where(np.all(confusion_matrix == 0, axis=0))[0])
    empty_cols = set(np.where(np.all(confusion_matrix == 0, axis=1))[0])
    empty_classes = empty_rows.intersection(empty_cols)
    not_empty_classes = np.array(
        [each_class for each_class in range(num_classes)
         if each_class not in empty_classes]).astype(np.int32)
    empty_classes_array = np.array(list(empty_classes), np.int32)
    return not_empty_classes, empty_classes_array
