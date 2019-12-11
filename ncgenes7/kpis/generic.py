# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Generic KPI
"""

import nucleus7 as nc7
import numpy as np
import sklearn.metrics


class ConfusionMatrixKPI(nc7.kpi.KPIPlugin):
    """
    Calculates confusion matrix from labels and predictions

    Parameters
    ----------
    num_classes
        number of classes

    Attributes
    ----------
    incoming_keys
        * predictions : predicted classes
        * labels :  ground truth labels
    """
    incoming_keys = [
        "predictions",
        "labels",
    ]
    generated_keys = [
        "confusion_matrix",
    ]

    def __init__(self,
                 num_classes: int,
                 **kpi_plugin_kwargs):
        super().__init__(**kpi_plugin_kwargs)
        self.num_classes = num_classes

    def process(self, predictions: np.ndarray, labels: np.ndarray) -> dict:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(labels.shape) > 1:
            labels = labels.flatten()
        confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true=labels, y_pred=predictions,
            labels=np.arange(self.num_classes)
        )
        return {"confusion_matrix": confusion_matrix}
