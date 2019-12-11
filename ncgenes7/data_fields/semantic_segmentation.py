# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Containers with names of input / outputs to use for semantic segmentation
"""


# pylint: disable=too-few-public-methods
# serves more as a container and not as an interface
class SegmentationDataFields:
    """
    Names for the groundtruth data.


    Parameters
    ----------
    segmentation_classes
        image with segmentation labels
    segmentation_classes_PNG
        image with segmentation labels with PNG encoding
    segmentation_class_probabilities
        raw class probabilities
    segmentation_class_logits
        segmentation class logits
    segmentation_rgb_codings
        rgb encodings for segmentation classes
    segmentation_edges
        image with edges
    segmentation_edges_PNG
        image with edges with PNG encoding
    segmentation_classes_fnames
        file name of segmentation classes
    segmentation_edges_fnames
        file name of edges
    """
    segmentation_classes = 'segmentation_classes'
    segmentation_classes_PNG = 'segmentation_classes_PNG'
    segmentation_class_probabilities = 'segmentation_class_probabilities'
    segmentation_class_logits = 'segmentation_class_logits'
    segmentation_edges = 'segmentation_edges'
    segmentation_edges_PNG = 'segmentation_edges_PNG'
    segmentation_rgb_codings = 'segmentation_rgb_codings'
    segmentation_classes_fnames = "segmentation_classes_fnames"
    segmentation_edges_fnames = "segmentation_edges_fnames"


class GroundtruthDataFields:
    """
    Names for the groundtruth data.


    Parameters
    ----------
    groundtruth_segmentation_classes
        image with segmentation labels
    groundtruth_segmentation_class_probabilities
        raw class probabilities
    groundtruth_segmentation_class_logits
        class logits
    groundtruth_segmentation_rgb_codings
        rgb encodings for segmentation classes
    groundtruth_segmentation_edges
        image with edges
    groundtruth_segmentation_classes_fnames
        file name of groundtruth
    groundtruth_groundtruth_edges_fnames
        file name of groundtruth edge
    """
    groundtruth_segmentation_classes = 'groundtruth_segmentation_classes'
    groundtruth_segmentation_class_probabilities = (
        'groundtruth_segmentation_class_probabilities')
    groundtruth_segmentation_class_logits = (
        'groundtruth_segmentation_class_logits')
    groundtruth_segmentation_edges = 'groundtruth_segmentation_edges'
    groundtruth_segmentation_rgb_codings = (
        'groundtruth_segmentation_rgb_codings')
    groundtruth_segmentation_classes_fnames = (
        "groundtruth_segmentation_classes_fnames")
    groundtruth_segmentation_edges_fnames = (
        "groundtruth_segmentation_edges_fnames")


class PredictionDataFields:
    """
    Names for the groundtruth data.


    Parameters
    ----------
    prediction_segmentation_classes
        image with segmentation labels
    prediction_segmentation_class_probabilities
        raw class probabilities
    prediction_segmentation_rgb_codings
        rgb encodings for segmentation classes
    prediction_segmentation_edges
        image with edges
    prediction_segmentation_classes_fnames
        file name of groundtruth
    prediction_groundtruth_edges_fnames
        file name of groundtruth edge
    """
    prediction_segmentation_classes = 'prediction_segmentation_classes'
    prediction_segmentation_class_probabilities = (
        'prediction_segmentation_class_probabilities')
    prediction_segmentation_class_logits = (
        'prediction_segmentation_class_logits')
    prediction_segmentation_edges = 'prediction_segmentation_edges'
    prediction_segmentation_rgb_codings = 'prediction_segmentation_rgb_codings'
    prediction_segmentation_classes_fnames = (
        "prediction_segmentation_classes_fnames")
    prediction_segmentation_edges_fnames = (
        "prediction_segmentation_edges_fnames")
