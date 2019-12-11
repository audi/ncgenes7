# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Containers with names of input / outputs to use for object detection

Similar to :obj:`object_detection.core.standard_fields`
"""


# pylint: disable=too-few-public-methods
# serves more as a container and not as an interface
class ObjectDataFields:
    """
    Naming conventions for storing the output of the objects.

    Attributes
    ----------
    object_instance_ids
        instance ids of detections
    object_boxes
        coordinates of the object boxes in the image.
    object_truncation
        truncation information of object
    object_occlusion
        occlusion information of object
    object_observation_angle
        observation angle of object
    object_scores
        detection scores for the object boxes in the image.
    object_classes
        detection-level class labels.
    object_instance_masks
        contains a segmentation mask for each object box.
    object_instance_masks_on_image
        object masks remapped to image
    object_boundaries
        contains an object boundary for each object box.
    object_keypoints
        contains object keypoints for each object box.
    object_keypoints_visibilities
        object keypoint visibilities
    object_keypoints_heatmaps
        contains object keypoint heatmaps
    object_keypoints_heatmaps_on_image
        object keypoint heatmaps remapped to image
    object_keypoints_scores
        scores for object keypoints
    num_objects
        number of detections in the batch.
    object_fnames
        file name with stored object
    images_with_objects
        images with objects
    """
    object_instance_ids = 'object_instance_ids'
    object_boxes = 'object_boxes'
    object_truncation = 'object_truncation'
    object_occlusion = 'object_occlusion'
    object_observation_angle = 'object_observation_angle'
    object_scores = 'object_scores'
    object_classes = 'object_classes'
    object_instance_masks = 'object_instance_masks'
    object_instance_masks_on_image = 'object_instance_masks_on_image'
    object_boundaries = 'object_boundaries'
    object_keypoints = 'object_keypoints'
    object_keypoints_visibilities = 'object_keypoints_visibilities'
    object_keypoints_heatmaps = 'object_keypoints_heatmaps'
    object_keypoints_heatmaps_on_image = 'object_keypoints_heatmaps_on_image'
    object_keypoints_scores = 'object_keypoints_scores'
    num_objects = 'num_objects'
    object_fnames = 'object_fnames'
    images_with_objects = 'images_with_objects'


class GroundtruthDataFields:
    """
    Names for the groundtruth object data.

    Attributes
    ----------
    groundtruth_object_boxes
        coordinates of the ground truth boxes in the image.
    groundtruth_object_truncation
        truncation information of object
    groundtruth_object_occlusion
        occlusion information of object
    groundtruth_object_observation_angle
        observation angle of object
    groundtruth_object_instance_ids
        instance ids of ground truth objects
    groundtruth_object_classes
        box-level class labels.
    groundtruth_object_instance_masks
        ground truth instance masks.
    groundtruth_object_instance_masks_on_image
        object masks remapped to image
    groundtruth_object_instance_classes
        instance mask-level class labels.
    groundtruth_object_keypoints
        ground truth keypoints.
    groundtruth_object_keypoints_visibilities
        ground truth keypoint visibilities.
    groundtruth_object_keypoints_heatmaps
        contains object keypoint heatmaps
    groundtruth_object_keypoints_heatmaps_on_image
        object keypoint heatmaps remapped to image
    groundtruth_object_keypoints_scores
        scores for object keypoints
    groundtruth_object_label_scores
        groundtruth label scores.
    groundtruth_object_weights
        groundtruth weight factor for bounding boxes.
    num_groundtruth_objects
        number of groundtruth boxes.
    groundtruth_object_fnames
        file name of groundtruth data
    """
    groundtruth_object_boxes = 'groundtruth_object_boxes'
    groundtruth_object_truncation = 'groundtruth_object_truncation'
    groundtruth_object_occlusion = 'groundtruth_object_occlusion'
    groundtruth_object_observation_angle = (
        'groundtruth_object_observation_angle')
    groundtruth_object_instance_ids = 'groundtruth_object_instance_ids'
    groundtruth_object_classes = 'groundtruth_object_classes'
    groundtruth_object_instance_masks = 'groundtruth_object_instance_masks'
    groundtruth_object_instance_masks_on_image = (
        'groundtruth_object_instance_masks_on_image')
    groundtruth_object_instance_boundaries = (
        'groundtruth_object_instance_boundaries')
    groundtruth_object_instance_classes = 'groundtruth_object_instance_classes'
    groundtruth_object_keypoints = 'groundtruth_object_keypoints'
    groundtruth_object_keypoints_visibilities = (
        'groundtruth_object_keypoints_visibilities')
    groundtruth_object_keypoints_heatmaps = (
        'groundtruth_object_keypoints_heatmaps')
    groundtruth_object_keypoints_heatmaps_on_image = (
        'groundtruth_object_keypoints_heatmaps_on_image')
    groundtruth_object_keypoints_scores = 'groundtruth_object_keypoints_scores'
    groundtruth_object_label_scores = 'groundtruth_object_label_scores'
    groundtruth_object_weights = 'groundtruth_object_weights'
    num_groundtruth_objects = 'num_groundtruth_objects'
    groundtruth_object_fnames = 'groundtruth_object_fnames'


class DetectionDataFields:
    """
    Naming conventions for storing the output of the detector.

    Attributes
    ----------
    detection_object_instance_ids
        instance ids of detections
    detection_object_boxes
        coordinates of the detection boxes in the image.
    detection_object_orientations
        predicted orientations of objects
    detection_object_scores
        detection scores for the detection boxes in the image.
    detection_object_classes
        detection-level class labels.
    detection_object_instance_masks
        contains a segmentation mask for each detection box.
    detection_object_instance_masks_on_image
        object masks remapped to image
    detection_object_boundaries
        contains an object boundary for each detection box.
    detection_object_keypoints
        contains detection keypoints for each detection box.
    detection_object_keypoints_heatmaps
        contains object keypoint heatmaps
    detection_object_keypoints_heatmaps_on_image
        object keypoint heatmaps remapped to image
    detection_object_keypoints_scores
        scores for object keypoints
    num_object_detections
        number of detections in the batch.
    detection_object_fnames
        file name with detections
    """
    detection_object_instance_ids = 'detection_object_instance_ids'
    detection_object_boxes = 'detection_object_boxes'
    detection_object_orientations = 'detection_object_orientations'
    detection_object_scores = 'detection_object_scores'
    detection_object_classes = 'detection_object_classes'
    detection_object_instance_masks = 'detection_object_instance_masks'
    detection_object_instance_masks_on_image = (
        'detection_object_instance_masks_on_image')
    detection_object_boundaries = 'detection_object_boundaries'
    detection_object_keypoints = 'detection_object_keypoints'
    detection_object_keypoints_heatmaps = 'detection_object_keypoints_heatmaps'
    detection_object_keypoints_heatmaps_on_image = (
        'detection_object_keypoints_heatmaps_on_image')
    detection_object_keypoints_scores = 'detection_object_keypoints_scores'
    num_object_detections = 'num_object_detections'
    detection_object_fnames = 'detection_object_fnames'
