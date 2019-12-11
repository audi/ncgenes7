# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for faster rcnn object detection architecture
"""
from nucleus7.utils import tf_ops
from object_detection.anchor_generators.grid_anchor_generator import (
    GridAnchorGenerator)
from object_detection.core import standard_fields as od_standard_fields
import tensorflow as tf

from ncgenes7.third_party.object_detection.faster_rcnn_meta_arch import (
    FasterRCNNMetaArch)
from ncgenes7.utils import object_detection_utils


# pylint: disable=too-many-instance-attributes
class FasterRCNNMetaArchFirstStage(FasterRCNNMetaArch):
    """
    Faster RCNN meta architecture that should be used inside of
    FasterRCNNFirstStagePlugin

    For detailed parameter descriptions, see `FasterRCNNMetaArch`
    """

    def __init__(self, *,
                 fuse_sampler=True,
                 first_stage_anchor_generator,
                 first_stage_target_assigner,
                 second_stage_target_assigner,
                 second_stage_sampler,
                 first_stage_box_predictor_arg_scope_fn,
                 first_stage_atrous_rate=1,
                 first_stage_box_predictor_kernel_size=3,
                 first_stage_box_predictor_depth=512,
                 clip_anchors_to_image=False,
                 freeze_batchnorm=False,
                 parallel_iterations=16):
        super().__init__(
            is_training=None,
            num_classes=1,
            image_resizer_fn=None,
            feature_extractor=None,
            number_of_stages=1,
            first_stage_anchor_generator=first_stage_anchor_generator,
            first_stage_target_assigner=first_stage_target_assigner,
            first_stage_atrous_rate=first_stage_atrous_rate,
            first_stage_box_predictor_arg_scope_fn=
            first_stage_box_predictor_arg_scope_fn,
            first_stage_box_predictor_kernel_size=
            first_stage_box_predictor_kernel_size,
            first_stage_box_predictor_depth=first_stage_box_predictor_depth,
            first_stage_minibatch_size=0,
            first_stage_sampler=None,
            first_stage_non_max_suppression_fn=None,
            first_stage_max_proposals=0,
            first_stage_localization_loss_weight=0,
            first_stage_objectness_loss_weight=0,
            crop_and_resize_fn=None,
            initial_crop_size=0,
            maxpool_kernel_size=0,
            maxpool_stride=0,
            second_stage_target_assigner=second_stage_target_assigner,
            second_stage_mask_rcnn_box_predictor=None,
            second_stage_batch_size=0,
            second_stage_sampler=second_stage_sampler,
            second_stage_non_max_suppression_fn=None,
            second_stage_score_conversion_fn=None,
            second_stage_localization_loss_weight=0,
            second_stage_classification_loss_weight=0,
            second_stage_classification_loss=None,
            second_stage_mask_prediction_loss_weight=0,
            hard_example_miner=None,
            parallel_iterations=parallel_iterations,
            add_summaries=None,
            clip_anchors_to_image=clip_anchors_to_image,
            use_static_shapes=False,
            resize_masks=False,
            freeze_batchnorm=freeze_batchnorm)
        self.fuse_sampler = fuse_sampler
        self._max_num_proposals = None
        self._rpn_feature_maps = None

    def set_is_training(self, is_training):
        """
        Set is_training parameter to all parts that use it

        Parameters
        ----------
        is_training
            is training flag
        """
        # pylint: disable=protected-access
        # no other way to assign it
        self._is_training = is_training
        self._first_stage_box_predictor._is_training = is_training
        for heads in self._first_stage_box_predictor._prediction_heads.values():
            for each_head in heads:
                each_head._is_training = is_training

    @property
    def max_num_proposals(self):
        return self._max_num_proposals

    @max_num_proposals.setter
    def max_num_proposals(self, max_num_proposals: tf.Tensor):
        self._max_num_proposals = max_num_proposals
        self._second_stage_batch_size = max_num_proposals

    def set_first_stage_nms_fn(self, first_stage_nms_fn):
        """
        Set first stage nms function

        Parameters
        ----------
        first_stage_nms_fn
            fist stage nms function
        """
        self._first_stage_nms_fn = first_stage_nms_fn

    @property
    def first_stage_anchor_generator(self):
        """
        Returns
        -------
        first_stage_anchor_generator
            first stage anchor generator
        """
        return self._first_stage_anchor_generator

    @first_stage_anchor_generator.setter
    def first_stage_anchor_generator(self, anchor_generator):
        self._first_stage_anchor_generator = anchor_generator

    @property
    def first_stage_box_predictor(self):
        """
        Returns
        -------
        first_stage_box_predictor
            first stage box predictor
        """
        return self._first_stage_box_predictor

    @property
    def first_stage_box_predictor_first_conv(self):
        """
        Returns
        -------
        first_stage_box_predictor_first_conv
            conv layer before first stage box predictor
        """
        return self._first_stage_box_predictor_first_conv

    def predict_first_stage(self, images, rpn_feature_maps):
        """
        Predict first stage

        Parameters
        ----------
        images
            images
        rpn_feature_maps
            rpn feature maps

        Returns
        -------
        proposals
            dict with proposals; see _predict_first_stage
        """
        self._rpn_feature_maps = rpn_feature_maps
        return super()._predict_first_stage(images)

    def _sample_box_classifier_batch(
            self, proposal_boxes, proposal_scores,
            num_proposals,
            groundtruth_boxlists=None,
            groundtruth_classes_with_background_list=None,
            groundtruth_weights_list=None):
        if self.fuse_sampler and self._is_training:
            return super()._sample_box_classifier_batch(
                proposal_boxes, proposal_scores, num_proposals,
                groundtruth_boxlists, groundtruth_classes_with_background_list,
                groundtruth_weights_list)
        return proposal_boxes, proposal_scores, num_proposals

    def _extract_proposal_features(self, preprocessed_inputs):
        return self._rpn_feature_maps, None

    def _format_groundtruth_data(self, true_image_shapes):
        if not self.fuse_sampler:
            return None, None, None, None

        gt_classes = self.groundtruth_lists(
            od_standard_fields.BoxListFields.classes)
        groundtruth_classes_with_background = tf.to_float(
            tf.pad(gt_classes, [[0, 0], [0, 0], [1, 0]], mode='CONSTANT'))

        if self.groundtruth_has_field(od_standard_fields.BoxListFields.weights):
            groundtruth_weights = self.groundtruth_lists(
                od_standard_fields.BoxListFields.weights)
        else:
            groundtruth_weights = tf.ones_like(
                groundtruth_classes_with_background[..., 0])
        groundtruth_masks = None
        groundtruth_boxes_normalized = self.groundtruth_lists(
            od_standard_fields.BoxListFields.boxes)
        groundtruth_boxes_absolute = (
            object_detection_utils.local_to_image_coordinates(
                bboxes=groundtruth_boxes_normalized,
                image_size=tf.transpose(true_image_shapes[0, :2])))
        return (groundtruth_boxes_absolute,
                groundtruth_classes_with_background,
                groundtruth_masks, groundtruth_weights)

# pylint: enable=too-many-instance-attributes


class FasterRCNNMetaArchSecondStage(FasterRCNNMetaArch):
    """
    Faster RCNN meta architecture that should be used inside of
    FasterRCNNSecondStagePlugin

    For detailed parameter descriptions, see `FasterRCNNMetaArch`
    """

    def __init__(self,
                 num_classes,
                 first_stage_target_assigner,
                 second_stage_target_assigner,
                 second_stage_mask_rcnn_box_predictor,
                 second_stage_non_max_suppression_fn,
                 second_stage_score_conversion_fn,
                 parallel_iterations=16):
        # pylint: disable=too-many-arguments
        # all the arguments are needed

        # is needed only for constructor
        first_stage_anchor_generator = GridAnchorGenerator()
        super().__init__(
            is_training=None,
            num_classes=num_classes,
            image_resizer_fn=None,
            feature_extractor=None,
            number_of_stages=2,
            first_stage_anchor_generator=first_stage_anchor_generator,
            first_stage_target_assigner=first_stage_target_assigner,
            first_stage_atrous_rate=None,
            first_stage_box_predictor_arg_scope_fn=None,
            first_stage_box_predictor_kernel_size=None,
            first_stage_box_predictor_depth=None,
            first_stage_minibatch_size=None,
            first_stage_sampler=None,
            first_stage_non_max_suppression_fn=None,
            first_stage_max_proposals=None,
            first_stage_localization_loss_weight=0,
            first_stage_objectness_loss_weight=0,
            crop_and_resize_fn=None,
            initial_crop_size=0,
            maxpool_kernel_size=0,
            maxpool_stride=0,
            second_stage_target_assigner=second_stage_target_assigner,
            second_stage_mask_rcnn_box_predictor=
            second_stage_mask_rcnn_box_predictor,
            second_stage_batch_size=None,
            second_stage_sampler=None,
            second_stage_non_max_suppression_fn=
            second_stage_non_max_suppression_fn,
            second_stage_score_conversion_fn=second_stage_score_conversion_fn,
            second_stage_localization_loss_weight=0,
            second_stage_classification_loss_weight=0,
            second_stage_classification_loss=None,
            second_stage_mask_prediction_loss_weight=None,
            hard_example_miner=None,
            parallel_iterations=parallel_iterations,
            add_summaries=False,
            clip_anchors_to_image=False,
            use_static_shapes=False,
            resize_masks=False,
            freeze_batchnorm=None
        )
        self._max_num_proposals = None

    @property
    def max_num_proposals(self):
        return self._max_num_proposals

    @max_num_proposals.setter
    def max_num_proposals(self, max_num_proposals: tf.Tensor):
        self._max_num_proposals = max_num_proposals
        self._second_stage_batch_size = max_num_proposals

    def set_second_stage_nms_fn(self, second_stage_nms_fn):
        """
        Set second stage nms function

        Parameters
        ----------
        second_stage_nms_fn
            second stage nms function
        """
        self._second_stage_nms_fn = second_stage_nms_fn

    def set_is_training(self, is_training):
        """
        Set is_training parameter to all parts that use it

        Parameters
        ----------
        is_training
            is training flag
        """
        # pylint: disable=protected-access
        # no other way to assign it
        self._is_training = is_training
        self._mask_rcnn_box_predictor._is_training = is_training
        predictor_heads = (
            [self._mask_rcnn_box_predictor._box_prediction_head,
             self._mask_rcnn_box_predictor._class_prediction_head]
            + list(self._mask_rcnn_box_predictor._third_stage_heads.values()))
        for each_head in predictor_heads:
            each_head._is_training = is_training

    def _compute_second_stage_input_feature_maps(self, features_to_crop,
                                                 proposal_boxes_normalized):
        return tf_ops.squash_dims(features_to_crop, [0, 1])

    def _extract_box_classifier_features(self, flattened_feature_maps):
        return flattened_feature_maps
