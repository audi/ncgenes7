# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Implementation of ModelLoss for object detection

Some of the code was taken from
object_detection.meta_architectures.faster_rcnn_meta_arch.
"""
from typing import Dict
from typing import Optional
from typing import Tuple

import nucleus7 as nc7
from object_detection.core import balanced_positive_negative_sampler
from object_detection.core import box_list
from object_detection.core import losses as od_losses
from object_detection.core import target_assigner
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import DetectionDataFields
from ncgenes7.data_fields.object_detection import GroundtruthDataFields
from ncgenes7.utils import object_detection_utils
from ncgenes7.utils.general_utils import broadcast_with_expand_to


class FasterRCNNFirstStageLoss(nc7.model.ModelLoss):
    """
    Loss for Faster RCNN first stage (RPN) using google object detection API

    Parameters
    ----------
    positive_balance_fraction
        fraction of positive examples
        per image for the RPN. The recommended value for Faster RCNN is 0.5.
    minibatch_size
        mini batch size to use to calculate losses
    parallel_iterations
        The number of iterations allowed to run in parallel for calls to
        tf.map_fn

    Attributes
    ----------
    incoming_keys
        * images : original images
        * rpn_box_encodings : predicted proposal box encodings,
          tf.float32, [bs, num_anchors, 4]
        * rpn_objectness_predictions_with_background : objectness predictions,
          tf.float32, [bs, num_anchors, 2]
        * anchors : anchors for predictions, tf.float32, [num_anchors, 4]
        * groundtruth_object_boxes : normalized bounding boxes in format
          ['ymin', 'xmin', 'ymax', 'xmax'], tf.float32, [bs, num_objects, 4]
        * groundtruth_object_weights : (optional) weights for groundtruth boxes

    generated_keys
        * loss_rpn_localization : localization loss
        * loss_rpn_objectness : objectness losss
    """
    incoming_keys = [
        ImageDataFields.images,
        "rpn_box_encodings",
        "rpn_objectness_predictions_with_background",
        "anchors",
        GroundtruthDataFields.groundtruth_object_boxes,
        "_groundtruth_object_weights",
    ]
    generated_keys = [
        "loss_rpn_localization",
        "loss_rpn_objectness",
    ]

    def __init__(self, *,
                 positive_balance_fraction: float = 0.5,
                 minibatch_size: int = 256,
                 parallel_iterations: int = 16,
                 **loss_kwargs):
        super().__init__(**loss_kwargs)
        self.positive_balance_fraction = positive_balance_fraction
        self.minibatch_size = minibatch_size
        self.parallel_iterations = parallel_iterations
        self._proposal_target_assigner = None
        self._sampler = None
        self._localization_loss = None
        self._objectness_loss = None

    def build(self):
        super(FasterRCNNFirstStageLoss, self).build()
        self._proposal_target_assigner = (
            target_assigner.create_target_assigner(
                'FasterRCNN', 'proposal'))
        self._sampler = (
            balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
                positive_fraction=self.positive_balance_fraction,
                is_static=False))
        self._localization_loss = od_losses.WeightedSmoothL1LocalizationLoss()
        self._objectness_loss = od_losses.WeightedSoftmaxClassificationLoss()
        return self

    def process(self, *,
                images: tf.Tensor,
                rpn_box_encodings: tf.Tensor,
                rpn_objectness_predictions_with_background: tf.Tensor,
                anchors: tf.Tensor, groundtruth_object_boxes: tf.Tensor,
                groundtruth_object_weights: Optional[tf.Tensor] = None
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        groundtruth_boxes_absolute = (
            object_detection_utils.local_to_image_coordinates(
                bboxes=groundtruth_object_boxes,
                image_size=tf.shape(images)[1:3]))
        (batch_cls_targets, batch_cls_weights, batch_reg_targets,
         batch_reg_weights) = _batch_assign_targets_dynamic(
             self._proposal_target_assigner,
             anchors,
             groundtruth_boxes_absolute,
             groundtruth_object_weights=groundtruth_object_weights,
             parallel_iterations=self.parallel_iterations)
        batch_cls_weights = tf.reduce_mean(batch_cls_weights, axis=2)
        batch_cls_targets = tf.squeeze(batch_cls_targets, axis=2)

        batch_sampled_indices = _minibatch_subsample(
            self._sampler, self.minibatch_size,
            batch_cls_targets, batch_cls_weights,
            parallel_iterations=self.parallel_iterations)

        losses = self._get_losses(
            rpn_box_encodings,
            rpn_objectness_predictions_with_background,
            batch_cls_targets,
            batch_reg_targets,
            batch_reg_weights,
            batch_sampled_indices)
        return losses

    def _get_losses(self, rpn_box_encodings,
                    rpn_objectness_predictions_with_background,
                    batch_cls_targets, batch_reg_targets, batch_reg_weights,
                    batch_sampled_indices):
        normalizer = tf.maximum(
            tf.reduce_sum(batch_sampled_indices, axis=1), 1.0)
        batch_one_hot_targets = tf.one_hot(
            tf.to_int32(batch_cls_targets), depth=2)
        sampled_reg_indices = tf.multiply(batch_sampled_indices,
                                          batch_reg_weights)
        localization_losses = self._localization_loss(
            rpn_box_encodings, batch_reg_targets, weights=sampled_reg_indices,
            losses_mask=None)
        objectness_losses = self._objectness_loss(
            rpn_objectness_predictions_with_background,
            batch_one_hot_targets,
            weights=tf.expand_dims(batch_sampled_indices, axis=-1),
            losses_mask=None)
        localization_loss = tf.reduce_mean(
            tf.reduce_sum(localization_losses, axis=1) / normalizer)
        objectness_loss = tf.reduce_mean(
            tf.reduce_sum(objectness_losses, axis=1) / normalizer)
        return {"loss_rpn_localization": localization_loss,
                "loss_rpn_objectness": objectness_loss}


class FasterRCNNSecondStageLoss(nc7.model.ModelLoss):
    """
    Loss for Faster RCNN second stage using google object detection API

    Parameters
    ----------
    num_classes
        number of classes
    second_stage_classification_loss_name
        name of loss function for classification, one of
        {WeightedSigmoidClassificationLoss, WeightedSoftmaxClassificationLoss}
    second_stage_localization_loss_name
        name of loss function for classification, e.g.
        WeightedSmoothL1LocalizationLoss
    second_stage_classification_loss_kwargs
        kwargs to pass to classification loss constructor
    second_stage_localization_loss_kwargs
        kwargs to pass to localization loss constructor

    Attributes
    ----------
    incoming_keys
        * images : original images
        * refined_box_encodings : refined box encodings in absolute coordinates;
          [bs, max_num_proposals, num_classes, 4], tf.float32
        * class_predictions_with_background : class logits for each anchor;
          [bs, max_num_proposals, num_classes + 1]; tf.float32
        * proposal_boxes : decoded proposal bounding boxes in absolute
          coordinates from RPN; [bs, max_num_proposals, 4], tf.float32
          decoded proposal bounding boxes in absolute coordinates.
        * num_proposals : number of proposals for each sample in batch; [bs],
          tf.int32
        * groundtruth_object_boxes : normalized bounding boxes in format
          ['ymin', 'xmin', 'ymax', 'xmax'], tf.float32, [bs, num_objects, 4]
        * groundtruth_object_classes : class labels, tf.int64, [bs, num_objects]
        * groundtruth_object_weights : (optional) weights for groundtruth boxes

    generated_keys
        * loss_second_stage_localization : localization loss
        * loss_second_stage_classification : classification loss
    """

    incoming_keys = [
        ImageDataFields.images,
        "refined_box_encodings",
        "class_predictions_with_background",
        "proposal_boxes",
        "num_proposals",
        GroundtruthDataFields.groundtruth_object_boxes,
        GroundtruthDataFields.groundtruth_object_classes,
        "_" + GroundtruthDataFields.groundtruth_object_weights,
    ]
    generated_keys = [
        "loss_second_stage_localization",
        "loss_second_stage_classification",
    ]

    def __init__(self, *,
                 num_classes: int,
                 parallel_iterations=16,
                 second_stage_classification_loss_name=
                 "WeightedSoftmaxClassificationLoss",
                 second_stage_localization_loss_name=
                 "WeightedSmoothL1LocalizationLoss",
                 second_stage_classification_loss_kwargs: Optional[dict] = None,
                 second_stage_localization_loss_kwargs: Optional[dict] = None,
                 **loss_kwargs):
        super().__init__(**loss_kwargs)

        self.second_stage_classification_loss_name = (
            second_stage_classification_loss_name)
        self.second_stage_classification_loss_kwargs = (
            second_stage_classification_loss_kwargs or {})
        self.second_stage_localization_loss_name = (
            second_stage_localization_loss_name)
        self.second_stage_localization_loss_kwargs = (
            second_stage_localization_loss_kwargs or {})
        self.num_classes = num_classes
        self.parallel_iterations = parallel_iterations
        self._detector_target_assigner = None
        self._localization_loss = None
        self._classification_loss = None

    def build(self):
        super().build()
        self._detector_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection')
        self._localization_loss = getattr(
            od_losses, self.second_stage_localization_loss_name
        )(**self.second_stage_localization_loss_kwargs)
        self._classification_loss = getattr(
            od_losses, self.second_stage_classification_loss_name
        )(**self.second_stage_classification_loss_kwargs)
        return self

    def process(self, *,
                images: tf.Tensor,
                refined_box_encodings: tf.Tensor,
                class_predictions_with_background: tf.Tensor,
                proposal_boxes: tf.Tensor,
                num_proposals: tf.Tensor,
                groundtruth_object_boxes: tf.Tensor,
                groundtruth_object_classes: tf.Tensor,
                groundtruth_object_weights: Optional[tf.Tensor] = None
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        (second_stage_cls_losses, second_stage_loc_losses
         ) = self._get_objectwise_losses(
             images, refined_box_encodings, class_predictions_with_background,
             proposal_boxes, num_proposals, groundtruth_object_boxes,
             groundtruth_object_classes, groundtruth_object_weights)

        loc_loss = tf.reduce_sum(second_stage_loc_losses,
                                 name="loss_second_stage_localization")
        cls_loss = tf.reduce_sum(second_stage_cls_losses,
                                 name="loss_second_stage_classification")
        result = {
            "loss_second_stage_localization": loc_loss,
            "loss_second_stage_classification": cls_loss
        }
        return result

    def _get_objectwise_losses(self, images, refined_box_encodings,
                               class_predictions_with_background,
                               proposal_boxes,
                               num_proposals, groundtruth_object_boxes,
                               groundtruth_object_classes,
                               groundtruth_object_weights):
        # pylint: disable=too-many-locals,too-many-arguments
        # cannot be refactored without more complexity
        max_num_proposals = tf.shape(proposal_boxes)[1]
        groundtruth_boxes_absolute = (
            object_detection_utils.local_to_image_coordinates(
                bboxes=groundtruth_object_boxes,
                image_size=tf.shape(images)[1:3]))
        groundtruth_classes_with_background = tf.one_hot(
            indices=groundtruth_object_classes, depth=self.num_classes + 1)
        (batch_cls_targets_with_background, batch_cls_weights,
         batch_reg_targets, batch_reg_weights) = _batch_assign_targets_dynamic(
             self._detector_target_assigner,
             proposal_boxes,
             groundtruth_boxes_absolute,
             num_classes=self.num_classes,
             groundtruth_classes_with_background=
             groundtruth_classes_with_background,
             groundtruth_object_weights=groundtruth_object_weights,
             parallel_iterations=self.parallel_iterations)
        refined_box_encodings_for_loss = (
            self._get_refined_box_encodings_for_loss(
                refined_box_encodings, batch_cls_targets_with_background))
        normalizer = _get_normalizer(tf.shape(images)[0], num_proposals,
                                     max_num_proposals)
        proposals_batch_indicator = _get_proposals_batch_indicator(
            num_proposals, max_num_proposals)
        second_stage_loc_losses = self._localization_loss(
            refined_box_encodings_for_loss,
            batch_reg_targets,
            weights=batch_reg_weights,
            losses_mask=None) / normalizer * proposals_batch_indicator
        second_stage_cls_losses = self._classification_loss(
            class_predictions_with_background,
            batch_cls_targets_with_background,
            weights=batch_cls_weights,
            losses_mask=None) / normalizer * proposals_batch_indicator
        return second_stage_cls_losses, second_stage_loc_losses

    def _get_refined_box_encodings_for_loss(
            self, refined_box_encodings: tf.Tensor,
            batch_cls_targets_with_background: tf.Tensor
    ) -> tf.Tensor:
        if self.num_classes == 1:
            return tf.squeeze(refined_box_encodings, 2)

        encodings_shape = tf.shape(refined_box_encodings)
        encodings_shape_after = tf.concat(
            [encodings_shape[:2], [encodings_shape[-1]]], 0)

        refined_box_encodings_with_background = tf.pad(
            refined_box_encodings, [[0, 0], [0, 0], [1, 0], [0, 0]])
        # this operation is just to make sure we have 0 and 1 in the mask
        batch_cls_targets_with_background_ = tf.one_hot(
            tf.argmax(batch_cls_targets_with_background, -1),
            self.num_classes + 1)
        mask = tf.expand_dims(
            tf.greater(batch_cls_targets_with_background_, 0), -1)
        mask = tf.broadcast_to(
            mask, tf.shape(refined_box_encodings_with_background))
        encodings_masked = tf.boolean_mask(
            refined_box_encodings_with_background, mask)
        return tf.reshape(encodings_masked, encodings_shape_after)


class OHEMLoss(FasterRCNNSecondStageLoss):
    """
    Online Hard Example Miner loss

    This loss uses a `object_detection.core.losses.HardExampleMiner` as
    implementation. For parameters refer to it.

    Parameters
    ----------
    num_hard_examples
        maximum number of hard examples to be
        selected per image (prior to enforcing max negative to positive ratio
        constraint).  If set to None, all examples obtained after NMS are
        considered.
    iou_threshold
        iou threshold for box matching
    loss_type
        which losses should be used for OHEM assignment; one of {loc, cls, both}
    cls_loss_weight
        weight for classification loss for OHEM assignment
    loc_loss_weight
        weight for location loss  for OHEM assignment
    min_negatives_per_image
        minimum number of negative anchors to sample for
        a given image

    References
    ----------
    for further parameter description see
    `object_detection.core.losses.HardExampleMiner`
    """
    # pylint: disable=too-many-arguments
    # not possible to have less arguments without more complexity
    def __init__(self,
                 num_hard_examples=64,
                 iou_threshold=1.0,
                 loss_type='both',
                 cls_loss_weight=0.05,
                 loc_loss_weight=0.06,
                 min_negatives_per_image=0,
                 **loss_kwargs):
        super().__init__(**loss_kwargs)
        self.num_hard_examples = num_hard_examples
        self.iou_threshold = iou_threshold
        self.loss_type = loss_type
        self.cls_loss_weight = cls_loss_weight
        self.loc_loss_weight = loc_loss_weight
        self.min_negatives_per_image = min_negatives_per_image
        self._hard_example_miner = None  # type: od_losses.HardExampleMiner

    def build(self):
        super().build()
        self._hard_example_miner = od_losses.HardExampleMiner(
            num_hard_examples=self.num_hard_examples,
            iou_threshold=self.iou_threshold,
            loss_type=self.loss_type,
            cls_loss_weight=self.cls_loss_weight,
            loc_loss_weight=self.loc_loss_weight,
            max_negatives_per_positive=0,
            min_negatives_per_image=self.min_negatives_per_image,
        )
        return self

    def process(self, *,
                images: tf.Tensor,
                refined_box_encodings: tf.Tensor,
                class_predictions_with_background: tf.Tensor,
                proposal_boxes: tf.Tensor,
                num_proposals: tf.Tensor,
                groundtruth_object_boxes: tf.Tensor,
                groundtruth_object_classes: tf.Tensor,
                groundtruth_object_weights: Optional[tf.Tensor] = None
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        (second_stage_cls_losses, second_stage_loc_losses
         ) = self._get_objectwise_losses(
             images, refined_box_encodings, class_predictions_with_background,
             proposal_boxes, num_proposals, groundtruth_object_boxes,
             groundtruth_object_classes, groundtruth_object_weights)

        cls_loss, loc_loss = self._apply_ohem(
            second_stage_cls_losses, second_stage_loc_losses,
            proposal_boxes)

        result = {
            "loss_second_stage_localization": loc_loss,
            "loss_second_stage_classification": cls_loss,
        }
        return result

    def _apply_ohem(self, second_stage_cls_losses: tf.Tensor,
                    second_stage_loc_losses: tf.Tensor,
                    proposal_boxes: tf.Tensor
                    ) -> Tuple[tf.Tensor, tf.Tensor]:
        def _ohem(sample_inputs):
            (sample_loc_losses, sample_cls_losses, sample_proposal_boxes
             ) = sample_inputs
            location_losses = tf.expand_dims(sample_loc_losses, 0)
            cls_losses = tf.expand_dims(sample_cls_losses, 0)
            decoded_boxlist_list = [box_list.BoxList(sample_proposal_boxes)]
            (sample_loc_loss_ohem, sample_cls_loss_ohem
             ) = self._hard_example_miner(
                 location_losses=location_losses,
                 cls_losses=cls_losses,
                 decoded_boxlist_list=decoded_boxlist_list)
            return sample_loc_loss_ohem, sample_cls_loss_ohem

        inputs = [second_stage_loc_losses, second_stage_cls_losses,
                  proposal_boxes]
        dtypes = (tf.float32, tf.float32)
        batch_loc_loss_ohem, batch_cls_loss_ohem = tf.map_fn(
            _ohem, inputs,
            dtype=dtypes,
            parallel_iterations=self.parallel_iterations)
        loc_loss_ohem = tf.reduce_sum(
            batch_loc_loss_ohem,
            name="loss_ohem_second_stage_localization")
        cls_loss_ohem = tf.reduce_sum(
            batch_cls_loss_ohem,
            name="loss_ohem_second_stage_classification")
        return cls_loss_ohem, loc_loss_ohem


class KeypointsToHeatmapsLoss(nc7.model.ModelLoss):
    """
    Loss from keypoints heatmaps and raw groundtruth keypoints

    Parameters
    ----------
    num_keypoints
        number of keypoints
    gt_keypoints_kernel_size
        gaussian kernel size for groundtruth keypoints heatmaps
    heatmaps_loss_name
        name of the heatmaps loss to use, must be inside of
        'object_detection.losses' e.g. WeightedL2LocalizationLoss or
        WeightedSigmoidClassificationLoss
    heatmaps_loss_kwargs
        kwargs that will be passed to the heatmaps loss constructor

    Attributes
    ----------
    incoming_keys
        * detection_object_boxes : detection boxes in normalized coordinates,
          tf.float32, [bs, num_detections, 4]
        * detection_object_keypoints_heatmaps : predicted heatmaps (or logits)
          for keypoints; tf.float32,
          [bs, num_detections, map_width, map_height, num_keypoints]
        * num_object_detections : number of detections in each sample; tf.int32,
          [bs]
        * groundtruth_object_boxes : groundtruth boxes in normalized
          coordinates, tf.float32, [bs, num_gt, 4]
        * groundtruth_object_keypoints : groundtruth keypoints with normalized
          to image coordinates in format [y, x]; tf.float32,
          [bs, num_gt, num_keypoints, 2]
        * groundtruth_object_weights : (optional) weights for groundtruth boxes;
          [bs, num_gt], tf.float32

    generated_keys
        * loss_keypoints_heatmaps : keypoints loss
    """

    incoming_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_keypoints_heatmaps,
        DetectionDataFields.num_object_detections,
        GroundtruthDataFields.groundtruth_object_boxes,
        GroundtruthDataFields.groundtruth_object_keypoints,
        "_" + GroundtruthDataFields.groundtruth_object_weights,
    ]
    generated_keys = [
        "loss_keypoints_heatmaps"
    ]

    def __init__(self, *,
                 num_keypoints: int,
                 gt_keypoints_kernel_size: int = 3,
                 heatmaps_loss_name: str = "WeightedL2LocalizationLoss",
                 heatmaps_loss_kwargs: Optional[dict] = None,
                 **loss_kwargs):
        super().__init__(**loss_kwargs)
        self.heatmaps_loss_name = heatmaps_loss_name
        self.num_keypoints = num_keypoints
        self.gt_keypoints_kernel_size = gt_keypoints_kernel_size
        self.heatmaps_loss_kwargs = heatmaps_loss_kwargs or {}
        self._loss_fn = None
        self._detector_target_assigner = None

    def build(self):
        super().build()
        self._detector_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection')
        self._loss_fn = getattr(od_losses, self.heatmaps_loss_name)(
            **self.heatmaps_loss_kwargs)
        return self

    def process(self, *,
                detection_object_boxes: tf.Tensor,
                num_object_detections: tf.Tensor,
                detection_object_keypoints_heatmaps: tf.Tensor,
                groundtruth_object_boxes: tf.Tensor,
                groundtruth_object_keypoints: tf.Tensor,
                groundtruth_object_weights: Optional[tf.Tensor] = None
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ,too-many-locals
        # parent save method has more generic signature
        detection_object_boxes = tf.stop_gradient(detection_object_boxes)
        heatmaps_shape = tf.shape(detection_object_keypoints_heatmaps)[2:4]
        groundtruth_object_keypoints = tf.maximum(
            groundtruth_object_keypoints, 0)
        (target_keypoints_encoded, keypoints_masks, target_weights
         ) = self._match_groundtruth_keypoints(
             detection_object_boxes,
             groundtruth_object_boxes,
             groundtruth_object_keypoints,
             groundtruth_object_weights)
        target_keypoints_heatmaps = self._get_target_keypoints_heatmaps(
            target_keypoints_encoded, keypoints_masks, heatmaps_shape)
        max_num_detections = tf.shape(detection_object_boxes)[1]
        batch_indicator = _get_proposals_batch_indicator(
            num_object_detections, max_num_detections)
        normalizer = tf.maximum(
            tf.reduce_sum(
                batch_indicator * target_weights
                * tf.reduce_prod(tf.cast(heatmaps_shape, tf.float32))),
            1.0
        )

        detection_object_keypoints_heatmaps_r = tf.reshape(
            detection_object_keypoints_heatmaps,
            [tf.shape(detection_object_keypoints_heatmaps)[0], -1,
             tf.shape(detection_object_keypoints_heatmaps)[-1]]
        )
        target_keypoints_heatmaps_r = tf.reshape(
            target_keypoints_heatmaps,
            [tf.shape(target_keypoints_heatmaps)[0], -1,
             tf.shape(target_keypoints_heatmaps)[-1]]
        )
        weights = tf.reshape(
            broadcast_with_expand_to(target_weights,
                                     target_keypoints_heatmaps[..., 0]),
            [tf.shape(target_weights)[0], -1]
        )
        batch_indicator_r = tf.reshape(
            broadcast_with_expand_to(batch_indicator,
                                     target_keypoints_heatmaps[..., 0]),
            [tf.shape(target_weights)[0], -1])
        if "Classification" in self.heatmaps_loss_name:
            weights = weights[..., tf.newaxis]
            batch_indicator_r = batch_indicator_r[..., tf.newaxis]

        heatmaps_loss_samples = self._loss_fn(
            detection_object_keypoints_heatmaps_r,
            target_keypoints_heatmaps_r,
            weights=weights,
        ) / normalizer

        loss = tf.reduce_sum(heatmaps_loss_samples * batch_indicator_r,
                             name="loss_keypoints_heatmaps")

        return {
            "loss_keypoints_heatmaps": loss
        }

    def _get_target_keypoints_heatmaps(self, target_keypoints_encoded,
                                       keypoints_masks,
                                       heatmaps_shape):
        target_keypoints_encoded_sq = tf.reshape(
            target_keypoints_encoded, [-1, 2])
        keypoints_masks_sq = tf.reshape(keypoints_masks, [-1])
        target_keypoints_heatmaps_sq = (
            object_detection_utils.create_keypoints_heatmaps(
                target_keypoints_encoded_sq, heatmaps_shape,
                keypoints_masks=keypoints_masks_sq,
                keypoints_kernel_size=self.gt_keypoints_kernel_size,
            ))
        target_keypoints_heatmaps_shape = tf.concat(
            [tf.shape(target_keypoints_encoded)[:-1],
             tf.shape(target_keypoints_heatmaps_sq)[1:]], 0)
        target_keypoints_heatmaps = tf.reshape(target_keypoints_heatmaps_sq,
                                               target_keypoints_heatmaps_shape)
        target_keypoints_heatmaps = tf.transpose(target_keypoints_heatmaps,
                                                 [0, 1, 3, 4, 2])
        return target_keypoints_heatmaps

    def _match_groundtruth_keypoints(self, detection_object_boxes,
                                     groundtruth_object_boxes,
                                     groundtruth_object_keypoints,
                                     groundtruth_object_weights):
        # pylint: disable=too-many-locals
        # is not possible to extract further
        groundtruth_object_keypoints_flat = tf.reshape(
            groundtruth_object_keypoints,
            tf.concat([tf.shape(groundtruth_object_keypoints)[:2],
                       [self.num_keypoints * 2]], 0))
        unmatched_mask_label = tf.zeros(
            [self.num_keypoints * 2], dtype=tf.float32)
        (keypoints_with_boxes_targets, _, _,
         batch_mask_target_weights
         ) = _batch_assign_targets_dynamic(
             self._detector_target_assigner,
             anchors=detection_object_boxes,
             groundtruth_object_boxes=groundtruth_object_boxes,
             groundtruth_classes_with_background=
             groundtruth_object_keypoints_flat,
             unmatched_class_label=unmatched_mask_label,
             groundtruth_object_weights=groundtruth_object_weights)
        batch_with_obj_dims = tf.shape(detection_object_boxes)[:2]
        keypoints_targets = tf.reshape(
            keypoints_with_boxes_targets,
            tf.concat([batch_with_obj_dims, [self.num_keypoints, 2]], 0)
        )
        keypoints_encoded, keypoints_masks = self._encode_keypoints_to_boxes(
            detection_object_boxes, keypoints_targets)
        return keypoints_encoded, keypoints_masks, batch_mask_target_weights

    @staticmethod
    def _encode_keypoints_to_boxes(detection_object_boxes, keypoints_targets):
        keypoints_in_boxes, valid_keypoints_mask = (
            object_detection_utils.crop_keypoints_to_boxes(
                keypoints_targets, detection_object_boxes))
        keypoints_encoded = object_detection_utils.encode_keypoints_to_boxes(
            keypoints_in_boxes, detection_object_boxes)
        object_masks = tf.reduce_any(detection_object_boxes > 0, -1)
        keypoints_masks = tf.logical_and(object_masks[..., tf.newaxis],
                                         valid_keypoints_mask)
        return keypoints_encoded, keypoints_masks


class ClassificationMatchLoss(nc7.model.ModelLoss):
    """
    Classification loss which assigns targets by bounding boxes

    Parameters
    ----------
    loss_name
        name of the loss from object_detection losses to use
    loss_kwargs
        kwargs that will be passed to the loss constructor

    Attributes
    ----------
    incoming_keys
        * class_predictions_with_background : a 3-D tensor with shape
          [bs, max_num_detections, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        * detection_object_boxes : detection object boxes in normalized
          to image coordinates in format ['ymin', 'xmin', 'ymax', 'xmax'],
          tf.float32, [bs, max_num_detections, 4]
        * num_object_detections : number of valid detections,
          tf.int32, [bs]
        * groundtruth_object_boxes : normalized bounding boxes in format
          ['ymin', 'xmin', 'ymax', 'xmax'], tf.float32, [bs, num_gt_objects, 4]
        * groundtruth_object_classes : class labels, tf.int64,
          [bs, num_gt_objects]
        * groundtruth_object_weights : (optional) weights for groundtruth boxes

    generated_keys
        * loss_classification_match : classification loss
    """
    incoming_keys = [
        "class_predictions_with_background",
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.num_object_detections,
        GroundtruthDataFields.groundtruth_object_boxes,
        GroundtruthDataFields.groundtruth_object_classes,
        "_" + GroundtruthDataFields.groundtruth_object_weights,
    ]
    generated_keys = [
        "loss_classification_match"
    ]

    def __init__(self, *,
                 loss_name="WeightedSoftmaxClassificationLoss",
                 loss_kwargs: Optional[dict] = None,
                 **loss_nucleotide_kwargs):
        super().__init__(**loss_nucleotide_kwargs)
        self.loss_name = loss_name
        self.loss_kwargs = loss_kwargs or {}
        self._detector_target_assigner = None
        self._loss_fn = None

    def build(self):
        super().build()
        self._detector_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection')
        self._loss_fn = getattr(od_losses, self.loss_name)(**self.loss_kwargs)
        return self

    def process(self,
                class_predictions_with_background: tf.Tensor,
                num_object_detections: tf.Tensor,
                detection_object_boxes: tf.Tensor,
                groundtruth_object_boxes: tf.Tensor,
                groundtruth_object_classes: tf.Tensor,
                groundtruth_object_weights: Optional[tf.Tensor] = None
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ,too-many-locals
        # parent save method has more generic signature
        detection_object_boxes = tf.stop_gradient(detection_object_boxes)
        num_classes_with_bg = (
            class_predictions_with_background.shape.as_list()[-1])
        groundtruth_classes_with_background = tf.one_hot(
            indices=groundtruth_object_classes, depth=num_classes_with_bg,
            dtype=tf.float32)
        (class_targets, class_targets_weights, _, _
         ) = _batch_assign_targets_dynamic(
             self._detector_target_assigner,
             anchors=detection_object_boxes,
             groundtruth_object_boxes=groundtruth_object_boxes,
             groundtruth_classes_with_background=
             groundtruth_classes_with_background,
             num_classes=num_classes_with_bg - 1,
             groundtruth_object_weights=groundtruth_object_weights)

        max_num_proposals = tf.shape(detection_object_boxes)[1]
        normalizer = _get_normalizer(tf.shape(detection_object_boxes)[0],
                                     num_object_detections,
                                     max_num_proposals)
        proposals_batch_indicator = _get_proposals_batch_indicator(
            num_object_detections, max_num_proposals)
        second_stage_cls_losses = self._loss_fn(
            class_predictions_with_background,
            class_targets,
            weights=class_targets_weights,
            losses_mask=None)
        normalizer = broadcast_with_expand_to(
            normalizer, second_stage_cls_losses)
        proposals_batch_indicator = broadcast_with_expand_to(
            proposals_batch_indicator, second_stage_cls_losses)
        second_stage_cls_losses = second_stage_cls_losses / normalizer
        second_stage_cls_loss = tf.reduce_sum(
            second_stage_cls_losses * proposals_batch_indicator,
            name="loss_classification_match")
        return {
            "loss_classification_match": second_stage_cls_loss
        }


def _minibatch_subsample(sampler, minibatch_size,
                         batch_cls_targets, batch_cls_weights,
                         parallel_iterations=16):
    def _minibatch_subsample_fn(inputs):
        cls_targets, cls_weights = inputs
        return sampler.subsample(
            tf.cast(cls_weights, tf.bool),
            minibatch_size, tf.cast(cls_targets, tf.bool))

    batch_sampled_indices = tf.cast(
        tf.map_fn(_minibatch_subsample_fn,
                  [batch_cls_targets, batch_cls_weights],
                  dtype=tf.bool,
                  parallel_iterations=parallel_iterations,
                  back_prop=True),
        tf.float32)
    return batch_sampled_indices


def _batch_assign_targets_dynamic(
        proposal_target_assigner,
        anchors,
        groundtruth_object_boxes,
        groundtruth_classes_with_background: Optional[tf.Tensor] = None,
        groundtruth_object_weights: Optional[tf.Tensor] = None,
        num_classes: Optional[int] = None,
        parallel_iterations=16,
        unmatched_class_label: Optional[tf.Tensor] = None):
    # pylint: disable=too-many-arguments
    # not possible to have less arguments without more complexity
    if (groundtruth_classes_with_background is not None
            and unmatched_class_label is None):
        unmatched_class_label = tf.constant(
            [1] + num_classes * [0], dtype=tf.float32)

    def _assign_single(inputs_):
        inputs_ = list(inputs_)
        if len(anchors.shape) == 3:
            anchors_ = inputs_.pop()
        else:
            anchors_ = anchors
        if groundtruth_object_weights is not None:
            groundtruth_object_weights_single = inputs_.pop()
        else:
            groundtruth_object_weights_single = None

        if groundtruth_classes_with_background is not None:
            groundtruth_classes_with_background_single = inputs_.pop()
        else:
            groundtruth_classes_with_background_single = None
        groundtruth_object_boxes_single = inputs_.pop()
        (cls_targets, cls_weights, reg_targets, reg_weights, _
         ) = target_assigner.batch_assign_targets(
             target_assigner=proposal_target_assigner,
             anchors_batch=box_list.BoxList(anchors_),
             gt_box_batch=[box_list.BoxList(groundtruth_object_boxes_single)],
             gt_class_targets_batch=
             [groundtruth_classes_with_background_single],
             unmatched_class_label=unmatched_class_label,
             gt_weights_batch=[groundtruth_object_weights_single])

        return (cls_targets[0], cls_weights[0], reg_targets[0],
                reg_weights[0])

    inputs = [groundtruth_object_boxes]
    if groundtruth_classes_with_background is not None:
        inputs.append(groundtruth_classes_with_background)
    if groundtruth_object_weights is not None:
        inputs.append(groundtruth_object_weights)
    if len(anchors.shape) == 3:
        inputs.append(anchors)

    (batch_cls_targets, batch_cls_weights, batch_reg_targets,
     batch_reg_weights) = tf.map_fn(
         _assign_single, inputs,
         dtype=(tf.float32, tf.float32, tf.float32, tf.float32),
         back_prop=True,
         parallel_iterations=parallel_iterations)
    return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
            batch_reg_weights)


def _get_normalizer(batch_size, num_proposals, max_num_proposals):
    num_proposals_or_one = tf.cast(
        tf.expand_dims(tf.maximum(
            num_proposals, tf.ones_like(num_proposals)), 1), tf.float32)
    normalizer = tf.tile(num_proposals_or_one,
                         [1, max_num_proposals]
                         ) * tf.cast(batch_size, tf.float32)
    return normalizer


def _get_proposals_batch_indicator(num_proposals, max_num_proposals):
    proposals_batch_indicator_mask = tf.sequence_mask(
        num_proposals, max_num_proposals)
    proposals_batch_indicator = tf.cast(proposals_batch_indicator_mask,
                                        tf.float32)
    return proposals_batch_indicator
