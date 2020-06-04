# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Faster-RCNN plugins adapted to use object_detection API from google
"""
from functools import partial
from typing import Dict
from typing import Optional

import nucleus7 as nc7
from nucleus7.utils import tf_ops
from object_detection.anchor_generators.grid_anchor_generator import (
    GridAnchorGenerator)
from object_detection.builders import box_predictor_builder
from object_detection.core import standard_fields as od_standard_fields
from object_detection.core import target_assigner
from object_detection.utils import ops as object_detection_ops
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import DetectionDataFields
from ncgenes7.data_fields.object_detection import GroundtruthDataFields
from ncgenes7.third_party.object_detection.balanced_positive_negative_sampler \
    import BalancedPositiveNegativeSampler
from ncgenes7.utils.faster_rcnn_utils import FasterRCNNMetaArchFirstStage
from ncgenes7.utils.faster_rcnn_utils import FasterRCNNMetaArchSecondStage
from ncgenes7.utils.object_detection_utils import (
    GridAnchorGeneratorRelative)
from ncgenes7.utils.object_detection_utils import KerasLayerHyperparamsFromData
from ncgenes7.utils.object_detection_utils import (
    batch_multiclass_non_max_suppression)
from ncgenes7.utils.object_detection_utils import get_true_image_shapes
from ncgenes7.utils.object_detection_utils import local_to_image_coordinates
from ncgenes7.utils.object_detection_utils import normalize_bbox


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class FasterRCNNFirstStagePlugin(nc7.model.ModelPlugin):
    """
    Plugin for first stage of Faster RCNN inherited from object detection API.

    Parameters
    ----------
    anchor_generator_config
        configuration of anchor generator; see :obj:`GridAnchorGenerator` for
        further notes on fields; if you include 'relative=True', then anchors
        base size will be set to image size and stride to the ratio between
        feature maps of RPN and image size
    rpn_feature_extractor_config
        config of the rpn convolution feature extractor; refer to
        :obj:`tf.keras.layers.Conv2d` for possible fields
    fuse_sampler
        if the sampling for second stage should be fused in this plugin
    max_num_proposals
        max number of proposals; in contrast to original model, this will be
        the same value for all modules; it will be add as default placeholder,
        so can be modified during inference
    max_num_proposals_train
        same as max_num_proposals but will be used for training; if not
        specified, then is equal to max_num_proposals; if fuse_sampler=False,
        then it is a number of proposals to use inside of the nms function
    nms_iou_threshold
        iou threshold for nms on the predicted boxes; will be added as a
        default placeholder, so can be changed during inference
    nms_score_threshold
        score threshold for nms on the predicted boxes; will be added as a
        default placeholder, so can be changed during inference
    clip_anchors_to_image
        if the anchors should be clipped to image
    freeze_batchnorm
        if the batch normalization parameters should be frozen during training;
        is better to set to True for small batch sizes
    batch_norm_params
        batch normalization parameters, like 'momentum', 'center', 'scale' and
        'epsilon'
    parallel_iterations
        number of parallel iteration to use for tf.map_fn

    Attributes
    ----------
    incoming_keys
        * images : original images
        * image_sizes : (optional) image sizes with shape [bs, 3] and every
          instance has [height, width, num_channels] inside; if not provided,
          will be inferred from images itself
        * feature_maps : features from rpn extractor, tf.float32
        * groundtruth_object_boxes : (optional) groundtruth bounding boxes
          normalized to image shape  and in format [ymin, xmin, ymax, xmax];
          only needed if fuse_sampler = True
        * groundtruth_object_classes : class labels, tf.int64,
          [bs, num_gt_objects];
          only needed if fuse_sampler = True
        * groundtruth_object_weights : (optional) weights for groundtruth
          boxes; only used if fuse_sampler = True

    generated_keys
        * detection_object_boxes : detection boxes in normalized coordinates;
          tf.float32, [bs, num_detections, 4]
        * detection_objects_scores : detection scores;
          tf.float32, [bs, num_detections, 4]
        * num_object_detections : number of detections in each sample;
          tf.int32, [bs]
        * anchors : anchors for predictions, tf.float32, [num_anchors, 4] in
          absolute coordinates
        * rpn_box_encodings : predicted proposal box encodings,
          tf.float32, [bs, num_anchors, 4]
        * rpn_objectness_predictions_with_background : objectness predictions,
          tf.float32, [bs, num_anchors, 2]
    """
    incoming_keys = [
        ImageDataFields.images,
        "_" + ImageDataFields.image_sizes,
        "feature_maps",
        "_" + GroundtruthDataFields.groundtruth_object_boxes,
        "_" + GroundtruthDataFields.groundtruth_object_classes,
        "_" + GroundtruthDataFields.groundtruth_object_weights,
    ]
    generated_keys = [
        "rpn_box_encodings",
        "rpn_objectness_predictions_with_background",
        "anchors",
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
    ]

    def __init__(self, *,
                 fuse_sampler: bool = True,
                 anchor_generator_config: Optional[dict] = None,
                 rpn_feature_extractor_config: Optional[dict] = None,
                 max_num_proposals: int = 64,
                 max_num_proposals_train: Optional[int] = None,
                 nms_iou_threshold=0.7,
                 nms_score_threshold=0.0,
                 clip_anchors_to_image=False,
                 freeze_batchnorm=False,
                 batch_norm_params: Optional[dict] = None,
                 second_stage_balance_fraction=0.25,
                 parallel_iterations=16,
                 **plugin_kwargs):
        # pylint: disable=too-many-locals
        allow_mixed_precision = plugin_kwargs.pop('allow_mixed_precision', None)
        if allow_mixed_precision:
            raise ValueError('allow_mixed_precision is not allowed for '
                             '{}! Remove it from config'.format(self.name))
        super().__init__(allow_mixed_precision=False, **plugin_kwargs)
        self.fuse_sampler = fuse_sampler
        self._use_relative_anchor_generator = (anchor_generator_config
                                               or {}).pop("relative", False)
        self._rescale_aspect_ratios = (
            anchor_generator_config
            or {}).pop("rescale_aspect_ratios", False)
        self.anchor_generator_config = anchor_generator_config
        self.rpn_feature_extractor_config = rpn_feature_extractor_config
        self.max_num_proposals = max_num_proposals
        self.max_num_proposals_train = (max_num_proposals_train
                                        or self.max_num_proposals)
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_threshold = nms_score_threshold
        self.clip_anchors_to_image = clip_anchors_to_image
        self.freeze_batchnorm = freeze_batchnorm
        self.batch_norm_params = batch_norm_params
        self.parallel_iterations = parallel_iterations
        self.second_stage_balance_fraction = second_stage_balance_fraction
        self._faster_rcnn_meta = None  # type: FasterRCNNMetaArchFirstStage

    @property
    def defaults(self):
        defaults = super().defaults
        defaults["anchor_generator_config"] = {
            "scales": [0.25, 0.5, 1.0, 2.0],
            "aspect_ratios": [0.5, 1, 2.0]
        }
        defaults["rpn_feature_extractor_config"] = {
            "filters": 512,
            "kernel_size": 3,
            "strides": 1,
            "atrous_rate": 1,
        }
        return defaults

    def build(self):
        super(FasterRCNNFirstStagePlugin, self).build()
        first_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'proposal')
        second_stage_target_assigner = (
            target_assigner.create_target_assigner('FasterRCNN', 'detection')
            if self.fuse_sampler else None)
        first_stage_anchor_generator = self._get_anchor_generator()
        first_stage_box_predictor_arg_scope_fn = KerasLayerHyperparamsFromData(
            activation=self.activation, initializer=self.initializer,
            batch_norm_params=self.batch_norm_params
        )
        second_stage_sampler = (
            BalancedPositiveNegativeSampler(
                positive_fraction=self.second_stage_balance_fraction)
            if self.fuse_sampler else None)
        self._faster_rcnn_meta = FasterRCNNMetaArchFirstStage(
            fuse_sampler=self.fuse_sampler,
            first_stage_anchor_generator=first_stage_anchor_generator,
            first_stage_target_assigner=first_stage_target_assigner,
            second_stage_target_assigner=second_stage_target_assigner,
            second_stage_sampler=second_stage_sampler,
            first_stage_box_predictor_arg_scope_fn=
            first_stage_box_predictor_arg_scope_fn,
            first_stage_atrous_rate=
            self.rpn_feature_extractor_config.get("atrous_rate", 1),
            first_stage_box_predictor_kernel_size=
            self.rpn_feature_extractor_config.get("kernel_size", 3),
            first_stage_box_predictor_depth=
            self.rpn_feature_extractor_config.get("filters", 3),
            clip_anchors_to_image=self.clip_anchors_to_image,
            freeze_batchnorm=self.freeze_batchnorm,
            parallel_iterations=self.parallel_iterations)

        self.add_keras_layer(self._faster_rcnn_meta.first_stage_box_predictor)
        self.add_keras_layer(
            self._faster_rcnn_meta.first_stage_box_predictor_first_conv)
        return self

    def reset_tf_graph(self):
        super().reset_tf_graph()
        self._faster_rcnn_meta.first_stage_anchor_generator = (
            self._get_anchor_generator())

    def predict(self, *,
                images: tf.Tensor,
                feature_maps: tf.Tensor,
                image_sizes: Optional[tf.Tensor] = None,
                groundtruth_object_boxes: Optional[tf.Tensor] = None,
                groundtruth_object_classes: Optional[tf.Tensor] = None,
                groundtruth_object_weights: Optional[tf.Tensor] = None):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        # pylint: disable=too-many-locals
        # cannot reduce number of locals without more code complexity
        if not self.fuse_sampler:
            if groundtruth_object_boxes is not None:
                raise ValueError(
                    "{}: groundtruth is not used because fuse_sampler=False"
                    "".format(self.name))
        self._faster_rcnn_meta.set_is_training(self.is_training)
        max_num_proposals = self.add_default_placeholder(
            self.max_num_proposals, "max_num_proposals")
        max_num_proposals_for_mode = (
            tf.convert_to_tensor(self.max_num_proposals_train, tf.int32)
            if self.mode == "train" else max_num_proposals)

        nms_iou_threshold = self.add_default_placeholder(
            self.nms_iou_threshold, "nms_iou_threshold")
        nms_score_threshold = self.add_default_placeholder(
            self.nms_score_threshold, "nms_score_threshold")
        if self._use_relative_anchor_generator:
            image_size = tf.shape(images)[1:3]
            feature_maps_size = tf.shape(feature_maps)[1:3]
            self._faster_rcnn_meta.first_stage_anchor_generator.rescale(
                image_size, feature_maps_size,
                rescale_aspect_ratios=self._rescale_aspect_ratios)

        self._faster_rcnn_meta.max_num_proposals = max_num_proposals_for_mode
        self._faster_rcnn_meta.set_first_stage_nms_fn(
            partial(_first_stage_nms_fn,
                    iou_threshold=nms_iou_threshold,
                    score_threshold=nms_score_threshold,
                    max_total_size=max_num_proposals,
                    max_size_per_class=max_num_proposals,
                    parallel_iterations=self.parallel_iterations))
        true_image_shapes = (image_sizes if image_sizes is not None
                             else get_true_image_shapes(images))
        prediction_dict = self._faster_rcnn_meta.predict_first_stage(
            images, feature_maps)

        # pylint: disable=protected-access
        if (self.is_training
                and self.fuse_sampler
                and groundtruth_object_boxes is not None):
            self._provide_groundtruth_to_meta(groundtruth_object_boxes,
                                              groundtruth_object_classes,
                                              groundtruth_object_weights)
        # pylint: enable=protected-access
        postprocessed_dict = self._faster_rcnn_meta.postprocess(
            prediction_dict, true_image_shapes)

        result = {
            "anchors": prediction_dict["anchors"],
            "rpn_box_encodings": prediction_dict["rpn_box_encodings"],
            "rpn_objectness_predictions_with_background":
                prediction_dict["rpn_objectness_predictions_with_background"],
            DetectionDataFields.detection_object_boxes:
                postprocessed_dict["detection_boxes"],
            DetectionDataFields.detection_object_scores:
                postprocessed_dict["detection_scores"],
            DetectionDataFields.num_object_detections:
                tf.cast(postprocessed_dict["num_detections"], tf.int32),
        }
        return result

    def _provide_groundtruth_to_meta(self, groundtruth_object_boxes,
                                     groundtruth_object_classes,
                                     groundtruth_object_weights):
        groundtruth_object_classes = tf.clip_by_value(
            groundtruth_object_classes, 0, 1)
        groundtruth_object_classes_one_hot = tf.one_hot(
            groundtruth_object_classes, 2)[..., 1:]
        self._faster_rcnn_meta.provide_groundtruth(
            groundtruth_object_boxes, groundtruth_object_classes_one_hot)
        if groundtruth_object_weights is not None:
            # pylint: disable=protected-access
            # is needed since in original method comparison is done as
            # if not tensor, which is not allowed
            self._faster_rcnn_meta._groundtruth_lists[
                od_standard_fields.BoxListFields.weights] = (
                    groundtruth_object_weights)

    def _get_anchor_generator(self):
        return GridAnchorGeneratorRelative(**self.anchor_generator_config)


class ProposalsSampler(nc7.model.ModelPlugin):
    """
    Plugin to sample the detections according to overlap with GT.

    Performs sampling only during training and for evaluation / inference
    passes results as is (except if sample_always = True).

    Parameters
    ----------
    sample_always
        if sampling should be performed also during inference
    sample_minibatch_size
        number of samples to generate
    positive_balance_fraction
        fraction of objects to sample (samples with an object inside)
    parallel_iterations
        number of parallel iteration to use for tf.map_fn

    Attributes
    ----------
    incoming_keys
        * images : (optional) original images
        * image_sizes : (optional) image sizes with shape [bs, 3] and every
          instance has [height, width, num_channels] inside; if not provided,
          will be inferred from images itself
        * detection_object_boxes : detection boxes in normalized coordinates;
          tf.float32, [bs, num_detections, 4]
        * detection_objects_scores : detection scores;
          tf.float32, [bs, num_detections, 4]
        * num_object_detections : number of detections in each sample;
          tf.int32, [bs]
        * groundtruth_object_boxes : groundtruth bounding boxes
          normalized to image shape  and in format [ymin, xmin, ymax, xmax];
        * groundtruth_object_classes : tensors of shape
          [bs, num_boxes] containing the class labels (indexes, not one hot);
          object classes start from 1 (0 class is a background);
        * groundtruth_object_weights : (optional) tensors to use for sampling
          of the proposals for second stage

    generated_keys
        * detection_object_boxes : detection boxes in normalized coordinates;
          tf.float32, [bs, num_detections, 4]
        * detection_objects_scores : detection scores;
          tf.float32, [bs, num_detections, 4]
        * num_object_detections : number of detections in each sample;
          tf.int32, [bs]
    """
    incoming_keys = [
        "_" + ImageDataFields.images,
        "_" + ImageDataFields.image_sizes,
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        GroundtruthDataFields.groundtruth_object_boxes,
        GroundtruthDataFields.groundtruth_object_classes,
        "_" + GroundtruthDataFields.groundtruth_object_weights,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
    ]

    def __init__(self, *,
                 sample_always: bool = False,
                 sample_minibatch_size: int = 64,
                 positive_balance_fraction: float = 0.25,
                 parallel_iterations=16,
                 **plugin_kwargs):
        super().__init__(**plugin_kwargs)
        allow_mixed_precision = plugin_kwargs.pop('allow_mixed_precision', None)
        if allow_mixed_precision:
            raise ValueError('allow_mixed_precision is not allowed for '
                             '{}! Remove it from config'.format(self.name))
        self.sample_always = sample_always
        self.positive_balance_fraction = positive_balance_fraction
        self.sample_minibatch_size = sample_minibatch_size
        self.parallel_iterations = parallel_iterations
        self._faster_rcnn_meta = None  # type: FasterRCNNMetaArchFirstStage

    def build(self):
        super().build()
        second_stage_sampler = BalancedPositiveNegativeSampler(
            positive_fraction=self.positive_balance_fraction)
        first_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'proposal')
        second_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection')
        self._faster_rcnn_meta = FasterRCNNMetaArchFirstStage(
            fuse_sampler=True,
            # needed only as a dummy
            first_stage_anchor_generator=self._get_dummy_anchor_generator(),
            first_stage_target_assigner=first_stage_target_assigner,
            second_stage_target_assigner=second_stage_target_assigner,
            second_stage_sampler=second_stage_sampler,
            first_stage_box_predictor_arg_scope_fn=None,
            first_stage_atrous_rate=None,
            first_stage_box_predictor_kernel_size=None,
            first_stage_box_predictor_depth=None,
            clip_anchors_to_image=None,
            freeze_batchnorm=None,
            parallel_iterations=self.parallel_iterations)
        return self

    def predict(self,
                detection_object_boxes: tf.Tensor,
                detection_object_scores: tf.Tensor,
                num_object_detections: tf.Tensor,
                groundtruth_object_boxes: Optional[tf.Tensor] = None,
                groundtruth_object_classes: Optional[tf.Tensor] = None,
                groundtruth_object_weights: Optional[tf.Tensor] = None,
                images: Optional[tf.Tensor] = None,
                image_sizes: Optional[tf.Tensor] = None,
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
        # parent save method has more generic signature
        if not self.sample_always and not self.is_training:
            return {
                DetectionDataFields.detection_object_boxes:
                    detection_object_boxes,
                DetectionDataFields.detection_object_scores:
                    detection_object_scores,
                DetectionDataFields.num_object_detections:
                    num_object_detections,
            }
        assert groundtruth_object_boxes is not None, (
            "{}: groundtruth_object_boxes must be provided!".format(self.name))
        assert groundtruth_object_classes is not None, (
            "{}: groundtruth_object_boxes must be provided!".format(self.name))

        # pylint: disable=protected-access
        self._faster_rcnn_meta._is_training = True
        # pylint: enable=protected-access
        if images is None and image_sizes is None:
            raise ValueError(
                "{}: provide images or image_sizes!".format(self.name))
        true_image_shapes = (image_sizes if image_sizes is not None
                             else get_true_image_shapes(images))
        detection_object_boxes_absolute = local_to_image_coordinates(
            detection_object_boxes, true_image_shapes=true_image_shapes)
        max_num_proposals = tf.convert_to_tensor(
            self.sample_minibatch_size)
        self._faster_rcnn_meta.max_num_proposals = max_num_proposals
        self._provide_groundtruth_to_meta(groundtruth_object_boxes,
                                          groundtruth_object_classes,
                                          groundtruth_object_weights)
        # pylint: disable=protected-access
        (groundtruth_boxlists, groundtruth_classes_with_background_list, _,
         groundtruth_weights_list
         ) = self._faster_rcnn_meta._format_groundtruth_data(true_image_shapes)
        (detection_object_boxes_absolute, detection_object_scores,
         num_object_detections
         ) = self._faster_rcnn_meta._sample_box_classifier_batch(
             detection_object_boxes_absolute, detection_object_scores,
             num_object_detections,
             groundtruth_boxlists, groundtruth_classes_with_background_list,
             groundtruth_weights_list)
        detection_object_boxes = normalize_bbox(
            detection_object_boxes_absolute,
            true_image_shapes=true_image_shapes)
        # pylint: enable=protected-access
        return {
            DetectionDataFields.detection_object_boxes: detection_object_boxes,
            DetectionDataFields.detection_object_scores:
                detection_object_scores,
            DetectionDataFields.num_object_detections: num_object_detections,
        }

    def _provide_groundtruth_to_meta(self,
                                     groundtruth_object_boxes,
                                     groundtruth_object_classes,
                                     groundtruth_object_weights):
        groundtruth_object_classes = tf.clip_by_value(
            groundtruth_object_classes, 0, 1)
        groundtruth_object_classes_one_hot = tf.one_hot(
            groundtruth_object_classes, 2)[..., 1:]
        self._faster_rcnn_meta.provide_groundtruth(
            groundtruth_object_boxes, groundtruth_object_classes_one_hot)
        if groundtruth_object_weights is not None:
            # pylint: disable=protected-access
            # is needed since in original method comparison is done as
            # if not tensor, which is not allowed
            self._faster_rcnn_meta._groundtruth_lists[
                od_standard_fields.BoxListFields.weights] = (
                    groundtruth_object_weights)

    @staticmethod
    def _get_dummy_anchor_generator():
        return GridAnchorGenerator(None, None)


class ROIPoolingPlugin(nc7.model.ModelPlugin):
    """
    Plugin to perform ROI pooling, e.g. it will crop the images using bounding
    boxes and then will rescale them to same size and perform max pooling on it

    Parameters
    ----------
    maxpool_kernel_size
        kernel size for max pooling
    maxpool_stride
        stride for max pooling
    initial_crop_size
        size of the features after crop
    use_matmul_crop_and_resize
        if the crop and resize method using matmul should be used; otherwise the
        original tf.image.crop_and_resize is used; may have performance gain for
        small crop sizes
    squash_batch_and_detection_dims
        if the output should have shape
        [bs*max_num_proposals, ROI_height, ROI_width, num_channels]

    Attributes
    ----------
    incoming_keys
        * feature_maps : feature maps to crop; float32,
          shape [bs, height, width, num_channels]
        * detection_object_boxes : detection boxes in normalized coordinates,
          tf.float32, [bs, num_detections, 4]

    generated_keys
        * feature_maps : ROI feature maps; float32,
          shape [bs, max_num_proposals, ROI_height, ROI_width, num_channels]
          or [bs*max_num_proposals, ROI_height, ROI_width, num_channels] if
          squash_batch_and_detection_dims == True
    """
    incoming_keys = [
        "feature_maps",
        DetectionDataFields.detection_object_boxes,
    ]
    generated_keys = [
        "feature_maps",
    ]

    def __init__(self, *,
                 maxpool_kernel_size: int = 2,
                 maxpool_stride: int = 2,
                 initial_crop_size: int = 18,
                 use_matmul_crop_and_resize: bool = True,
                 squash_batch_and_detection_dims: bool = True,
                 **plugin_kwargs):
        super().__init__(**plugin_kwargs)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.initial_crop_size = initial_crop_size
        self.use_matmul_crop_and_resize = use_matmul_crop_and_resize
        self.squash_batch_and_detection_dims = squash_batch_and_detection_dims
        self._maxpool_layer = None

    def create_keras_layers(self):
        self._maxpool_layer = self.add_keras_layer(
            tf.keras.layers.MaxPooling2D(self.maxpool_kernel_size,
                                         strides=self.maxpool_stride))

    def predict(self, feature_maps, detection_object_boxes
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        batch_size_and_num_detections = tf.shape(detection_object_boxes)[:2]
        crop_and_resize_fn = (
            object_detection_ops.matmul_crop_and_resize
            if self.use_matmul_crop_and_resize
            else object_detection_ops.native_crop_and_resize)
        cropped_feature_maps = crop_and_resize_fn(
            feature_maps, detection_object_boxes,
            [self.initial_crop_size, self.initial_crop_size])
        cropped_feature_maps_squashed = tf_ops.squash_dims(
            cropped_feature_maps, [0, 1])

        roi_feature_maps_squashed = self._maxpool_layer(
            cropped_feature_maps_squashed)

        if self.squash_batch_and_detection_dims:
            return {"feature_maps": roi_feature_maps_squashed}

        roi_shape_unsquashed = tf.concat(
            [batch_size_and_num_detections,
             tf.shape(roi_feature_maps_squashed)[1:]], 0)
        roi_feature_maps = tf.reshape(
            roi_feature_maps_squashed, roi_shape_unsquashed)
        return {"feature_maps": roi_feature_maps}


class FasterRCNNSecondStagePlugin(nc7.model.ModelPlugin):
    """
    Plugin for second stage of Faster RCNN inherited from object detection API
    from google.

    In contrast to original implementation, it will not perform nms in the end.

    Parameters
    ----------
    num_classes
        number of classes
    freeze_batchnorm
        if the batch normalization parameters should be frozen during training;
        is better to set to True for small batch sizes
    batch_norm_params
        batch normalization parameters, like 'momentum', 'center', 'scale' and
        'epsilon'
    parallel_iterations
        number of parallel iteration to use for tf.map_fn

    Attributes
    ----------
    incoming_keys
        * images : original images
        * image_sizes : (optional) image sizes with shape [bs, 3] and every
          instance has [height, width, num_channels] inside; if not provided,
          will be inferred from images itself
        * feature_maps : feature maps after ROI crop, tf.float32, shape
          [bs, max_num_proposals, ROI_height, ROI_width, num_channels] or
          of shape [total_num_proposals, ROI_height, ROI_width, num_channels]
        * proposal_boxes : decoded proposal bounding boxes in normalized
          coordinates from RPN; [bs, max_num_proposals, 4], tf.float32
        * num_proposals : number of proposals from rpn

    generated_keys
        * proposal_boxes : decoded proposal bounding boxes in absolute
          coordinates from RPN; [bs, max_num_proposals, 4], tf.float32
        * refined_box_encodings : refined box encodings in absolute coordinates;
          [bs, max_num_proposals, num_classes, 4], tf.float32
        * class_predictions_with_background : class logits for each anchor;
          [bs, max_num_proposals, num_classes + 1]; tf.float32
          Note that this tensor *includes* background class predictions
          (at class index 0).
        * num_proposals : number of proposals for each sample in batch; [bs],
          tf.int32
        * detection_object_boxes : detection boxes in normalized coordinates,
          tf.float32, [bs, num_detections, 4]
        * detection_objects_scores : detection scores;
          tf.float32, [bs, num_detections, 4]
        * num_object_detections : number of detections in each sample; tf.int32,
          [bs]
        * detection_object_classes : object classes 0-based; tf.int32,
          [bs, max_num_proposals]
    """
    incoming_keys = [
        ImageDataFields.images,
        "feature_maps",
        "proposal_boxes",
        "num_proposals",
        "_" + ImageDataFields.image_sizes,
    ]
    generated_keys = [
        "refined_box_encodings",
        "class_predictions_with_background",
        "proposal_boxes",
        "num_proposals",
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.detection_object_classes,
        DetectionDataFields.num_object_detections,
    ]

    def __init__(self, *,
                 num_classes,
                 batch_norm_params: Optional[dict] = None,
                 freeze_batchnorm=False,
                 parallel_iterations=16,
                 **plugin_kwargs):
        allow_mixed_precision = plugin_kwargs.pop('allow_mixed_precision', None)
        if allow_mixed_precision:
            raise ValueError('allow_mixed_precision is not allowed for '
                             '{}! Remove it from config'.format(self.name))
        super().__init__(allow_mixed_precision=False, **plugin_kwargs)
        self.num_classes = num_classes
        self.parallel_iterations = parallel_iterations
        self.batch_norm_params = batch_norm_params
        self.freeze_batchnorm = freeze_batchnorm
        if isinstance(self.dropout, dict) and "rate" in self.dropout:
            self._dropout_keep_prob = 1.0 - self.dropout["rate"]
        else:
            self._dropout_keep_prob = None
        self._faster_rcnn_meta = None  # type:  FasterRCNNMetaArchSecondStage

    def build(self):
        super(FasterRCNNSecondStagePlugin, self).build()

        first_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'proposal')
        second_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection')
        second_stage_mask_rcnn_box_predictor = self.add_keras_layer(
            self._get_mask_rcnn_box_predictor())

        self._faster_rcnn_meta = FasterRCNNMetaArchSecondStage(
            num_classes=self.num_classes,
            first_stage_target_assigner=first_stage_target_assigner,
            second_stage_target_assigner=second_stage_target_assigner,
            second_stage_mask_rcnn_box_predictor=
            second_stage_mask_rcnn_box_predictor,
            second_stage_non_max_suppression_fn=None,
            second_stage_score_conversion_fn=tf.nn.softmax,
            parallel_iterations=self.parallel_iterations)
        return self

    def predict(self, images: tf.Tensor,
                feature_maps: tf.Tensor,
                proposal_boxes: tf.Tensor,
                num_proposals: tf.Tensor,
                image_sizes: Optional[tf.Tensor] = None):
        # pylint: disable=arguments-differ,too-many-locals
        # parent save method has more generic signature
        max_num_proposals = tf.shape(proposal_boxes)[1]

        if len(feature_maps.shape) == 4:
            unsquashed_shape = tf.concat(
                [[-1, max_num_proposals], tf.shape(feature_maps)[1:]], 0)
            feature_maps = tf.reshape(feature_maps, unsquashed_shape)

        true_image_shapes = (image_sizes if image_sizes is not None
                             else get_true_image_shapes(images))
        image_shape = tf.shape(images)

        # pylint: enable=unused-argument,too-many-arguments
        def _second_stage_nms_fn(boxes_classwise, scores_classwise,
                                 change_coordinate_frame=False,
                                 **kwargs):  # pylint: disable=unused-argument
            max_scores, max_scores_inds = tf.nn.top_k(
                scores_classwise, 1, sorted=False)
            max_scores = tf.squeeze(max_scores, -1)
            max_classes = tf.squeeze(max_scores_inds, -1)
            max_class_boxes = tf.batch_gather(boxes_classwise, max_scores_inds)
            max_class_boxes = tf.squeeze(max_class_boxes, 2)
            num_proposals_max_score = tf.reduce_sum(
                tf.cast(tf.greater(max_scores, 0), tf.int32), -1)
            if change_coordinate_frame:
                max_class_boxes = normalize_bbox(
                    max_class_boxes, tf.cast(tf.shape(images)[1:3], tf.float32))
            return (max_class_boxes, max_scores, max_classes,
                    None, None, num_proposals_max_score)

        self._faster_rcnn_meta.set_second_stage_nms_fn(_second_stage_nms_fn)
        self._faster_rcnn_meta.set_is_training(self.is_training)
        # pylint: disable=protected-access
        self._faster_rcnn_meta._max_num_proposals = max_num_proposals
        prediction_dict = self._faster_rcnn_meta._box_prediction(
            feature_maps, proposal_boxes, image_shape)

        postprocessed_dict = self._faster_rcnn_meta._postprocess_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            num_proposals,
            true_image_shapes,
        )
        # pylint: enable=protected-access

        refined_box_encodings = _total_num_proposals_to_batch(
            prediction_dict["refined_box_encodings"],
            max_num_proposals)
        class_predictions_with_background = _total_num_proposals_to_batch(
            prediction_dict["class_predictions_with_background"],
            max_num_proposals)

        result = {
            "refined_box_encodings": refined_box_encodings,
            "class_predictions_with_background":
                class_predictions_with_background,
            "proposal_boxes": prediction_dict["proposal_boxes"],
            "num_proposals": num_proposals,
            DetectionDataFields.detection_object_boxes:
                postprocessed_dict["detection_boxes"],
            DetectionDataFields.detection_object_scores:
                postprocessed_dict["detection_scores"],
            DetectionDataFields.detection_object_classes:
                tf.cast(postprocessed_dict["detection_classes"], tf.int32),
            DetectionDataFields.num_object_detections:
                postprocessed_dict["num_detections"],
        }
        return result

    def _get_mask_rcnn_box_predictor(self):
        use_dropout = self.is_training and self._dropout_keep_prob is not None
        conv_hyperparams = KerasLayerHyperparamsFromData(
            activation=self.activation, initializer=self.initializer,
            batch_norm_params=self.batch_norm_params
        )
        fc_hyperparams = KerasLayerHyperparamsFromData(
            activation=self.activation, initializer=self.initializer,
            batch_norm_params=self.batch_norm_params)
        mask_rcnn_box_predictor = (
            box_predictor_builder.build_mask_rcnn_keras_box_predictor(
                is_training=None,
                num_classes=self.num_classes,
                add_background_class=True,
                fc_hyperparams=fc_hyperparams,
                freeze_batchnorm=self.freeze_batchnorm,
                use_dropout=use_dropout,
                dropout_keep_prob=self._dropout_keep_prob,
                box_code_size=4,
                conv_hyperparams=conv_hyperparams))
        return mask_rcnn_box_predictor


# pylint: disable=unused-argument,too-many-arguments
# clip_window must be as an arg
def _first_stage_nms_fn(boxes, scores, clip_window=None,
                        iou_threshold=0.7,
                        score_threshold=0.0,
                        max_size_per_class=1000,
                        max_total_size=1000,
                        parallel_iterations=16):
    object_classes = tf.zeros_like(scores, dtype=tf.int32)
    (object_boxes_nms, object_scores_nms, _, num_objects_nms, _
     ) = batch_multiclass_non_max_suppression(
         object_boxes=boxes, object_scores=scores,
         object_classes=object_classes,
         num_classes=1,
         iou_threshold=iou_threshold,
         score_threshold=score_threshold,
         max_total_size=max_total_size,
         max_size_per_class=max_size_per_class,
         parallel_iterations=parallel_iterations,
         is_classwise=True)
    return (object_boxes_nms, object_scores_nms, None, None, None,
            num_objects_nms)


def _total_num_proposals_to_batch(tensor, max_num_proposals):
    return tf.reshape(tensor,
                      tf.concat([[-1, max_num_proposals],
                                 tf.shape(tensor)[1:]], 0))
