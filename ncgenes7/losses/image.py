# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Implementation of ModelLoss for image related tasks
"""
from functools import wraps
from typing import List

import nucleus7 as nc7
from nucleus7.utils.tf_varscopes_utils import with_name_scope
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields


def _change_single_to_list(function):
    @wraps(function)
    def wrapped(self, **data):
        for each_key, each_value in data.items():
            if not isinstance(each_value, list):
                data[each_key] = [each_value]
        return function(self, **data)

    return wrapped


class StereoDepthLoss(nc7.model.ModelLoss):
    """
    Loss for stereo depth estimation

    Parameters
    ----------
    use_2nd_order_gradients
        if second order image gradient loss should be calculated
    loss_ssim_alpha
        ssim reconstruction coefficient

    Attributes
    ----------
    incoming_keys
        * images_left : left input image, {tf.float32, [tf.float32]*n},
          [bs, h, w, num_channels]
        * images_right : right image, {tf.float32, [tf.float32]*n},
          [bs, h, w, num_channels]
        * images_left_reconstructed : reconstructed left image,
          {tf.float32, [tf.float32]*n}, [bs, h, w, num_channels]
        * images_right_reconstructed : reconstructed right image,
          {tf.float32, [tf.float32]*n}, [bs, h, w, num_channels]
        * disparity_images_left : disparity for left image,
          {tf.float32, [tf.float32]*n}, [bs, h, w, 1]
        * disparity_images_right : disparity to right image,
          {tf.float32, [tf.float32]*n}, [bs, h, w, 1]
        * disparity_images_left_inverted : disparity reconstructed from right
          disparity using left one as transformation,
          {tf.float32, [tf.float32]*n}, [bs, h, w, 1]
        * disparity_images_right_inverted : disparity reconstructed from left
          disparity using right one as transformation,
          {tf.float32, [tf.float32]*n}, [bs, h, w, 1]

    generated_keys
        * loss_appearance_matching : matching of reconstruction
        * loss_disp_smoothness : disparity smoothness loss
        * loss_disp_consistency : disparity consistency
        * loss_disp_smoothness_2nd_order : (optional) second order
          smoothness loss

    References
    ----------
    loss_disp_smoothness, disp_consistency, appearance_matching
        https://arxiv.org/abs/1609.03677
    disp_smoothness_2nd_order
        https://arxiv.org/abs/1704.07813
    """
    incoming_keys = [
        ImageDataFields.images_left,
        ImageDataFields.images_right,
        ImageDataFields.images_left + "_reconstructed",
        ImageDataFields.images_right + "_reconstructed",
        ImageDataFields.disparity_images_left,
        ImageDataFields.disparity_images_right,
        ImageDataFields.disparity_images_left + "_inverted",
        ImageDataFields.disparity_images_right + "_inverted",
    ]
    generated_keys = [
        "loss_appearance_matching",
        "loss_disp_smoothness",
        "loss_disp_consistency",
        "loss_disp_smoothness_2nd_order",
    ]

    def __init__(self, *,
                 use_2nd_order_gradients: bool = False,
                 loss_ssim_alpha: float = 0.85,
                 **loss_kwargs):
        super().__init__(**loss_kwargs)
        self.use_2nd_order_gradients = use_2nd_order_gradients
        self.loss_ssim_alpha = loss_ssim_alpha

    @_change_single_to_list
    def process(self, *,
                images_left, images_right,
                images_left_reconstructed, images_right_reconstructed,
                disparity_images_left, disparity_images_right,
                disparity_images_left_inverted, disparity_images_right_inverted
                ):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        assert (len(images_left) == len(images_right) ==
                len(disparity_images_left) == len(disparity_images_right) ==
                len(disparity_images_left_inverted) ==
                len(disparity_images_right_inverted)
                ), ("Number of disparities should be the same for left "
                    "and right and and be equal to number of images!!! "
                    "images: {}: {}, disparities: {} : {}, inv: {} : {}"
                    ).format(len(images_left), len(images_right),
                             len(disparity_images_left),
                             len(disparity_images_right),
                             len(disparity_images_left_inverted),
                             len(disparity_images_right_inverted))

        loss_appearance = _get_loss_appearance(
            alpha=self.loss_ssim_alpha,
            images_left=images_left,
            images_right=images_right,
            images_left_reconstructed=images_left_reconstructed,
            images_right_reconstructed=images_right_reconstructed,
        )
        loss_disp_smoothness = _get_loss_disp_smoothness(
            images_left=images_left,
            images_right=images_right,
            disparity_images_left=disparity_images_left,
            disparity_images_right=disparity_images_right,
        )
        loss_disp_consistency = _get_loss_disp_consistency(
            disparity_images_left=disparity_images_left,
            disparity_images_right=disparity_images_right,
            disparity_images_left_inverted=disparity_images_left_inverted,
            disparity_images_right_inverted=disparity_images_right_inverted,
        )
        losses = {"loss_appearance_matching": loss_appearance,
                  "loss_disp_smoothness": loss_disp_smoothness,
                  "loss_disp_consistency": loss_disp_consistency}

        if self.use_2nd_order_gradients:
            loss_disp_smoothness_2nd_order = (
                _get_loss_disp_smoothness_2nd_order(
                    disparity_images_left, disparity_images_right))
            losses["loss_disp_smoothness_2nd_order"] = (
                loss_disp_smoothness_2nd_order)
        return losses


class SoftmaxLossWithResizedLabels(nc7.model.ModelLoss):
    """
    Sparse softmax loss for case when spatial dimensions of predictions
    are not the same as labels. It will resize labels to predictions size.
    Resizing is done by :func:`tf.image.resize_nearest_neighbor`

    Attributes
    ----------
    incoming_keys
        * logits : logits, np.float
        * labels : labels,  np.int32
    generated_keys
        * loss : calculated loss
    """
    incoming_keys = [
        "logits",
        "labels",
    ]
    generated_keys = [
        "loss",
    ]

    def process(self, logits, labels):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        logits_size = tf.shape(logits)[1:3]
        if labels.dtype == tf.uint8:
            labels = tf.cast(labels, tf.int64)
        labels_resized = tf.image.resize_nearest_neighbor(labels, logits_size)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels_resized, logits=logits)
        return {'loss': loss}


class SSIMWithResizedLabels(nc7.model.ModelLoss):
    """
    Structural Similarity Index (SSIM) for measuring image quality
    when spatial dimensions of predictions
    are not the same as labels. It will resize labels to predictions size.
    Resizing is done by :func:`tf.image.resize_nearest_neighbor`

    Parameters
    ----------
    max_value
        dynamic range of the images (i.e., the difference between the maximum
        the and minimum allowed values)

    Attributes
    ----------
    incoming_keys
        * predictions : predicted images, tf.float
        * labels : groundtruth images,  tf.float
    generated_keys
        * loss : calculated loss
    """
    incoming_keys = [
        "predictions",
        'labels',
    ]
    generated_keys = [
        "loss",
    ]

    def __init__(self, max_value: float = 1.0, **loss_kwargs):
        super().__init__(**loss_kwargs)
        self.max_value = max_value

    def process(self, predictions, labels):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        predictions_size = tf.shape(predictions)[1:3]
        labels_resized = tf.image.resize_nearest_neighbor(
            labels, predictions_size)
        ssim = tf.image.ssim(img1=labels_resized, img2=predictions,
                             max_val=self.max_value)
        loss = 1.0 - ssim
        loss = tf.reduce_mean(loss)
        return {'loss': loss}


@with_name_scope("loss_appearance_matching")
def _get_loss_appearance(alpha: float,
                         images_left: List[tf.Tensor],
                         images_right: List[tf.Tensor],
                         images_left_reconstructed: List[tf.Tensor],
                         images_right_reconstructed: List[tf.Tensor]
                         ) -> tf.Tensor:
    # pylint: disable=too-many-locals
    # cannot reduce number of locals without more code complexity
    losses_at_scales = []
    images_pairs = [(images_left, images_left_reconstructed),
                    (images_right, images_right_reconstructed)]
    number_of_scales = len(images_left)
    for each_images, each_images_reconstr in images_pairs:
        for i_scale in range(number_of_scales):
            images_at_scale = each_images[i_scale]
            images_reconstr_at_scale = each_images_reconstr[i_scale]
            ssim = tf.image.ssim(images_at_scale, images_reconstr_at_scale,
                                 max_val=1.0)
            loss1 = tf.reduce_mean(
                alpha * tf.clip_by_value((1 - ssim) / 2, 0, 1))
            loss2 = (1 - alpha) * tf.losses.absolute_difference(
                images_at_scale, images_reconstr_at_scale)
            losses_at_scales.append(loss1 + loss2)
    loss = tf.add_n(losses_at_scales, name="loss_appearance_matching")
    return loss


@with_name_scope("loss_disp_smoothness")
def _get_loss_disp_smoothness(*,
                              images_left: List[tf.Tensor],
                              images_right: List[tf.Tensor],
                              disparity_images_left: List[tf.Tensor],
                              disparity_images_right: List[tf.Tensor]
                              ) -> tf.Tensor:
    # pylint: disable=too-many-locals
    # cannot reduce number of locals without more code complexity
    losses_at_scales = []
    images_pairs = [(images_left, disparity_images_left),
                    (images_right, disparity_images_right)]
    number_of_scales = len(images_left)
    for each_images, each_images_disparity in images_pairs:
        for i_scale in range(number_of_scales):
            images_at_scale = each_images[i_scale]
            disp_at_scale = each_images_disparity[i_scale]

            disp_dx = _get_image_dx(disp_at_scale)[:, 1:, :, :]
            disp_dy = _get_image_dy(disp_at_scale)[:, :, 1:, :]

            image_dx = _get_image_dx(images_at_scale)[:, 1:, :, :]
            image_dy = _get_image_dy(images_at_scale)[:, :, 1:, :]

            # pylint: disable=invalid-unary-operand-type
            pow_x = tf.exp(-tf.abs(image_dx))
            pow_y = tf.exp(-tf.abs(image_dy))
            # pylint: enable=invalid-unary-operand-type

            pow_x_at_scale = pow_x / 2 ** (number_of_scales - i_scale - 1)
            pow_y_at_scale = pow_y / 2 ** (number_of_scales - i_scale - 1)

            loss_x = tf.abs(disp_dx) * pow_x_at_scale
            loss_y = tf.abs(disp_dy) * pow_y_at_scale
            losses_at_scales.append(tf.reduce_mean(loss_x + loss_y))
    loss = tf.add_n(losses_at_scales, name="loss_disp_smoothness")
    return loss


@with_name_scope("loss_disp_smoothness_2nd_order")
def _get_loss_disp_smoothness_2nd_order_single(disparity: List[tf.Tensor],
                                               name_suffix: str) -> tf.Tensor:
    losses_at_scales = []
    number_of_scales = len(disparity)
    for i_scale, each_disparity in enumerate(disparity):
        disp_dx = _get_image_dx(each_disparity)[:, 1:, :, :]
        disp_dy = _get_image_dy(each_disparity)[:, :, 1:, :]

        disp_d2x = _get_image_dx(disp_dx)[:, 1:, :, :]
        disp_d2y = _get_image_dy(disp_dy)[:, :, 1:, :]

        loss_x = tf.abs(disp_d2x) / 2 ** (number_of_scales - i_scale - 1)
        loss_y = tf.abs(disp_d2y) / 2 ** (number_of_scales - i_scale - 1)
        losses_at_scales.append(tf.reduce_mean(loss_x + loss_y))
    loss = tf.add_n(losses_at_scales,
                    name="loss_disp_smoothness_2nd_order" + name_suffix)
    return loss


@with_name_scope("loss_disp_smoothness_2nd_order")
def _get_loss_disp_smoothness_2nd_order(
        disparity_images_left: List[tf.Tensor],
        disparity_images_right: List[tf.Tensor]
) -> tf.Tensor:
    loss_disp_smoothness_2nd_order_left = (
        _get_loss_disp_smoothness_2nd_order_single(
            disparity=disparity_images_left, name_suffix="_left"))
    loss_disp_smoothness_2nd_order_right = (
        _get_loss_disp_smoothness_2nd_order_single(
            disparity=disparity_images_right, name_suffix="_right"))
    loss_disp_smoothness_2nd_order = tf.add_n(
        [loss_disp_smoothness_2nd_order_left,
         loss_disp_smoothness_2nd_order_right],
        name="loss_disp_smoothness_2nd_order")
    return loss_disp_smoothness_2nd_order


@with_name_scope("loss_disp_consistency")
def _get_loss_disp_consistency(*,
                               disparity_images_left: List[tf.Tensor],
                               disparity_images_right: List[tf.Tensor],
                               disparity_images_left_inverted: List[tf.Tensor],
                               disparity_images_right_inverted: List[tf.Tensor]
                               ) -> tf.Tensor:
    losses_at_scales = []
    images_pairs = [(disparity_images_left, disparity_images_left_inverted),
                    (disparity_images_right, disparity_images_right_inverted)]
    number_of_scales = len(disparity_images_left)
    for each_disp, each_disp_inv in images_pairs:
        for i_scale in range(number_of_scales):
            losses_at_scales.append(
                tf.losses.absolute_difference(
                    each_disp[i_scale], each_disp_inv[i_scale]))
    loss = tf.add_n(losses_at_scales, name="loss_disp_consistency")
    return loss


def _get_image_dx(image: tf.Tensor) -> tf.Tensor:
    """calculates image gradients in x direction"""
    image_dx = image[:, :, :-1, :] - image[:, :, 1:, :]
    return image_dx


def _get_image_dy(image: tf.Tensor) -> tf.Tensor:
    """calculates image gradients in y direction"""
    image_dy = image[:, :-1, :, :] - image[:, 1:, :, :]
    return image_dy
