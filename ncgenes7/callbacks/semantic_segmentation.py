# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Callbacks for semantic segmentation
"""

from typing import Optional
import warnings

import nucleus7 as nc7
from nucleus7.utils import io_utils
import numpy as np
import skimage.io

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.semantic_segmentation import SegmentationDataFields
from ncgenes7.utils import general_ops
from ncgenes7.utils import image_utils


class SemanticSegmentationSaver(nc7.coordinator.SaverCallback):
    """
    Saver callback for semantic segmentation

    If it has incoming key as images, it will blend the rgb images generated
    from classes with original image itself (resulted image resolution is
    resolution of semantic segmentation classes)

    Images are stored to following file names inside of log_dir:
        * semantic segmentation classes: {image_fname}_segmentation_classes.png
        * raw probabilities (optional): {image_fname}_probabilities.npy

    Parameters
    ----------
    rgb_class_mapping
        which mapping should be used to map classes to rgb codes; can be file
        name of json file or dict with mapping; if not specified, than
        {class: (class, class, class)} mapping will be used
    save_probabilities
        if probabilities should be stored as numpy arrays
    save_file_extension
        extension to use to save the image data

    Attributes
    ----------
    incoming_keys
        * images : image, [bs, h, w, num_channels], np.float
        * segmentation_class_logits : (optional) class logits,
          [bs, h, w, num_classes], np.float
        * segmentation_classes : (optional) segmented classes, [bs, h, w],
          np.float
        * segmentation_edges : (optional) edge probabilities, [bs, h, w],
          np.float
        * images_fnames : (optional) image file name
    """
    incoming_keys = [
        "_" + ImageDataFields.images,
        "_" + SegmentationDataFields.segmentation_class_logits,
        "_" + SegmentationDataFields.segmentation_classes,
    ]

    def __init__(self, *,
                 rgb_class_mapping: Optional[dict] = None,
                 save_file_extension: str = ".png",
                 save_probabilities: bool = False,
                 **callback_kwargs):
        super(SemanticSegmentationSaver, self).__init__(
            **callback_kwargs)

        if save_file_extension.startswith("."):
            save_file_extension = save_file_extension[1:]
        self.save_file_extension = save_file_extension
        self.rgb_class_mapping = io_utils.maybe_load_json(rgb_class_mapping)
        if (self.rgb_class_mapping is not None and
                'class_ids' in self.rgb_class_mapping):
            self.rgb_class_mapping = dict(zip(
                self.rgb_class_mapping['class_ids'],
                self.rgb_class_mapping['rgb']))
        self.save_probabilities = save_probabilities

    def save_sample(self, *, images=None, segmentation_class_logits=None,
                    segmentation_classes=None):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        self._save_segmentation_classes(
            images=images,
            segmentation_class_logits=segmentation_class_logits,
            segmentation_classes=segmentation_classes)
        if self.save_probabilities:
            self._save_probabilities(segmentation_class_logits)

    def on_iteration_end(self, *, images=None,
                         segmentation_class_logits=None,
                         segmentation_classes=None,
                         save_names=None):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        if (segmentation_class_logits is None
                and segmentation_classes is None):
            raise ValueError(
                "At least one of segmentation_class_logits or"
                "segmentation_classes should be provided!")
        if self.save_probabilities and segmentation_class_logits is None:
            raise ValueError(
                "Provide class probabilities if you want to save them as"
                "binary files!")
        return super(SemanticSegmentationSaver, self).on_iteration_end(
            images=images,
            segmentation_class_logits=segmentation_class_logits,
            segmentation_classes=segmentation_classes,
            save_names=save_names)

    def _save_segmentation_classes(self,
                                   images=None, segmentation_class_logits=None,
                                   segmentation_classes=None):
        if segmentation_classes is None:
            segmentation_classes = np.argmax(segmentation_class_logits, -1)

        segmentation_classes_rgb = image_utils.labels2rgb(
            segmentation_classes, self.rgb_class_mapping)
        image_to_save = segmentation_classes_rgb

        if images is not None:
            if segmentation_classes_rgb.max() > 1:
                segmentation_classes_rgb = segmentation_classes_rgb.astype(
                    np.float32)
                segmentation_classes_rgb = segmentation_classes_rgb / 255.0
            images_resized = image_utils.maybe_resize_and_expand_image(
                images, segmentation_classes_rgb)
            image_to_save = image_utils.blend_images(
                images_resized, segmentation_classes_rgb)

        save_fname = ".".join([self.save_name + "_segmentation_classes",
                               self.save_file_extension])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(save_fname, image_to_save)

    def _save_probabilities(self, segmentation_class_logits: np.ndarray):
        save_fname = ".".join([self.save_name + "_probabilities", "npy"])
        segmentation_class_probabilities = general_ops.softmax_np(
            segmentation_class_logits, axis=-1)
        np.save(save_fname, segmentation_class_probabilities,
                allow_pickle=False)
