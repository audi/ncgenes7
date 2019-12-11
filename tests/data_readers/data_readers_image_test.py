# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import io
import os

from absl.testing import parameterized
from nucleus7.utils import tf_data_utils
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_readers.image import ImageDataReader
from ncgenes7.data_readers.image import ImageDataReaderTf
from ncgenes7.data_readers.image import ImageDataReaderTfRecords
from ncgenes7.data_readers.image import ImageEncoder
from ncgenes7.utils import image_utils


class TestImageDataReader(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.orig_image_size = [10, 16]
        self.image_fname = os.path.join(self.get_temp_dir(), "image_0.png")

    @parameterized.parameters(
        {"image_number_of_channels": 3, "image_size": None},
        {"image_number_of_channels": 1, "image_size": [24, 36]},
        {"image_number_of_channels": 3, "image_size": [24, 36]},
        {"image_number_of_channels": 3, "image_size": [24, 36],
         "result_dtype": "uint8"},
        {"image_number_of_channels": 3, "image_size": [24, 36],
         "interpolation_order": 0},
        {"image_number_of_channels": 3, "image_size": [24, 36],
         "interpolation_order": 0, "result_dtype": "uint8"},
        {"image_number_of_channels": 3, "image_size": None,
         "result_dtype": "uint8"},
        {"image_number_of_channels": 1, "image_size": None},
    )
    def test_read(self, image_size, image_number_of_channels,
                  interpolation_order=1, result_dtype="float32"):
        image = self._get_image(image_number_of_channels)
        skimage.io.imsave(self.image_fname, image)

        reader = ImageDataReader(
            image_size=image_size,
            image_number_of_channels=image_number_of_channels,
            interpolation_order=interpolation_order,
            result_dtype=result_dtype).build()
        data_read = reader.read(images=self.image_fname)

        image_must = image.astype(np.float32) / np.iinfo("uint8").max
        if image_must.ndim == 2:
            image_must = np.expand_dims(image_must, -1)
        if image_size:
            image_must = skimage.transform.resize(
                image_must, output_shape=image_size, order=interpolation_order,
                mode="reflect")
            image_must = (image_must * np.iinfo(np.uint8).max).astype(np.uint8)
            image_must = image_must.astype(np.float32) / np.iinfo(np.uint8).max
        if result_dtype != "float32":
            image_must = (image_must * np.iinfo(result_dtype).max).astype(
                result_dtype)

        data_must = {
            "images": image_must,
            "image_sizes": np.asarray(np.shape(image_must)[:2], np.int32),
            "images_fnames": np.asarray(self.image_fname)}

        self.assertSetEqual(set(data_must),
                            set(data_read))
        for each_key in data_must:
            self.assertDTypeEqual(data_read[each_key],
                                  data_must[each_key].dtype)
            if data_read[each_key].dtype == np.float32:
                self.assertAllClose(data_must[each_key],
                                    data_read[each_key])
            else:
                self.assertAllEqual(data_must[each_key],
                                    data_read[each_key])

    def _get_image(self, num_channels):
        if num_channels == 1:
            image = np.random.rand(*self.orig_image_size)
        else:
            image = np.random.rand(*self.orig_image_size, num_channels)
        dtype_max = np.iinfo("uint8").max
        image *= dtype_max
        image = image.astype(np.uint8)
        return image


class TestImageEncoder(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.image = np.random.randint(
            0, 255, size=[20, 30, 3]).astype(np.uint8)

    @parameterized.parameters({"input_dtype": "uint8"},
                              {"input_dtype": "float32"})
    def test_process_png(self, input_dtype):
        image = (self.image if input_dtype == "uint8"
                 else self.image.astype(np.float32) / 255)
        image_encoder = ImageEncoder(encoding="PNG").build()
        result = image_encoder.process(images=image)
        self.assertEqual({"images_PNG"},
                         set(result))
        image_encoded = result["images_PNG"]
        image_from_encoded = skimage.io.imread(io.BytesIO(image_encoded))
        self.assertAllEqual(self.image,
                            image_from_encoded)


class TestImageDataReaderTf(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.image_fname = os.path.join(self.get_temp_dir(), "image_1.png")
        self.original_image_size = [20, 30]

    @parameterized.parameters(
        {"image_size": None},
        {"image_size": None, "result_dtype": "float32"},
        {"image_size": [30, 45]},
        {"image_size": [30, 45],
         "result_dtype": "float32", "interpolation_order": 0},
        {"image_size": [30, 45],
         "result_dtype": "float32", "interpolation_order": 1},
        {"image_size": [30, 45],
         "result_dtype": "uint16", "interpolation_order": 0},
    )
    def test_read(self, image_size=None,
                  result_dtype="uint8",
                  interpolation_order=0,
                  image_number_of_channels=3):
        image = self._get_image(image_number_of_channels)
        reader = ImageDataReaderTf(
            image_number_of_channels=image_number_of_channels,
            image_size=image_size, interpolation_order=interpolation_order,
            result_dtype=result_dtype).build()
        image_fname = tf.convert_to_tensor(self.image_fname)
        result = reader.read(images=image_fname)

        if not image_size:
            tensor_image_shape_must = [None, None, image_number_of_channels]
        else:
            tensor_image_shape_must = image_size + [image_number_of_channels]
        self.assertListEqual(tensor_image_shape_must,
                             result[ImageDataFields.images].shape.as_list())
        self.assertEqual(getattr(tf, result_dtype),
                         result[ImageDataFields.images].dtype)
        self.assertEqual(tf.int32,
                         result[ImageDataFields.image_sizes].dtype)
        self.assertEqual(tf.string,
                         result[ImageDataFields.images_fnames].dtype)

        result_eval = self.evaluate(result)

        image_size_must = image_size if image_size else self.original_image_size
        image_size_must = np.array(image_size_must, np.int32)
        fname_must = self.image_fname

        image_must = image
        if image_size:
            # need to use tf resize method since skimage or open cv resize
            # has slightly different interpolation
            tf_resize_method = image_utils.INTERPOLATION_ORDER_TO_RESIZE_METHOD[
                interpolation_order]
            image_resized = tf.image.resize_images(
                image_must, image_size, method=tf_resize_method,
                align_corners=True)
            if interpolation_order > 0 and result_dtype != "float32":
                image_resized = tf.cast(image_resized, tf.uint8)
            image_must = self.evaluate(image_resized)

        if result_dtype == "float32":
            image_must = image_must.astype(np.float32) / tf.uint8.max
        elif result_dtype == "uint16":
            image_must = (image_must.astype(np.float32) / 255 * (2 ** 16 - 1)
                          ).astype(np.uint16)
        self.assertAllClose(image_must,
                            result_eval[ImageDataFields.images])
        self.assertAllClose(image_size_must,
                            result_eval[ImageDataFields.image_sizes])
        self.assertEqual(fname_must,
                         result_eval[ImageDataFields.images_fnames].decode())

    def _get_image(self, image_number_of_channels):
        image = np.random.randint(
            0, 255, size=[*self.original_image_size, image_number_of_channels]
        ).astype(np.uint8)
        if image_number_of_channels == 1:
            image_to_save = np.squeeze(image, -1)
        else:
            image_to_save = image
        skimage.io.imsave(self.image_fname, image_to_save)
        return image


class TestImageDataReaderTfRecords(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.images_fnames = "sample_fname"
        self.original_image_size = [20, 30]

    @parameterized.parameters(
        {"image_encoding": None, "image_size": None},
        {"image_encoding": None, "image_size": [30, 45]},
        {"image_encoding": None, "image_size": [30, 45],
         "image_number_of_channels": 1},
        {"image_encoding": None, "image_size": [30, 45],
         "result_dtype": "float32", "interpolation_order": 0},
        {"image_encoding": None, "image_size": [30, 45],
         "result_dtype": "float32", "interpolation_order": 1},
        {"image_encoding": None, "image_size": None, "result_dtype": "float32"},
        {"image_encoding": "PNG", "image_size": None},
        {"image_encoding": "PNG", "image_size": None,
         "image_number_of_channels": 1},
        {"image_encoding": "PNG", "image_size": [30, 45]},
        {"image_encoding": "PNG", "image_size": [30, 45],
         "result_dtype": "float32", "interpolation_order": 0},
        {"image_encoding": "PNG", "image_size": None,
         "result_dtype": "float32"},
        {"image_encoding": "PNG", "image_size": [30, 45],
         "result_dtype": "float32", "interpolation_order": 1},
        {"image_encoding": "PNG", "image_size": [30, 45],
         "interpolation_order": 1, "result_dtype": "uint8"},
        {"image_encoding": "PNG", "result_dtype": "uint16"},
    )
    def test_parse_tfrecord_example(self, image_encoding=None, image_size=None,
                                    result_dtype="uint8",
                                    interpolation_order=0,
                                    image_number_of_channels=3,
                                    with_fname=True):
        image, serialized_example = self._get_image_example(
            image_number_of_channels, image_encoding, with_fname)

        reader = ImageDataReaderTfRecords(
            image_encoding=image_encoding,
            image_number_of_channels=image_number_of_channels,
            image_size=image_size,
            interpolation_order=interpolation_order,
            result_dtype=result_dtype,
        ).build()
        result = reader.parse_tfrecord_example(serialized_example)

        self.assertSetEqual({ImageDataFields.images,
                             ImageDataFields.image_sizes,
                             ImageDataFields.images_fnames},
                            set(result))
        if image_encoding:
            tensor_image_shape_must = (image_size if image_size
                                       else [None, None]
                                       ) + [image_number_of_channels]
        else:
            tensor_image_shape_must = (image_size if image_size
                                       else self.original_image_size
                                       ) + [image_number_of_channels]
        self.assertListEqual(tensor_image_shape_must,
                             result[ImageDataFields.images].shape.as_list())
        self.assertEqual(getattr(tf, result_dtype),
                         result[ImageDataFields.images].dtype)
        self.assertEqual(tf.int32,
                         result[ImageDataFields.image_sizes].dtype)
        self.assertEqual(tf.string,
                         result[ImageDataFields.images_fnames].dtype)

        result_eval = self.evaluate(result)
        image_size_must = image_size if image_size else self.original_image_size
        image_size_must = np.array(image_size_must, np.int32)
        fname_must = self.images_fnames if with_fname else "no_file_name"

        image_must = image
        if image_size:
            # need to use tf resize method since skimage or open cv resize
            # has slightly different interpolation
            tf_resize_method = image_utils.INTERPOLATION_ORDER_TO_RESIZE_METHOD[
                interpolation_order]
            image_resized = tf.image.resize_images(
                image_must, image_size, method=tf_resize_method,
                align_corners=True)
            if interpolation_order > 0 and result_dtype != "float32":
                image_resized = tf.cast(image_resized, tf.uint8)
            image_must = self.evaluate(image_resized)

        if result_dtype == "float32":
            image_must = image_must.astype(np.float32) / tf.uint8.max
        elif result_dtype == "uint16":
            image_must = (image_must.astype(np.float32) / 255 * (2 ** 16 - 1)
                          ).astype(np.uint16)
        self.assertAllClose(image_must,
                            result_eval[ImageDataFields.images])
        self.assertAllClose(image_size_must,
                            result_eval[ImageDataFields.image_sizes])
        self.assertEqual(fname_must,
                         result_eval[ImageDataFields.images_fnames].decode())

    def _get_image_example(self, image_number_of_channels,
                           image_encoding, with_fname):
        image = np.random.randint(
            0, 255, size=[*self.original_image_size, image_number_of_channels]
        ).astype(np.uint8)

        feature_data = {}
        if with_fname:
            feature_data[ImageDataFields.images_fnames] = self.images_fnames
        if not image_encoding:
            feature_data[ImageDataFields.images] = image
            image_size = np.asarray(image.shape[:2], np.int32)
            feature_data[ImageDataFields.image_sizes] = image_size
        else:
            with io.BytesIO() as bytes_io:
                if image_number_of_channels == 1:
                    image_to_save = np.squeeze(image, -1)
                else:
                    image_to_save = image
                skimage.io.imsave(bytes_io, image_to_save,
                                  format_str=image_encoding)
                image_png = bytes_io.getvalue()
            feature_data[ImageDataFields.images_PNG] = image_png

        feature = tf_data_utils.nested_to_tfrecords_feature(
            feature_data)
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
        serialized_example = example.SerializePartialToString()
        return image, serialized_example
