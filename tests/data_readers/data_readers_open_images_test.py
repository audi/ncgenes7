# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow as tf

from ncgenes7.data_readers.open_images import OpenImagesObjectsReader


class TestOpenImagesDataFeeder(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        np.random.seed(546)

        self.input_dir = self.get_temp_dir()
        self.save_dir = os.path.join(self.input_dir, "save_dir")
        self.image_fname = os.path.join(
            self.input_dir, "inputs", "images", "000595fe6fee6369.jpg")

        annotation_columns = [
            "ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax",
            "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
            "IsDepiction", "IsInside"
        ]
        self.annotations = pd.DataFrame(
            [["ffff21932da3ed01", "freeform", "/m/0120dh", 1,
              0.5, 0.9, 0.1, 0.2, 0, 0, 0, 0, 0],
             ["000595fe6fee6369", "freeform", "/m/0120dh", 1,
              0.35, 0.75, 0.28, 0.87, 0, 0, 0, 0, 0],
             ["000595fe6fee6369", "freeform", "/m/011q46kg", 1,
              0.01, 0.68, 0.11, 0.43, 0, 0, 0, 0, 0],
             ["0001eeaf4aed83f9", "freeform", "/m/02xwb", 1,
              0.1, 0.5, 0.2, 0.9, 0, 0, 0, 0, 0]],
            columns=annotation_columns
        )

        self.class_descriptions = pd.DataFrame(
            [("/m/011k07", "bicycle"),
             ("/m/011q46kg", "motorcycle"),
             ("/m/0120dh", "person"),
             ("/m/02xwb", "traffic light")]
        )

        self.object_data_must = {
            "object_classes": np.array([3, 2]),
            "object_instance_ids": np.array([-1, -1]),
            "object_boxes": np.array([
                [0.28, 0.35, 0.87, 0.75],
                [0.11, 0.01, 0.43, 0.68]
            ]),
        }

        self.raw_object_data_saved_must = [
            {
                "bbox": {
                    "ymin": 5.6,
                    "xmin": 10.5,
                    "ymax": 17.4,
                    "xmax": 22.5,
                    "h": 11.8,
                    "w": 12.0,
                },
                "class_label": 3,
                "id": -1,
                "score": 1.0,
            },
            {
                "bbox": {
                    "ymin": 2.2,
                    "xmin": 0.3,
                    "ymax": 8.6,
                    "xmax": 20.4,
                    "h": 6.4,
                    "w": 20.1,
                },
                "class_label": 2,
                "id": -1,
                "score": 1.0,
            }
        ]

        os.mkdir(os.path.join(self.input_dir, "annotations"))
        os.mkdir(os.path.join(self.input_dir, "inputs"))
        self.file_name_annotations = os.path.join(
            self.input_dir, "annotations", "annotations-bbox.csv")
        self.file_name_class_descriptions = os.path.join(
            self.input_dir, "annotations", "class-descriptions-boxable.csv")

        self.annotations.to_csv(self.file_name_annotations, index=False)
        self.class_descriptions.to_csv(
            self.file_name_class_descriptions, index=False, header=False)

    def test_read_element_from_file_names(self):
        data_feeder = OpenImagesObjectsReader(
            file_name_annotations=self.file_name_annotations,
            file_name_class_descriptions=self.file_name_class_descriptions,
        ).build()

        result = data_feeder.read(images=self.image_fname)
        result_objects = {k: v for k, v in result.items()
                          if k in self.object_data_must}
        self.assertAllClose(self.object_data_must,
                            result_objects)
