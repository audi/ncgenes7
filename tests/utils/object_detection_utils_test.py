# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import skimage.filters
import tensorflow as tf

from ncgenes7.utils import object_detection_utils


class TestObjectDetectionUtils(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        np.random.seed(6547)
        self.num_classes = 4
        self.image_size = [200, 350]
        self.object_boxes_np = np.array(
            [[[0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.31, 0.41],
              [0.5, 0.6, 0.7, 0.8],
              [0.5, 0.6, 0.7, 0.82],
              [0, 0, 0, 0]],
             [[0.5, 0.2, 0.6, 0.3],
              [0.1, 0.2, 0.31, 0.41],
              [0.5, 0.2, 0.6, 0.3],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]],
            np.float32
        )
        self.object_boxes_np_image_frame = np.array(
            [[[20., 70., 60., 140.],
              [20., 70., 62., 143.5],
              [100., 210., 140., 280.],
              [100., 210., 140., 287.],
              [0., 0., 0., 0.]],
             [[100., 70., 120., 105.],
              [20., 70., 62., 143.5],
              [100., 70., 120., 105.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]]],
            np.float32
        )
        self.num_object_detections_np = np.array(
            [4, 3])
        self.object_classes_np = np.array(
            [[0, 0, 2, 2, 0],
             [1, 0, 3, 0, 0]], np.int32)
        self.object_scores_np = np.array(
            [[0.5, 0.6, 0.7, 0.2, 0],
             [0.7, 0.6, 0.1, 0, 0]], np.float32)
        self.object_instance_ids_np = np.array(
            [[0, 1, 2, 10, -1],
             [10, 5, 1, -1, -1]]
        )

        self.num_object_detections = tf.placeholder(tf.int32, [None])
        self.object_boxes = tf.placeholder(tf.float32, [None, None, 4])
        self.object_scores = tf.placeholder(tf.float32, [None, None])
        self.object_classes = tf.placeholder(tf.int32, [None, None])
        self.object_instance_ids = tf.placeholder(tf.int32, [None, None])

        self.object_boxes_nms_must = np.array(
            [[[0.5, 0.6, 0.7, 0.8],
              [0.1, 0.2, 0.31, 0.41],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]],
             [[0.5, 0.2, 0.6, 0.3],
              [0.1, 0.2, 0.31, 0.41],
              [0.5, 0.2, 0.6, 0.3],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]],
            np.float32)

        self.object_scores_nms_must = np.array(
            [[0.7, 0.6, 0.0, 0.0, 0],
             [0.7, 0.6, 0.1, 0, 0]], np.float32)
        self.object_nms_classes_nms_must = np.array(
            [[2, 0, 0, 0, 0],
             [1, 0, 3, 0, 0]], np.int32)
        self.object_instance_ids_nms_must = np.array(
            [[2, 1, 0, 0, 0],
             [10, 5, 1, 0, 0]]
        )
        self.num_object_detections_nms = np.array([2, 3])

        self.iou_threshold = 0.5
        self.max_size_per_class = 50
        self.max_total_size = 60

    def test_normalize_bbox(self):
        self.assertAllClose(
            self.object_boxes_np,
            self.evaluate(object_detection_utils.normalize_bbox(
                tf.constant(self.object_boxes_np_image_frame),
                self.image_size)))

    def test_normalize_bbox_np(self):
        self.assertAllClose(
            self.object_boxes_np,
            object_detection_utils.normalize_bbox_np(
                self.object_boxes_np_image_frame, self.image_size))

    def test_local_to_image_coordinates(self):
        self.assertAllClose(
            self.object_boxes_np_image_frame,
            self.evaluate(object_detection_utils.local_to_image_coordinates(
                tf.constant(self.object_boxes_np),
                self.image_size)))

    def test_local_to_image_coordinates_np(self):
        self.assertAllClose(
            self.object_boxes_np_image_frame,
            object_detection_utils.local_to_image_coordinates_np(
                self.object_boxes_np,
                self.image_size))

    def test_multiclass_non_max_suppression_np(self):
        for i in range(2):
            (object_boxes_nms, object_scores_nms, object_classes_nms,
             object_instance_nms_ids
             ) = object_detection_utils.multiclass_non_max_suppression_np(
                object_boxes=self.object_boxes_np[i],
                object_scores=self.object_scores_np[i],
                object_classes=self.object_classes_np[i],
                num_classes=self.num_classes,
                iou_threshold=self.iou_threshold,
                score_thresh=0,
                instance_ids=self.object_instance_ids_np[i],
            )
            num_objects = self.num_object_detections_nms[i]
            self.assertAllClose(self.object_boxes_nms_must[i][:num_objects],
                                object_boxes_nms)
            self.assertAllClose(self.object_scores_nms_must[i][:num_objects],
                                object_scores_nms)
            self.assertAllClose(
                self.object_nms_classes_nms_must[i][:num_objects],
                object_classes_nms)
            self.assertAllClose(
                self.object_instance_ids_nms_must[i][:num_objects],
                object_instance_nms_ids)

    def test_batch_multiclass_non_max_suppression(self):
        (object_boxes_nms, object_scores_nms, object_classes_nms,
         num_objects_nms, additional_fields_nms
         ) = object_detection_utils.batch_multiclass_non_max_suppression(
            object_boxes=self.object_boxes,
            object_scores=self.object_scores,
            object_classes=self.object_classes,
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            object_instance_ids=self.object_instance_ids,
        )
        object_instance_nms_ids = additional_fields_nms["object_instance_ids"]

        feed_dict = {
            self.object_boxes: self.object_boxes_np,
            self.object_scores: self.object_scores_np,
            self.object_classes: self.object_classes_np,
            self.object_instance_ids:
                self.object_instance_ids_np,
            self.num_object_detections: self.num_object_detections_np,
        }
        with self.test_session() as sess:
            (object_boxes_nms_eval, object_scores_nms_eval,
             object_classes_nms_eval, num_objects_nms_eval,
             object_instance_ids_nms_eval) = sess.run(
                [object_boxes_nms, object_scores_nms, object_classes_nms,
                 num_objects_nms, object_instance_nms_ids],
                feed_dict=feed_dict,
            )

        self.assertAllClose(self.object_boxes_nms_must,
                            object_boxes_nms_eval)
        self.assertAllClose(self.object_scores_nms_must,
                            object_scores_nms_eval)
        self.assertAllClose(self.object_nms_classes_nms_must,
                            object_classes_nms_eval)
        self.assertAllClose(self.object_instance_ids_nms_must,
                            object_instance_ids_nms_eval)

    def test_crop_keypoints_to_boxes(self):
        boxes_np = np.array([
            [[0, 0, 10, 20],
             [5, 7, 15, 17]],
            [[1, 0, 10, 5],
             [0, 0, 0, 0]]
        ], np.float32)
        keypoints_np = np.array([
            [[[5, 10],
              [5, 20],
              [7, 25],
              [11, 1],
              [0, 5]],
             [[5, 15],
              [0, 0],
              [6, 10],
              [15, 17],
              [0, 0]]],
            [[[9, 4],
              [5, 1],
              [1, 9],
              [5, 4],
              [1, 5]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        keypoints_must = np.array([
            [[[5, 10],
              [5, 20],
              [0, 0],
              [0, 0],
              [0, 5]],
             [[5, 15],
              [0, 0],
              [6, 10],
              [15, 17],
              [0, 0]]],
            [[[9, 4],
              [5, 1],
              [0, 0],
              [5, 4],
              [1, 5]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        keypoints_mask_must = np.array([
            [[1, 1, 0, 0, 1],
             [1, 0, 1, 1, 0]],
            [[1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1]],
        ])
        keypoints_eval, mask_eval = self.evaluate(
            object_detection_utils.crop_keypoints_to_boxes(
                keypoints=tf.constant(keypoints_np), boxes=tf.constant(boxes_np)
            ))
        self.assertAllClose(keypoints_must,
                            keypoints_eval)
        self.assertAllClose(keypoints_mask_must,
                            mask_eval)

    def test_encode_keypoints_to_boxes(self):
        boxes_np = np.array([
            [[0, 0, 10, 20],
             [5, 7, 15, 17]],
            [[1, 0, 10, 5],
             [0, 0, 0, 0]]
        ], np.float32)
        keypoints_np = np.array([
            [[[5, 10],
              [5, 20],
              [0, 0],
              [0, 0],
              [0, 5]],
             [[5, 15],
              [0, 0],
              [6, 10],
              [15, 17],
              [0, 0]]],
            [[[9, 4],
              [5, 1],
              [0, 0],
              [5, 4],
              [1, 5]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        keypoints_must = np.array([
            [[[0.5, 0.5],
              [0.5, 1.0],
              [0, 0],
              [0, 0],
              [0, 0.25]],
             [[0, 0.8],
              [0, 0],
              [0.1, 0.3],
              [1, 1],
              [0, 0]]],
            [[[8 / 9, 4 / 5],
              [4 / 9, 1 / 5],
              [0, 0],
              [4 / 9, 4 / 5],
              [0, 1]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        result = self.evaluate(object_detection_utils.encode_keypoints_to_boxes(
            keypoints=tf.constant(keypoints_np), boxes=tf.constant(boxes_np)
        ))
        self.assertAllClose(keypoints_must,
                            result)

    def test_decode_keypoints_from_boxes(self):
        boxes_np = np.array([
            [[0, 0, 10, 20],
             [5, 7, 15, 17]],
            [[1, 0, 10, 5],
             [0, 0, 0, 0]]
        ], np.float32)
        keypoints_in_boxes = np.array([
            [[[0.5, 0.5],
              [0.5, 1.0],
              [0, 0],
              [0, 0],
              [0, 0.25]],
             [[0, 0.8],
              [0, 0],
              [0.1, 0.3],
              [1, 1],
              [0, 0]]],
            [[[8 / 9, 4 / 5],
              [4 / 9, 1 / 5],
              [0, 0],
              [4 / 9, 4 / 5],
              [0, 1]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        keypoints_must = np.array([
            [[[5, 10],
              [5, 20],
              [0, 0],
              [0, 0],
              [0, 5]],
             [[5, 15],
              [5, 7],
              [6, 10],
              [15, 17],
              [5, 7]]],
            [[[9, 4],
              [5, 1],
              [1, 0],
              [5, 4],
              [1, 5]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        result = self.evaluate(
            object_detection_utils.decode_keypoints_from_boxes(
                keypoints=tf.constant(keypoints_in_boxes),
                boxes=tf.constant(boxes_np)
            ))
        self.assertAllClose(keypoints_must,
                            result)

    @parameterized.parameters(
        {"keypoints_kernel_size": 1},
        {"keypoints_kernel_size": 3},
        {"keypoints_kernel_size": 3, "with_keypoints_masks": False},
        {"keypoints_kernel_size": 5})
    def test_create_keypoints_heatmaps(self, keypoints_kernel_size,
                                       with_keypoints_masks=True):
        keypoints_np = np.array([
            [0, 0],
            [1, 1],
            [0, 0],
            [0.5, 0.5],
            [0.35, 0.4],
            [0, 0]
        ], np.float32)
        if with_keypoints_masks:
            keypoints_masks_np = np.array([
                1, 1, 0, 1, 1, 0
            ], np.bool)
        else:
            keypoints_masks_np = None
        heatmaps_image_shape_np = np.array([21, 27], np.int32)

        keypoints = tf.placeholder(tf.float32, [None, 2])
        if with_keypoints_masks:
            keypoints_masks = tf.placeholder(tf.bool, [None])
        else:
            keypoints_masks = None

        heatmaps_image_shape = tf.placeholder(tf.int32, [2])

        result = object_detection_utils.create_keypoints_heatmaps(
            keypoints, heatmaps_image_shape, keypoints_masks,
            keypoints_kernel_size)

        with self.test_session() as sess:
            if with_keypoints_masks:
                feed_dict = {keypoints: keypoints_np,
                             keypoints_masks: keypoints_masks_np,
                             heatmaps_image_shape: heatmaps_image_shape_np}
            else:
                feed_dict = {keypoints: keypoints_np,
                             heatmaps_image_shape: heatmaps_image_shape_np}
            result_eval = sess.run(result, feed_dict=feed_dict)

        num_keypoints = keypoints_np.shape[0]
        result_must = []

        for i in range(num_keypoints):
            result_keypoint = np.zeros(heatmaps_image_shape_np, np.float32)
            if with_keypoints_masks:
                draw_keypoint = keypoints_masks_np[i]
            else:
                draw_keypoint = True

            if not draw_keypoint:
                result_must.append(result_keypoint)
                continue

            center_coord = (
                    np.ceil(keypoints_np[i] * heatmaps_image_shape_np) - 1)
            center_coord = np.clip(center_coord, 0, heatmaps_image_shape_np - 1
                                   ).astype(np.int32)
            result_keypoint[center_coord[0], center_coord[1]] = 1
            if keypoints_kernel_size > 1:
                gaussian_filtered = skimage.filters.gaussian(
                    result_keypoint, sigma=(keypoints_kernel_size // 2),
                    mode="mirror",
                    truncate=2)
                result_keypoint = gaussian_filtered / gaussian_filtered.max()
            result_must.append(result_keypoint)
        result_must = np.stack(result_must, 0)
        self.assertAllClose(result_must,
                            result_eval)

    @parameterized.parameters({"smooth_size": 1, "keypoints_kernel_size": 1},
                              {"smooth_size": 3, "keypoints_kernel_size": 3},
                              {"smooth_size": 1, "keypoints_kernel_size": 5})
    def test_extract_keypoints_from_heatmaps(self, smooth_size,
                                             keypoints_kernel_size):
        keypoints_np = np.array([
            [[0, 0],
             [0.8, 1],
             [0, 0]],
            [[0.5, 0.5],
             [0.25, 0.4],
             [0, 0]]
        ], np.float32)
        keypoints_masks_np = np.array([
            0, 1, 0, 1, 1, 1
        ], np.bool)

        keypoints_flat_np = keypoints_np.reshape([6, 2])
        heatmaps_image_shape_np = np.array([21, 27], np.int32)

        keypoints = tf.placeholder(tf.float32, [None, 2])
        heatmaps_image_shape = tf.placeholder(tf.int32, [2])
        keypoints_heatmaps = object_detection_utils.create_keypoints_heatmaps(
            keypoints, heatmaps_image_shape,
            keypoints_masks_np,
            keypoints_kernel_size=keypoints_kernel_size)

        keypoints_heatmaps_reshaped = tf.reshape(
            keypoints_heatmaps,
            tf.concat([[-1, 3], tf.shape(keypoints_heatmaps)[1:]], 0))
        keypoints_heatmaps_reshaped_tr = tf.transpose(
            keypoints_heatmaps_reshaped, [0, 2, 3, 1])

        result = object_detection_utils.extract_keypoints_from_heatmaps(
            keypoints_heatmaps_reshaped_tr, smooth_size=smooth_size)
        with self.test_session() as sess:
            feed_dict = {keypoints: keypoints_flat_np,
                         heatmaps_image_shape: heatmaps_image_shape_np}
            keypoints_result_eval, scores_res_eval = sess.run(
                result, feed_dict=feed_dict)

        if smooth_size == 1:
            self.assertAllClose(
                keypoints_masks_np.reshape([2, 3]).astype(np.float32),
                scores_res_eval)
        self.assertAllClose(keypoints_np,
                            keypoints_result_eval, atol=1 / 26)

    def test_decode_instance_masks_to_image(self):
        image_sizes = tf.placeholder(tf.int32, (None, 2))
        instance_masks = tf.placeholder(tf.float32, (None, None, None, None))
        boxes = tf.placeholder(tf.float32, (None, None, 4))

        _ = object_detection_utils.decode_instance_masks_to_image(
            instance_masks, boxes, image_sizes)
