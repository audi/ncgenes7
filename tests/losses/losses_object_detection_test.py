# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.losses.object_detection import ClassificationMatchLoss
from ncgenes7.losses.object_detection import FasterRCNNFirstStageLoss
from ncgenes7.losses.object_detection import FasterRCNNSecondStageLoss
from ncgenes7.losses.object_detection import KeypointsToHeatmapsLoss
from ncgenes7.losses.object_detection import OHEMLoss


class TestFasterRCNNFirstStageLoss(parameterized.TestCase,
                                   tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.num_anchors = 65
        self.batch_size = 4

        self.images_np = np.random.rand(self.batch_size, 10, 15, 3)
        self.rpn_box_encodings_np = np.random.rand(
            self.batch_size, self.num_anchors, 4)
        self.rpn_objectness_predictions_with_background_np = (
            np.random.rand(self.batch_size, self.num_anchors, 2))
        self.anchors_np = np.random.rand(self.num_anchors, 4)
        self.groundtruth_object_boxes_np = np.random.rand(
            self.batch_size, 2, 4)
        self.groundtruth_object_weights_np = np.random.rand(
            self.batch_size, 2)

    @parameterized.parameters({"with_object_weights": True},
                              {"with_object_weights": False})
    def test_process(self, with_object_weights):
        loss = FasterRCNNFirstStageLoss(
            positive_balance_fraction=0.25, minibatch_size=128).build()
        self.assertEmpty(tf.get_default_graph().get_operations())
        self._get_placeholders()
        groundtruth_object_weights = (
            self.groundtruth_object_weights if with_object_weights
            else None)
        result = loss.process(
            images=self.images,
            rpn_box_encodings=self.rpn_box_encodings,
            rpn_objectness_predictions_with_background=
            self.rpn_objectness_predictions_with_background,
            anchors=self.anchors,
            groundtruth_object_boxes=self.groundtruth_object_boxes,
            groundtruth_object_weights=groundtruth_object_weights,
        )

        self.assertSetEqual(set(loss.generated_keys),
                            set(result.keys()))
        self.assertEmpty(result["loss_rpn_localization"].shape)
        self.assertEmpty(result["loss_rpn_objectness"].shape)

        feed_dict = {
            self.images: self.images_np,
            self.rpn_box_encodings: self.rpn_box_encodings_np,
            self.rpn_objectness_predictions_with_background:
                self.rpn_objectness_predictions_with_background_np,
            self.anchors: self.anchors_np,
            self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
            self.groundtruth_object_weights:
                self.groundtruth_object_weights_np
        }
        if with_object_weights:
            feed_dict[self.groundtruth_object_weights] = (
                self.groundtruth_object_weights_np)

        with self.test_session() as sess:
            _ = sess.run(result, feed_dict=feed_dict)

    def _get_placeholders(self):
        self.images = tf.placeholder(tf.float32, [None, 10, 15, 3])
        self.rpn_box_encodings = tf.placeholder(tf.float32,
                                                [None, self.num_anchors, 4])
        self.rpn_objectness_predictions_with_background = tf.placeholder(
            tf.float32, [None, self.num_anchors, 2])
        self.anchors = tf.placeholder(tf.float32,
                                      [self.num_anchors, 4])
        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_weights = tf.placeholder(
            tf.float32, [None, None])


class TestFasterRCNNSecondStageLoss(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.batch_size = 4
        self.num_classes = 4
        self.max_num_proposals_np = 10

        self.num_proposals_np = np.random.randint(0, self.max_num_proposals_np,
                                                  size=[self.batch_size])

        self.images_np = np.random.rand(self.batch_size, 10, 15, 3)
        self.refined_box_encodings_np = np.random.rand(
            self.batch_size, self.max_num_proposals_np, self.num_classes, 4)
        self.class_predictions_with_background_np = (
            np.random.rand(self.batch_size, self.max_num_proposals_np,
                           self.num_classes + 1))
        self.proposal_boxes_np = np.random.rand(
            self.batch_size, self.max_num_proposals_np, 4)
        self.groundtruth_object_boxes_np = np.random.rand(
            self.batch_size, 2, 4)
        self.groundtruth_object_classes_np = np.random.randint(
            1, self.num_classes + 1, size=[self.batch_size, 2])
        self.groundtruth_object_weights_np = np.random.rand(
            self.batch_size, 2)

    @parameterized.parameters(
        {"with_object_weights": True},
        {"with_object_weights": False}
    )
    def test_process(self, with_object_weights):
        loss = FasterRCNNSecondStageLoss(
            num_classes=self.num_classes).build()
        self.assertEmpty(tf.get_default_graph().get_operations())
        self._get_placeholders()

        groundtruth_object_weights = (
            self.groundtruth_object_weights if with_object_weights
            else None)

        result = loss.process(
            images=self.images,
            refined_box_encodings=self.refined_box_encodings,
            class_predictions_with_background=
            self.class_predictions_with_background,
            proposal_boxes=self.proposal_boxes,
            num_proposals=self.num_proposals,
            groundtruth_object_boxes=self.groundtruth_object_boxes,
            groundtruth_object_classes=self.groundtruth_object_classes,
            groundtruth_object_weights=groundtruth_object_weights,
        )

        self.assertSetEqual(set(loss.generated_keys),
                            set(result.keys()))
        self.assertEmpty(result["loss_second_stage_localization"].shape)
        self.assertEmpty(result["loss_second_stage_classification"].shape)

        feed_dict = {
            self.images: self.images_np,
            self.num_proposals: self.num_proposals_np,
            self.refined_box_encodings: self.refined_box_encodings_np,
            self.class_predictions_with_background:
                self.class_predictions_with_background_np,
            self.proposal_boxes: self.proposal_boxes_np,
            self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
            self.groundtruth_object_classes: self.groundtruth_object_classes_np,
        }

        if with_object_weights:
            feed_dict[self.groundtruth_object_weights] = (
                self.groundtruth_object_weights_np)

        with self.test_session() as sess:
            _ = sess.run(result, feed_dict=feed_dict)

    def _get_placeholders(self):
        self.images = tf.placeholder(tf.float32, [None, 10, 15, 3])
        self.max_num_proposals = tf.placeholder(tf.int32, [])
        self.num_proposals = tf.placeholder(tf.int32, [None])
        self.refined_box_encodings = tf.placeholder(
            tf.float32, [None, None, self.num_classes, 4])
        self.class_predictions_with_background = tf.placeholder(
            tf.float32, [None, None, self.num_classes + 1])
        self.proposal_boxes = tf.placeholder(
            tf.float32, [self.batch_size, None, 4])

        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_classes = tf.placeholder(
            tf.int32, [None, None])
        self.groundtruth_object_weights = tf.placeholder(
            tf.float32, [None, None])


class TestOHEMLoss(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.batch_size = 4
        self.num_classes = 4
        self.max_num_proposals_np = 10

        self.num_proposals_np = np.random.randint(0, self.max_num_proposals_np,
                                                  size=[self.batch_size])

        self.images_np = np.random.rand(self.batch_size, 10, 15, 3)
        self.refined_box_encodings_np = np.random.rand(
            self.batch_size, self.max_num_proposals_np, self.num_classes, 4)
        self.class_predictions_with_background_np = (
            np.random.rand(self.batch_size, self.max_num_proposals_np,
                           self.num_classes + 1))
        self.proposal_boxes_np = np.random.rand(
            self.batch_size, self.max_num_proposals_np, 4)
        self.groundtruth_object_boxes_np = np.random.rand(
            self.batch_size, 2, 4)
        self.groundtruth_object_classes_np = np.random.randint(
            1, self.num_classes + 1, size=[self.batch_size, 2])
        self.groundtruth_object_weights_np = np.random.rand(
            self.batch_size, 2)

    @parameterized.parameters(
        {"with_object_weights": True},
        {"with_object_weights": False}
    )
    def test_process(self, with_object_weights):
        loss = OHEMLoss(
            num_classes=self.num_classes).build()
        self.assertEmpty(tf.get_default_graph().get_operations())
        self._get_placeholders()

        groundtruth_object_weights = (
            self.groundtruth_object_weights if with_object_weights
            else None)

        result = loss.process(
            images=self.images,
            refined_box_encodings=self.refined_box_encodings,
            class_predictions_with_background=
            self.class_predictions_with_background,
            proposal_boxes=self.proposal_boxes,
            num_proposals=self.num_proposals,
            groundtruth_object_boxes=self.groundtruth_object_boxes,
            groundtruth_object_classes=self.groundtruth_object_classes,
            groundtruth_object_weights=groundtruth_object_weights,
        )

        self.assertSetEqual(set(loss.generated_keys),
                            set(result.keys()))
        self.assertEmpty(result["loss_second_stage_localization"].shape)
        self.assertEmpty(result["loss_second_stage_classification"].shape)

        feed_dict = {
            self.images: self.images_np,
            self.num_proposals: self.num_proposals_np,
            self.refined_box_encodings: self.refined_box_encodings_np,
            self.class_predictions_with_background:
                self.class_predictions_with_background_np,
            self.proposal_boxes: self.proposal_boxes_np,
            self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
            self.groundtruth_object_classes: self.groundtruth_object_classes_np,
        }

        if with_object_weights:
            feed_dict[self.groundtruth_object_weights] = (
                self.groundtruth_object_weights_np)

        with self.test_session() as sess:
            _ = sess.run(result, feed_dict=feed_dict)

    def _get_placeholders(self):
        self.images = tf.placeholder(tf.float32, [None, 10, 15, 3])
        self.max_num_proposals = tf.placeholder(tf.int32, [])
        self.num_proposals = tf.placeholder(tf.int32, [None])
        self.refined_box_encodings = tf.placeholder(
            tf.float32, [None, None, self.num_classes, 4])
        self.class_predictions_with_background = tf.placeholder(
            tf.float32, [None, None, self.num_classes + 1])
        self.proposal_boxes = tf.placeholder(
            tf.float32, [self.batch_size, None, 4])

        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_classes = tf.placeholder(
            tf.int32, [None, None])
        self.groundtruth_object_weights = tf.placeholder(
            tf.float32, [None, None])


class TestKeypointsToHeatmapsLoss(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.num_keypoints = 5
        self.batch_size = 2
        self.num_detections = 3
        self.num_gt_objects = 2

        self.detection_boxes_np = np.array([
            [[0, 0, 0.1, 0.1],
             [0.2, 0.3, 0.4, 0.6],
             [0, 0, 0, 0]],
            [[0.6, 0.7, 0.8, 0.9],
             [0.25, 0.35, 0.38, 0.4],
             [0.1, 0.3, 0.2, 0.4]]
        ], np.float32)
        self.groundtruth_object_boxes_np = np.array([
            [[0.7, 0.6, 0.8, 0.9],
             [0.01, 0.0, 0.1, 0.11]],
            [[0.24, 0.37, 0.4, 0.4],
             [0, 0, 0, 0]]
        ], np.float32)
        self.groundtruth_object_keypoints_np = np.array([
            [[[0.8, 0.65],
              [0.75, 0.9],
              [0.7, 0.6],
              [0.72, 0.8],
              [0.8, 0.9]],
             [[0.075, 0.11],
              [0.05, 0.05],
              [0.04, 0.06],
              [0.1, 0.1],
              [0, 0]]],
            [[[0.3, 0.37],
              [0.4, 0.4],
              [0.25, 0.3],
              [0.27, 0.5],
              [0.26, 0.36]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        self.detection_keypoints_heatmaps_np = np.random.rand(
            self.batch_size, self.num_detections, 10, 11, self.num_keypoints)
        self.num_object_detections_np = np.array([2, 3], np.int32)

        self.detection_boxes = tf.placeholder(tf.float32, [None, None, 4])
        self.num_object_detections = tf.placeholder(tf.int32, [None])
        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_keypoints = tf.placeholder(
            tf.float32, [None, None, self.num_keypoints, 2])
        self.detection_keypoints_heatmaps = tf.placeholder(
            tf.float32, [None, None, None, None, self.num_keypoints]
        )

    @parameterized.parameters(
        {"loss_name": "WeightedL2LocalizationLoss"},
        {"loss_name": "SigmoidFocalClassificationLoss"},
        {"loss_name": "WeightedSigmoidClassificationLoss"})
    def test_predict(self, loss_name):
        loss = KeypointsToHeatmapsLoss(
            num_keypoints=self.num_keypoints,
            heatmaps_loss_name=loss_name
        ).build()
        result = loss.process(
            detection_object_boxes=self.detection_boxes,
            num_object_detections=self.num_object_detections,
            groundtruth_object_boxes=self.groundtruth_object_boxes,
            groundtruth_object_keypoints=self.groundtruth_object_keypoints,
            detection_object_keypoints_heatmaps=
            self.detection_keypoints_heatmaps)
        with self.test_session() as sess:
            result_eval = sess.run(result, feed_dict={
                self.num_object_detections: self.num_object_detections_np,
                self.detection_boxes: self.detection_boxes_np,
                self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
                self.groundtruth_object_keypoints:
                    self.groundtruth_object_keypoints_np,
                self.detection_keypoints_heatmaps:
                    self.detection_keypoints_heatmaps_np})
        self.assertSetEqual(set(loss.generated_keys),
                            set(result_eval))


class TestClassificationMatchLoss(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.batch_size = 2
        self.num_classes = 2
        self.max_num_proposals = 2

        self.num_proposals_np = np.array([3, 1], np.float32)

        self.class_predictions_with_background_np = (
            np.random.rand(self.batch_size, 3,
                           self.num_classes + 1))
        self.proposal_boxes_np = np.array([
            [[0, 0, 0.5, 0.5],
             [0.2, 0.2, 0.3, 0.4],
             [0.4, 0.7, 0.5, 0.8]],
            [[0.6, 0.7, 0.8, 0.9],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], np.float32)
        self.groundtruth_object_boxes_np = np.array([
            [[0.4, 0.7, 0.5, 0.8],
             [0.0, 0.0, 0.55, 0.55]],
            [[0.6, 0.7, 0.8, 0.9],
             [0, 0, 0, 0]]
        ], np.float32)
        self.groundtruth_object_classes_np = np.array([
            [0, 1],
            [2, 0]
        ])
        self.groundtruth_object_weights_np = np.array([
            [1, 0],
            [0, 0]
        ])

    @parameterized.parameters(
        {"loss_name": "WeightedSoftmaxClassificationLoss",
         "with_object_weights": True},
        {"loss_name": "WeightedSoftmaxClassificationLoss",
         "with_object_weights": False},
        {"loss_name": "WeightedSigmoidClassificationLoss",
         "with_object_weights": True},
        {"loss_name": "WeightedSigmoidClassificationLoss",
         "with_object_weights": False},
        {"loss_name": "SigmoidFocalClassificationLoss",
         "with_object_weights": True},
        {"loss_name": "SigmoidFocalClassificationLoss",
         "with_object_weights": False},
    )
    def test_process(self, with_object_weights,
                     loss_name):
        loss = ClassificationMatchLoss(loss_name=loss_name).build()
        self.assertEmpty(tf.get_default_graph().get_operations())
        self._get_placeholders()

        groundtruth_object_weights = (
            self.groundtruth_object_weights if with_object_weights
            else None)

        result = loss.process(
            class_predictions_with_background=
            self.class_predictions_with_background,
            detection_object_boxes=self.proposal_boxes,
            num_object_detections=self.num_proposals,
            groundtruth_object_boxes=self.groundtruth_object_boxes,
            groundtruth_object_classes=self.groundtruth_object_classes,
            groundtruth_object_weights=groundtruth_object_weights,
        )

        self.assertSetEqual(set(loss.generated_keys),
                            set(result.keys()))
        self.assertEmpty(result["loss_classification_match"].shape)

        feed_dict = {
            self.num_proposals: self.num_proposals_np,
            self.class_predictions_with_background:
                self.class_predictions_with_background_np,
            self.proposal_boxes: self.proposal_boxes_np,
            self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
            self.groundtruth_object_classes: self.groundtruth_object_classes_np,
        }

        if with_object_weights:
            feed_dict[self.groundtruth_object_weights] = (
                self.groundtruth_object_weights_np)

        with self.test_session() as sess:
            _ = sess.run(result, feed_dict=feed_dict)

    def _get_placeholders(self):
        self.max_num_proposals = tf.placeholder(tf.int32, [])
        self.num_proposals = tf.placeholder(tf.int32, [None])
        self.refined_box_encodings = tf.placeholder(
            tf.float32, [None, None, self.num_classes, 4])
        self.class_predictions_with_background = tf.placeholder(
            tf.float32, [None, None, self.num_classes + 1])
        self.proposal_boxes = tf.placeholder(
            tf.float32, [self.batch_size, None, 4])

        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_classes = tf.placeholder(
            tf.int32, [None, None])
        self.groundtruth_object_weights = tf.placeholder(
            tf.float32, [None, None])
