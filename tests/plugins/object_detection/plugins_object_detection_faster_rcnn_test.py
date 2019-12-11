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

from ncgenes7.plugins.object_detection.faster_rcnn import (
    FasterRCNNFirstStagePlugin)
from ncgenes7.plugins.object_detection.faster_rcnn import (
    FasterRCNNSecondStagePlugin)
from ncgenes7.plugins.object_detection.faster_rcnn import (
    ProposalsSampler)
from ncgenes7.plugins.object_detection.faster_rcnn import (
    ROIPoolingPlugin)


class TestFasterRCNNFirstPluginStage(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.images_np = np.random.rand(3, 10, 15, 1)
        self.feature_maps_np = np.random.rand(3, 8, 10, 16)
        gt_boxes_yx_min = np.random.rand(3, 5, 2)
        gt_boxes_hw = np.random.rand(3, 5, 2)
        self.groundtruth_object_boxes_np = np.concatenate(
            [gt_boxes_yx_min, gt_boxes_yx_min + gt_boxes_hw], -1)
        self.groundtruth_object_classes_np = np.random.randint(
            0, 5, size=[3, 5])
        self.groundtruth_object_weights_np = np.random.rand(3, 5)

        self.images = tf.placeholder(tf.float32, [None, None, None, 1])
        self.feature_maps = tf.placeholder(tf.float32, [None, None, None, 16])
        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_classes = tf.placeholder(tf.int32, [None, None])
        self.groundtruth_object_weights = tf.placeholder(
            tf.float32, [None, None])

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = FasterRCNNFirstStagePlugin().build()
        self.assertEmpty(tf.get_default_graph().get_operations())

    @parameterized.parameters(
        {},
        {"with_gt": True, "fuse_sampler": True},
        {"with_gt": True, "fuse_sampler": False},
        {"relative_anchor_generator": True},
        {"with_gt": False, "clip_anchors_to_image": False},
        {"with_gt": False, "clip_anchors_to_image": True},
        {"with_groundtruth_object_weights": True},
        {"multi_processes": True},
        {"mode": "eval", "with_gt": False})
    def test_predict(self, mode='train',
                     fuse_sampler=True,
                     with_gt=True,
                     with_groundtruth_object_weights=False,
                     multi_processes=False,
                     clip_anchors_to_image=False,
                     relative_anchor_generator=False):
        if relative_anchor_generator:
            anchor_generator_config = {"relative": True}
        else:
            anchor_generator_config = None
        plugin = FasterRCNNFirstStagePlugin(
            fuse_sampler=fuse_sampler,
            anchor_generator_config=anchor_generator_config,
            clip_anchors_to_image=clip_anchors_to_image).build()
        plugin.mode = mode
        inputs = {"images": self.images,
                  "feature_maps": self.feature_maps}
        if with_gt:
            inputs["groundtruth_object_boxes"] = (
                self.groundtruth_object_boxes)
            inputs["groundtruth_object_classes"] = (
                self.groundtruth_object_classes)
        if with_groundtruth_object_weights:
            inputs["groundtruth_object_weights"] = (
                self.groundtruth_object_weights)

        if not fuse_sampler and with_gt:
            with self.assertRaises(ValueError):
                _ = plugin.predict(**inputs)
            return

        if mode == "train" and fuse_sampler and not with_gt:
            with self.assertRaises(RuntimeError):
                _ = plugin.predict(**inputs)
            return

        result = plugin.predict(**inputs)
        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))

        if multi_processes:
            trainable_vars = tf.trainable_variables()
            result = plugin.predict(**inputs)
            self.assertSetEqual(set(trainable_vars),
                                set(tf.trainable_variables()))
            plugin.reset_keras_layers()
            result = plugin.predict(**inputs)
            self.assertEqual(len(trainable_vars) * 2,
                             len(tf.trainable_variables()))

        feed_dict = {
            self.images: self.images_np,
            self.feature_maps: self.feature_maps_np,
        }
        if with_gt:
            feed_dict[self.groundtruth_object_boxes] = (
                self.groundtruth_object_boxes_np)
            feed_dict[self.groundtruth_object_classes] = (
                self.groundtruth_object_classes_np)
        if with_groundtruth_object_weights:
            feed_dict[self.groundtruth_object_weights] = (
                self.groundtruth_object_weights_np)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result_eval = sess.run(result, feed_dict=feed_dict)


class TestProposalsSampler(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.images_np = np.random.rand(3, 10, 15, 1)
        det_boxes_yx_min = np.random.rand(3, 10, 2)
        det_boxes_hw = np.random.rand(3, 10, 2)
        self.detection_object_boxes_np = np.concatenate(
            [det_boxes_yx_min, det_boxes_hw + det_boxes_hw], -1)
        self.detection_object_scores_np = np.random.rand(3, 10)
        self.num_object_detections_np = np.random.randint(1, 5, [3])

        gt_boxes_yx_min = np.random.rand(3, 5, 2)
        gt_boxes_hw = np.random.rand(3, 5, 2)
        self.groundtruth_object_boxes_np = np.concatenate(
            [gt_boxes_yx_min, gt_boxes_yx_min + gt_boxes_hw], -1)
        self.groundtruth_object_classes_np = np.random.randint(
            0, 5, size=[3, 5])
        self.groundtruth_object_weights_np = np.random.rand(3, 5)

        self.images = tf.placeholder(tf.float32, [None, None, None, 1])
        self.detection_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.detection_object_scores = tf.placeholder(
            tf.float32, [None, None])
        self.num_object_detections = tf.placeholder(
            tf.int32, [None])
        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_classes = tf.placeholder(tf.int32, [None, None])
        self.groundtruth_object_weights = tf.placeholder(
            tf.float32, [None, None])

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = ProposalsSampler().build()
        self.assertEmpty(tf.get_default_graph().get_operations())

    @parameterized.parameters(
        {},
        {"with_groundtruth_object_weights": True},
        {"multi_processes": True},
        {"mode": "eval"})
    def test_predict(self,
                     mode='train',
                     with_groundtruth_object_weights=False,
                     multi_processes=False):
        plugin = ProposalsSampler(
            sample_minibatch_size=6,
            positive_balance_fraction=0.25
        ).build()
        plugin.mode = mode
        inputs = {
            "images": self.images,
            "detection_object_boxes": self.detection_object_boxes,
            "detection_object_scores": self.detection_object_scores,
            "num_object_detections": self.num_object_detections,
            "groundtruth_object_boxes": self.groundtruth_object_boxes,
            "groundtruth_object_classes": self.groundtruth_object_classes,
        }
        if with_groundtruth_object_weights:
            inputs["groundtruth_object_weights"] = (
                self.groundtruth_object_weights)

        result = plugin.predict(**inputs)
        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))

        if multi_processes:
            trainable_vars = tf.trainable_variables()
            result = plugin.predict(**inputs)
            self.assertSetEqual(set(trainable_vars),
                                set(tf.trainable_variables()))
            plugin.reset_keras_layers()
            result = plugin.predict(**inputs)
            self.assertEqual(len(trainable_vars) * 2,
                             len(tf.trainable_variables()))

        feed_dict = {
            self.images: self.images_np,
            self.detection_object_boxes: self.detection_object_boxes_np,
            self.detection_object_scores: self.detection_object_scores_np,
            self.num_object_detections: self.num_object_detections_np,
            self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
            self.groundtruth_object_classes: self.groundtruth_object_classes_np
        }
        if with_groundtruth_object_weights:
            feed_dict[self.groundtruth_object_weights] = (
                self.groundtruth_object_weights_np)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result_eval = sess.run(result, feed_dict=feed_dict)

        if mode == "eval":
            result_must = {
                "detection_object_boxes": self.detection_object_boxes_np,
                "detection_object_scores": self.detection_object_scores_np,
                "num_object_detections": self.num_object_detections_np,
            }
            self.assertAllClose(result_must,
                                result_eval)


class TestROIPoolingPlugin(parameterized.TestCase,
                           tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.feature_maps_np = np.random.rand(3, 30, 30, 16)
        boxes_yx_min = np.random.rand(3, 5, 2)
        boxes_hw = np.random.rand(3, 5, 2)
        self.detection_object_boxes_np = np.concatenate(
            [boxes_yx_min, boxes_yx_min + boxes_hw], -1)

        self.feature_maps = tf.placeholder(tf.float32, [None, None, None, 16])
        self.detection_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])

    @parameterized.parameters({"squash_batch_and_detection_dims": False},
                              {"squash_batch_and_detection_dims": True})
    def test_predict(self, squash_batch_and_detection_dims):
        plugin = ROIPoolingPlugin(
            maxpool_kernel_size=3,
            maxpool_stride=3,
            initial_crop_size=18,
            squash_batch_and_detection_dims=squash_batch_and_detection_dims,
        ).build()
        result = plugin.predict(
            feature_maps=self.feature_maps,
            detection_object_boxes=self.detection_object_boxes)

        self.assertSetEqual({"feature_maps"},
                            set(result.keys()))

        shape_must = ([None, 6, 6, 16] if squash_batch_and_detection_dims
                      else [None, None, 6, 6, 16])
        self.assertListEqual(shape_must,
                             result["feature_maps"].get_shape().as_list())
        feed_dict = {
            self.feature_maps: self.feature_maps_np,
            self.detection_object_boxes: self.detection_object_boxes_np}
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result_eval = sess.run(result, feed_dict=feed_dict)

        shape_eval_must = ([15, 6, 6, 16] if squash_batch_and_detection_dims
                           else [3, 5, 6, 6, 16])
        self.assertListEqual(shape_eval_must,
                             list(result_eval["feature_maps"].shape))


class TestFasterRCNNSecondStagePlugin(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.num_classes = 10

        self.images_np = np.random.rand(3, 10, 15, 1)
        self.feature_maps_np = np.random.rand(3, 5, 10, 10, 16)
        boxes_yx_min = np.random.rand(3, 5, 2)
        boxes_hw = np.random.rand(3, 5, 2)
        self.proposal_boxes_np = np.concatenate(
            [boxes_yx_min, boxes_yx_min + boxes_hw], -1)

        self.num_proposals_np = np.array([5, 2, 0])

        self.images = tf.placeholder(tf.float32, [None, None, None, 1])
        self.feature_maps = tf.placeholder(tf.float32, [None, None, 10, 10, 16])
        self.proposal_boxes = tf.placeholder(tf.float32, [None, None, 4])
        self.num_proposals = tf.placeholder(tf.int32, [None])
        self.max_num_proposals = tf.placeholder(tf.int32, [])

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = FasterRCNNSecondStagePlugin(num_classes=10).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters({"mode": "train"},
                              {"mode": "eval"})
    def test_predict(self, mode):
        plugin = FasterRCNNSecondStagePlugin(num_classes=9).build()
        plugin.mode = mode

        inputs = {
            "images": self.images,
            "feature_maps": self.feature_maps,
            "proposal_boxes": self.proposal_boxes,
            "num_proposals": self.num_proposals,
        }

        result = plugin.predict(**inputs)
        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))

        feed_dict = {
            self.images: self.images_np,
            self.feature_maps: self.feature_maps_np,
            self.proposal_boxes: self.proposal_boxes_np,
            self.num_proposals: self.num_proposals_np,
        }
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result_eval = sess.run(result, feed_dict=feed_dict)


class TestFasterRCNNIntegration(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.batch_size = 3
        self.images_np = np.random.rand(self.batch_size, 10, 15, 1)
        self.rpn_feature_maps_np = np.random.rand(self.batch_size, 8, 10, 16)
        gt_boxes_yx_min = np.random.rand(self.batch_size, 5, 2)
        gt_boxes_hw = np.random.rand(self.batch_size, 5, 2)
        self.groundtruth_object_boxes_np = np.concatenate(
            [gt_boxes_yx_min, gt_boxes_yx_min + gt_boxes_hw], -1)
        self.groundtruth_object_classes_np = np.random.randint(
            0, 5, size=[3, 5])

        self.images = tf.placeholder(tf.float32, [None, None, None, 1])
        self.rpn_feature_maps = tf.placeholder(tf.float32,
                                               [None, None, None, 16])
        self.groundtruth_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.groundtruth_object_classes = tf.placeholder(tf.int32, [None, None])

    @parameterized.parameters(
        {"mode": "train", "clip_anchors_to_image": True},
        {"mode": "train", "clip_anchors_to_image": False},
        {"mode": "eval", "clip_anchors_to_image": True},
        {"mode": "eval", "clip_anchors_to_image": False},
        {"fuse_sampler": False, "mode": "train", "clip_anchors_to_image": True},
        {"fuse_sampler": False, "mode": "train",
         "clip_anchors_to_image": False},
        {"fuse_sampler": False, "mode": "eval", "clip_anchors_to_image": True},
        {"fuse_sampler": False, "mode": "eval", "clip_anchors_to_image": False}
    )
    def test_predict(self, mode="train", clip_anchors_to_image=False,
                     fuse_sampler=True):
        max_num_proposals = 32
        max_num_proposals_train = 16
        max_num_proposals_mode = (max_num_proposals_train if mode == "train"
                                  else max_num_proposals)

        first_stage = FasterRCNNFirstStagePlugin(
            anchor_generator_config={"relative": True},
            clip_anchors_to_image=clip_anchors_to_image,
            max_num_proposals=max_num_proposals,
            max_num_proposals_train=(max_num_proposals_train
                                     if fuse_sampler else None),
            fuse_sampler=fuse_sampler,
        ).build()
        if not fuse_sampler:
            sampler = ProposalsSampler(
                sample_minibatch_size=max_num_proposals_train).build()
            sampler.mode = mode
        else:
            sampler = None
        roi_plugin = ROIPoolingPlugin(
            maxpool_kernel_size=3,
            maxpool_stride=3,
            initial_crop_size=18).build()
        second_stage = FasterRCNNSecondStagePlugin(
            num_classes=10).build()
        second_stage_feature_extractor = tf.keras.layers.Conv2D(
            64, 3, padding="same")

        first_stage.mode = mode
        roi_plugin.mode = mode
        second_stage.mode = mode

        if fuse_sampler:
            first_stage_output = first_stage.predict(
                images=self.images,
                feature_maps=self.rpn_feature_maps,
                groundtruth_object_boxes=self.groundtruth_object_boxes,
                groundtruth_object_classes=self.groundtruth_object_classes)
        else:
            first_stage_output = first_stage.predict(
                images=self.images,
                feature_maps=self.rpn_feature_maps)
            sampler_output = sampler.predict(
                images=self.images,
                detection_object_boxes=
                first_stage_output["detection_object_boxes"],
                detection_object_scores=
                first_stage_output["detection_object_scores"],
                num_object_detections=
                first_stage_output["num_object_detections"],
                groundtruth_object_boxes=self.groundtruth_object_boxes,
                groundtruth_object_classes=self.groundtruth_object_classes)
            first_stage_output.update(sampler_output)
        roi_features = roi_plugin.predict(
            feature_maps=self.rpn_feature_maps,
            detection_object_boxes=first_stage_output["detection_object_boxes"])
        second_stage_feature_maps = second_stage_feature_extractor(
            roi_features["feature_maps"])
        second_stage_output = second_stage.predict(
            images=self.images, feature_maps=second_stage_feature_maps,
            proposal_boxes=first_stage_output["detection_object_boxes"],
            num_proposals=first_stage_output["num_object_detections"],
        )

        all_outputs = [first_stage_output, roi_features,
                       second_stage_feature_maps, second_stage_output]

        feed_dict = {
            self.images: self.images_np,
            self.rpn_feature_maps: self.rpn_feature_maps_np,
            self.groundtruth_object_boxes: self.groundtruth_object_boxes_np,
            self.groundtruth_object_classes: self.groundtruth_object_classes_np,
        }

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            all_outputs_eval = sess.run(all_outputs, feed_dict=feed_dict)

        self.assertTupleEqual(
            (self.batch_size, max_num_proposals_mode, 4),
            all_outputs_eval[0]["detection_object_boxes"].shape)
        self.assertTupleEqual(
            (self.batch_size, max_num_proposals_mode),
            all_outputs_eval[0]["detection_object_scores"].shape)
        self.assertTupleEqual(
            (self.batch_size * max_num_proposals_mode, 6, 6, 16),
            all_outputs_eval[1]["feature_maps"].shape)
        self.assertTupleEqual(
            (self.batch_size, max_num_proposals_mode, 4),
            all_outputs_eval[-1]["detection_object_boxes"].shape)
        self.assertTupleEqual(
            (self.batch_size, max_num_proposals_mode),
            all_outputs_eval[-1]["detection_object_classes"].shape)
        self.assertTupleEqual(
            (self.batch_size, max_num_proposals_mode),
            all_outputs_eval[-1]["detection_object_scores"].shape)
