# Copyright 2017 The TensorFlow Authors.  All rights reserved.
#
# Modifications are commented in the code
# original code is from
# https://github.com/tensorflow/models/blob/master/research/object_detection/
#   meta_architectures/faster_rcnn_meta_arch.py
# Format was modified to have indent of 4 spaces
#
#                              Apache License
#                        Version 2.0, January 2004
#                     http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# 1. Definitions.
#
#   "License" shall mean the terms and conditions for use, reproduction,
#   and distribution as defined by Sections 1 through 9 of this document.
#
#   "Licensor" shall mean the copyright owner or entity authorized by
#   the copyright owner that is granting the License.
#
#   "Legal Entity" shall mean the union of the acting entity and all
#   other entities that control, are controlled by, or are under common
#   control with that entity. For the purposes of this definition,
#   "control" means (i) the power, direct or indirect, to cause the
#   direction or management of such entity, whether by contract or
#   otherwise, or (ii) ownership of fifty percent (50%) or more of the
#   outstanding shares, or (iii) beneficial ownership of such entity.
#
#   "You" (or "Your") shall mean an individual or Legal Entity
#   exercising permissions granted by this License.
#
#   "Source" form shall mean the preferred form for making modifications,
#   including but not limited to software source code, documentation
#   source, and configuration files.
#
#   "Object" form shall mean any form resulting from mechanical
#   transformation or translation of a Source form, including but
#   not limited to compiled object code, generated documentation,
#   and conversions to other media types.
#
#   "Work" shall mean the work of authorship, whether in Source or
#   Object form, made available under the License, as indicated by a
#   copyright notice that is included in or attached to the work
#   (an example is provided in the Appendix below).
#
#   "Derivative Works" shall mean any work, whether in Source or Object
#   form, that is based on (or derived from) the Work and for which the
#   editorial revisions, annotations, elaborations, or other modifications
#   represent, as a whole, an original work of authorship. For the purposes
#   of this License, Derivative Works shall not include works that remain
#   separable from, or merely link (or bind by name) to the interfaces of,
#   the Work and Derivative Works thereof.
#
#   "Contribution" shall mean any work of authorship, including
#   the original version of the Work and any modifications or additions
#   to that Work or Derivative Works thereof, that is intentionally
#   submitted to Licensor for inclusion in the Work by the copyright owner
#   or by an individual or Legal Entity authorized to submit on behalf of
#   the copyright owner. For the purposes of this definition, "submitted"
#   means any form of electronic, verbal, or written communication sent
#   to the Licensor or its representatives, including but not limited to
#   communication on electronic mailing lists, source code control systems,
#   and issue tracking systems that are managed by, or on behalf of, the
#   Licensor for the purpose of discussing and improving the Work, but
#   excluding communication that is conspicuously marked or otherwise
#   designated in writing by the copyright owner as "Not a Contribution."
#
#   "Contributor" shall mean Licensor and any individual or Legal Entity
#   on behalf of whom a Contribution has been received by Licensor and
#   subsequently incorporated within the Work.
#
# 2. Grant of Copyright License. Subject to the terms and conditions of
#   this License, each Contributor hereby grants to You a perpetual,
#   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#   copyright license to reproduce, prepare Derivative Works of,
#   publicly display, publicly perform, sublicense, and distribute the
#   Work and such Derivative Works in Source or Object form.
#
# 3. Grant of Patent License. Subject to the terms and conditions of
#   this License, each Contributor hereby grants to You a perpetual,
#   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#   (except as stated in this section) patent license to make, have made,
#   use, offer to sell, sell, import, and otherwise transfer the Work,
#   where such license applies only to those patent claims licensable
#   by such Contributor that are necessarily infringed by their
#   Contribution(s) alone or by combination of their Contribution(s)
#   with the Work to which such Contribution(s) was submitted. If You
#   institute patent litigation against any entity (including a
#   cross-claim or counterclaim in a lawsuit) alleging that the Work
#   or a Contribution incorporated within the Work constitutes direct
#   or contributory patent infringement, then any patent licenses
#   granted to You under this License for that Work shall terminate
#   as of the date such litigation is filed.
#
# 4. Redistribution. You may reproduce and distribute copies of the
#   Work or Derivative Works thereof in any medium, with or without
#   modifications, and in Source or Object form, provided that You
#   meet the following conditions:
#
#   (a) You must give any other recipients of the Work or
#       Derivative Works a copy of this License; and
#
#   (b) You must cause any modified files to carry prominent notices
#       stating that You changed the files; and
#
#   (c) You must retain, in the Source form of any Derivative Works
#       that You distribute, all copyright, patent, trademark, and
#       attribution notices from the Source form of the Work,
#       excluding those notices that do not pertain to any part of
#       the Derivative Works; and
#
#   (d) If the Work includes a "NOTICE" text file as part of its
#       distribution, then any Derivative Works that You distribute must
#       include a readable copy of the attribution notices contained
#       within such NOTICE file, excluding those notices that do not
#       pertain to any part of the Derivative Works, in at least one
#       of the following places: within a NOTICE text file distributed
#       as part of the Derivative Works; within the Source form or
#       documentation, if provided along with the Derivative Works; or,
#       within a display generated by the Derivative Works, if and
#       wherever such third-party notices normally appear. The contents
#       of the NOTICE file are for informational purposes only and
#       do not modify the License. You may add Your own attribution
#       notices within Derivative Works that You distribute, alongside
#       or as an addendum to the NOTICE text from the Work, provided
#       that such additional attribution notices cannot be construed
#       as modifying the License.
#
#   You may add Your own copyright statement to Your modifications and
#   may provide additional or different license terms and conditions
#   for use, reproduction, or distribution of Your modifications, or
#   for any such Derivative Works as a whole, provided Your use,
#   reproduction, and distribution of the Work otherwise complies with
#   the conditions stated in this License.
#
# 5. Submission of Contributions. Unless You explicitly state otherwise,
#   any Contribution intentionally submitted for inclusion in the Work
#   by You to the Licensor shall be under the terms and conditions of
#   this License, without any additional terms or conditions.
#   Notwithstanding the above, nothing herein shall supersede or modify
#   the terms of any separate license agreement you may have executed
#   with Licensor regarding such Contributions.
#
# 6. Trademarks. This License does not grant permission to use the trade
#   names, trademarks, service marks, or product names of the Licensor,
#   except as required for reasonable and customary use in describing the
#   origin of the Work and reproducing the content of the NOTICE file.
#
# 7. Disclaimer of Warranty. Unless required by applicable law or
#   agreed to in writing, Licensor provides the Work (and each
#   Contributor provides its Contributions) on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#   implied, including, without limitation, any warranties or conditions
#   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#   PARTICULAR PURPOSE. You are solely responsible for determining the
#   appropriateness of using or redistributing the Work and assume any
#   risks associated with Your exercise of permissions under this License.
#
# 8. Limitation of Liability. In no event and under no legal theory,
#   whether in tort (including negligence), contract, or otherwise,
#   unless required by applicable law (such as deliberate and grossly
#   negligent acts) or agreed to in writing, shall any Contributor be
#   liable to You for damages, including any direct, indirect, special,
#   incidental, or consequential damages of any character arising as a
#   result of this License or out of the use or inability to use the
#   Work (including but not limited to damages for loss of goodwill,
#   work stoppage, computer failure or malfunction, or any and all
#   other commercial damages or losses), even if such Contributor
#   has been advised of the possibility of such damages.
#
# 9. Accepting Warranty or Additional Liability. While redistributing
#   the Work or Derivative Works thereof, You may choose to offer,
#   and charge a fee for, acceptance of support, warranty, indemnity,
#   or other liability obligations and/or rights consistent with this
#   License. However, in accepting such obligations, You may act only
#   on Your own behalf and on Your sole responsibility, not on behalf
#   of any other Contributor, and only if You agree to indemnify,
#   defend, and hold each Contributor harmless for any liability
#   incurred by, or claims asserted against, such Contributor by reason
#   of your accepting any such warranty or additional liability.
#
# END OF TERMS AND CONDITIONS
#
# APPENDIX: How to apply the Apache License to your work.
#
#   To apply the Apache License to your work, attach the following
#   boilerplate notice, with the fields enclosed by brackets "[]"
#   replaced with your own identifying information. (Don't include
#   the brackets!)  The text should be enclosed in the appropriate
#   comment syntax for the file format. We also recommend that a
#   file or class name and description of purpose be included on the
#   same "printed page" as the copyright notice for easier
#   identification within third-party archives.
# ==============================================================================

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import standard_fields as od_standard_fields
from object_detection.meta_architectures.faster_rcnn_meta_arch import \
    FasterRCNNMetaArch as FasterRCNNMetaArchOrig
import tensorflow as tf


class FasterRCNNMetaArch(FasterRCNNMetaArchOrig):

    def _predict_first_stage(self, preprocessed_inputs):
        # needs to be overridden since original one is not handling the clipping
        # of anchors and boxes to window in right way for inference and
        # static_shapes == False and also it leaves the anchors that do not
        # overlap at all with window
        # TODO(oleksandr.vorobiov@audi.de) make PR to tensorflow models
        (rpn_box_predictor_features, rpn_features_to_crop, anchors_boxlist,
         image_shape) = self._extract_rpn_feature_maps(preprocessed_inputs)
        (rpn_box_encodings, rpn_objectness_predictions_with_background
         ) = self._predict_rpn_proposals(rpn_box_predictor_features)

        # The Faster R-CNN paper recommends pruning anchors that venture outside
        # the image window at training time and clipping at inference time.
        clip_window = tf.to_float(
            tf.stack([0, 0, image_shape[1], image_shape[2]]))
        if self._is_training:
            if self.clip_anchors_to_image:
                # Modification: use injected code from
                # _clip_anchors_and_predictions_to_window instead of
                # box_list_ops.clip_to_window
                (anchors_boxlist, rpn_box_encodings,
                 rpn_objectness_predictions_with_background) = (
                    self._clip_anchors_and_predictions_to_window(
                        anchors_boxlist, rpn_box_encodings,
                        rpn_objectness_predictions_with_background,
                        clip_window))
                # End of modification
            else:
                (rpn_box_encodings, rpn_objectness_predictions_with_background,
                 anchors_boxlist
                 ) = self._remove_invalid_anchors_and_predictions(
                    rpn_box_encodings,
                    rpn_objectness_predictions_with_background,
                    anchors_boxlist, clip_window)
        else:
            # Modification: use injected code from
            # _clip_anchors_and_predictions_to_window instead of
            # box_list_ops.clip_to_window
            (anchors_boxlist, rpn_box_encodings,
             rpn_objectness_predictions_with_background) = (
                self._clip_anchors_and_predictions_to_window(
                    anchors_boxlist, rpn_box_encodings,
                    rpn_objectness_predictions_with_background,
                    clip_window))
            # End of modification

        self._anchors = anchors_boxlist
        prediction_dict = {
            'rpn_box_predictor_features':
                rpn_box_predictor_features,
            'rpn_features_to_crop':
                rpn_features_to_crop,
            'image_shape':
                image_shape,
            'rpn_box_encodings':
                rpn_box_encodings,
            'rpn_objectness_predictions_with_background':
                rpn_objectness_predictions_with_background,
            'anchors':
                anchors_boxlist.data['boxes'],
        }
        return prediction_dict

    # Modification: add new method
    def _clip_anchors_and_predictions_to_window(
            self, anchors_boxlist, rpn_box_encodings,
            rpn_objectness_predictions_with_background,
            clip_window
    ):
        anchors_and_predictions_boxlist = box_list.BoxList(
            anchors_boxlist.get())
        anchors_and_predictions_boxlist.add_field(
            "rpn_box_encodings", tf.transpose(rpn_box_encodings, [1, 0, 2]))
        anchors_and_predictions_boxlist.add_field(
            "rpn_objectness_predictions_with_background",
            tf.transpose(rpn_objectness_predictions_with_background, [1, 0, 2]))

        anchors_and_predictions_boxlist_clipped = box_list_ops.clip_to_window(
            anchors_and_predictions_boxlist, clip_window,
            filter_nonoverlapping=not self._use_static_shapes)
        anchors_boxlist_clipped = box_list.BoxList(
            anchors_and_predictions_boxlist_clipped.get())
        rpn_box_encodings_clipped = tf.transpose(
            anchors_and_predictions_boxlist_clipped.get_field(
                "rpn_box_encodings"), [1, 0, 2])
        rpn_objectness_predictions_with_background_clipped = tf.transpose(
            anchors_and_predictions_boxlist_clipped.get_field(
                "rpn_objectness_predictions_with_background"), [1, 0, 2])
        return (anchors_boxlist_clipped, rpn_box_encodings_clipped,
                rpn_objectness_predictions_with_background_clipped)

    def _sample_box_classifier_batch(self, proposal_boxes, proposal_scores,
                                     num_proposals, groundtruth_boxlists,
                                     groundtruth_classes_with_background_list,
                                     groundtruth_weights_list):
        # Modification: use _sample_single_image and tf.map_fn to handle
        # dynamic batch_size
        # TODO(oleksandr.vorobiov@audi.de) make PR to tensorflow models
        def _sample_single_image(inputs):
            (single_image_proposal_boxes, single_image_proposal_scores,
             single_image_num_proposals, single_image_groundtruth_boxes,
             single_image_groundtruth_classes_with_background,
             single_image_groundtruth_weights) = inputs
            single_image_boxlist = box_list.BoxList(single_image_proposal_boxes)
            single_image_boxlist.add_field(
                od_standard_fields.BoxListFields.scores,
                single_image_proposal_scores)
            single_image_groundtruth_boxlist = box_list.BoxList(
                single_image_groundtruth_boxes)
            (sampled_boxlist
             ) = self._sample_box_classifier_minibatch_single_image(
                single_image_boxlist,
                single_image_num_proposals,
                single_image_groundtruth_boxlist,
                single_image_groundtruth_classes_with_background,
                single_image_groundtruth_weights)
            sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
                sampled_boxlist,
                num_boxes=self._second_stage_batch_size)

            single_num_proposals_sampled = tf.minimum(
                sampled_boxlist.num_boxes(),
                self._second_stage_batch_size)
            single_boxes_sampled = sampled_padded_boxlist.get()
            single_scores_sampled = sampled_padded_boxlist.get_field(
                od_standard_fields.BoxListFields.scores)
            return (single_boxes_sampled, single_scores_sampled,
                    single_num_proposals_sampled)

        boxes_sampled, scores_sampled, num_proposals_sampled = tf.map_fn(
            _sample_single_image,
            elems=(proposal_boxes, proposal_scores, num_proposals,
                   groundtruth_boxlists,
                   groundtruth_classes_with_background_list,
                   groundtruth_weights_list),
            dtype=(tf.float32, tf.float32, tf.int32),
            parallel_iterations=self._parallel_iterations,
            back_prop=False,
        )
        # End of modification
        return boxes_sampled, scores_sampled, num_proposals_sampled
