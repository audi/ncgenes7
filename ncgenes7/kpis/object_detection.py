# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
KPI Implementations for object detection
"""
from typing import List
from typing import Optional
from typing import Union

import nucleus7 as nc7
import numpy as np
from object_detection.utils import object_detection_evaluation as od_eval
from object_detection.utils import per_image_evaluation

from ncgenes7.data_fields.object_detection import DetectionDataFields
from ncgenes7.data_fields.object_detection import GroundtruthDataFields
from ncgenes7.utils import io_utils
from ncgenes7.utils import object_detection_utils

# is needed because of compatibility issues with object_detection
od_eval.unicode = lambda x: x


# pylint: disable=too-many-instance-attributes,too-many-arguments
# attributes cannot be combined or extracted further
class DetectionsMatcherKPIPlugin(nc7.kpi.KPIPlugin):
    """
    KPI Plugin to calculate the mathing stats between object detections and
    groundtruth objects.

    Parameters
    ----------
    num_classes
        number of classes
    matching_iou_threshold
        iou threshold to match the ground truth object to the detection;
        all the objects that intersect with more than this threshold, then
        detection is treated as true positive, otherwise it is a false positive
    detections_nms_iou_threshold
        nms threshold to apply on detections before matching; if equal to 1.0,
        nms will not be performed
    detections_nms_max_output_boxes
        number of bounding boxes to select out of provided
    num_classes
        number of classes
    groundtruth_boxes_in_normalized
        flag if the groundtruth boxes are in normalized coordinates
    detection_boxes_in_normalized
        flag if the detection boxes are in normalized coordinates

    Attributes
    ----------
    incoming_keys
        * detection_object_boxes : detected boxes in format
          [ymin, xmin, ymax, xmax] and with shape [num_boxes, 4]
        * detection_object_scores : scores of detections with shape [num_boxes]
        * detection_object_classes : classes for detections starting from 1
          with shape [num_boxes]; all objects with class == 0 will be removed
        * groundtruth_object_boxes : ground truth boxes in format
          [ymin, xmin, ymax, xmax] and with shape [num_boxes, 4]
        * groundtruth_object_classes : ground truth objects classes starting
          from 1; with shape [num_boxes];
          all objects with class == 0 will be removed
    generated_keys
        * matched_scores_per_class : list with each item is a score of detection
          for particular class
        * matched_true_pos_false_pos_per_class : list of arrays where each array
          represents True or False for all detections for particular class,
          where True is for true positive (was matched to GT) and False if for
          false positive (no match in GT)
        * correctly_detected_classes
          array with shape [num_classes] with 1 indicating that at least one
          detection for that class was matched to ground truth
        * number_of_groundtruth_objects_per_class
          array with shape [num_classes] indicating how many ground truth boxes
          with that class were inside of groundtruth
        * number_of_samples_with_groundtruth_per_class
          array with shape [num_classes] with 1 if there is a ground truth
          with that class
    """
    incoming_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_classes,
        DetectionDataFields.detection_object_scores,
        GroundtruthDataFields.groundtruth_object_boxes,
        GroundtruthDataFields.groundtruth_object_classes,
    ]
    generated_keys = [
        "matched_scores_per_class",
        "matched_true_pos_false_pos_per_class",
        "correctly_detected_classes",
        "number_of_groundtruth_objects_per_class",
        "number_of_samples_with_groundtruth_per_class",
    ]

    def __init__(self, num_classes: int,
                 matching_iou_threshold: float = 0.5,
                 detections_nms_iou_threshold: float = 1.0,
                 detections_nms_max_output_boxes: int = 10000,
                 image_size=None,
                 groundtruth_boxes_in_normalized: bool = True,
                 detection_boxes_in_normalized: bool = True,
                 **kpi_plugin_kwargs):
        super().__init__(**kpi_plugin_kwargs)
        self._per_image_evaluation = per_image_evaluation.PerImageEvaluation(
            num_groundtruth_classes=num_classes,
            matching_iou_threshold=matching_iou_threshold,
            nms_iou_threshold=detections_nms_iou_threshold,
            nms_max_output_boxes=detections_nms_max_output_boxes,
        )
        self.num_classes = num_classes
        self.matching_iou_threshold = matching_iou_threshold
        self.detections_nms_iou_threshold = detections_nms_iou_threshold
        self.detections_nms_max_output_boxes = detections_nms_max_output_boxes
        self.image_size = image_size
        self.groundtruth_boxes_in_normalized = groundtruth_boxes_in_normalized
        self.detection_boxes_in_normalized = detection_boxes_in_normalized

    def process(self, *,
                detection_object_boxes,
                detection_object_classes,
                detection_object_scores,
                groundtruth_object_boxes,
                groundtruth_object_classes) -> dict:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        (detection_object_classes, detection_object_boxes,
         detection_object_scores) = _select_valid_objects_by_class_id(
             detection_object_classes, detection_object_boxes,
             detection_object_scores)
        (groundtruth_object_classes, groundtruth_object_boxes
         ) = _select_valid_objects_by_class_id(
             groundtruth_object_classes, groundtruth_object_boxes)

        number_of_boxes = groundtruth_object_boxes.shape[0]
        groundtruth_is_difficult_list = np.zeros(number_of_boxes, dtype=bool)
        groundtruth_is_group_of_list = np.zeros(number_of_boxes, dtype=bool)

        (detection_object_boxes, groundtruth_object_boxes
         ) = self._maybe_normalize_boxes(detection_object_boxes,
                                         groundtruth_object_boxes)

        # since classes must be 1-based
        detection_object_classes = detection_object_classes - 1
        groundtruth_object_classes = groundtruth_object_classes - 1

        scores, tp_fp_labels, is_class_correctly_detected = (
            self._per_image_evaluation.compute_object_detection_metrics(
                detected_boxes=detection_object_boxes,
                detected_scores=detection_object_scores,
                detected_class_labels=detection_object_classes,
                groundtruth_boxes=groundtruth_object_boxes,
                groundtruth_class_labels=groundtruth_object_classes,
                groundtruth_is_difficult_list=groundtruth_is_difficult_list,
                groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            )
        )
        number_of_groundtruth_objects = np.array(
            [np.sum(groundtruth_object_classes == class_id)
             for class_id in range(self.num_classes)], dtype=float)

        number_of_groundtruth_samples = np.zeros(self.num_classes, dtype=int)
        if groundtruth_object_classes.shape[0] > 0:
            number_of_groundtruth_samples[groundtruth_object_classes] = 1

        return {"matched_scores_per_class": tuple(scores),
                "matched_true_pos_false_pos_per_class": tuple(tp_fp_labels),
                "correctly_detected_classes": is_class_correctly_detected,
                "number_of_groundtruth_objects_per_class":
                    number_of_groundtruth_objects,
                "number_of_samples_with_groundtruth_per_class":
                    number_of_groundtruth_samples}

    def _maybe_normalize_boxes(self, detection_object_boxes,
                               groundtruth_object_boxes):
        if (self.groundtruth_boxes_in_normalized
                == self.detection_boxes_in_normalized):
            return detection_object_boxes, groundtruth_object_boxes

        if not self.groundtruth_boxes_in_normalized:
            groundtruth_object_boxes = (
                object_detection_utils.normalize_bbox_np(
                    groundtruth_object_boxes, self.image_size))
        if not self.detection_boxes_in_normalized:
            detection_object_boxes = (
                object_detection_utils.normalize_bbox_np(
                    detection_object_boxes, self.image_size))
        return detection_object_boxes, groundtruth_object_boxes


# pylint: enable=too-many-instance-attributes,too-many-arguments


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class ObjectDetectionKPIAccumulator(nc7.kpi.KPIAccumulator):
    """
    Accumulator to calculate main object detection metrics like:
        * mean average precision (default)
        * corloc (optional)
        * precision and recall (optional)

    Parameters
    ----------
    num_classes
        number of classes
    class_names_to_labels_mapping
        file name or mapping itself; mapping should be in format
        {"class name": {"class_id": 1}, "other class_name": {"class_id": 2}},
        where class_id is an unique integer; if multiple class names have same
        class id, then the last name inside of json file will be used as class
        name
    evaluate_corlocs
        if corlocs also class wise should be calculated
    evaluate_precision_recall
        if precision and recall also class wise should be calculated
    use_weighted_mean_ap
        if the mean average precision should be calculated directly for all
        classes and not as average of classwise precisions
    fscore_beta
        the beta of fscore (e.g. 1 if you want to use f1score) to calculate
        best confidence thresholds for each class or list of scores

    Attributes
    ----------
    incoming_keys
        * matched_scores_per_class : list with each item is a score of detection
          for particular class
        * matched_true_pos_false_pos_per_class : list of arrays where each array
          represents True or False for all detections for particular class,
          where True is for true positive (was matched to GT) and False if for
          false positive (no match in GT)
        * correctly_detected_classes
          array with shape [num_classes] with 1 indicating that at least one
          detection for that class was matched to ground truth
        * number_of_groundtruth_objects_per_class
          array with shape [num_classes] indicating how many ground truth boxes
          with that class were inside of groundtruth
        * number_of_samples_with_groundtruth_per_class
          array with shape [num_classes] with 1 if there is a ground truth
          with that class
    generated_keys
        * mAP : mean average precision over all classes
        * AP-classwise-* : average precision for all classes
        * meanCorLoc : mean corloc if evaluate_corlocs == True
        * CorLoc-classwise-* : corloc for all classes if
          evaluate_corlocs == True
        * Precision-classwise-* : precision for all classes if
          evaluate_precision_recall == True
        * Recall-classwise-* : precision for all classes if
          evaluate_precision_recall == True
    """
    incoming_keys = [
        "matched_scores_per_class",
        "matched_true_pos_false_pos_per_class",
        "correctly_detected_classes",
        "number_of_groundtruth_objects_per_class",
        "number_of_samples_with_groundtruth_per_class",
    ]
    generated_keys = [
        "mAP",
        "_meanCorLoc",
    ]
    dynamic_generated_keys = True

    def __init__(
            self,
            class_names_to_labels_mapping: Optional[Union[str, dict]] = None,
            num_classes: Optional[int] = None,
            evaluate_corlocs=False,
            evaluate_precision_recall=False,
            use_weighted_mean_ap=False,
            fscore_beta: int = 1,
            **kpi_accumulator_kwargs):
        super().__init__(**kpi_accumulator_kwargs)
        assert class_names_to_labels_mapping is not None or num_classes, (
            "Either num_classes or class_names_to_labels_mapping "
            "should be provided!!!")
        self.num_classes = num_classes
        self.class_names_to_labels_mapping = class_names_to_labels_mapping
        self.use_weighted_mean_ap = use_weighted_mean_ap
        self.evaluate_corlocs = evaluate_corlocs
        self.evaluate_precision_recall = evaluate_precision_recall
        if not isinstance(fscore_beta, (list, tuple)):
            fscore_beta = [fscore_beta]
        self.fscore_beta = fscore_beta
        self._class_labels_to_names_mapping = None  # type: dict
        self._od_evaluation = None  # type: od_eval.ObjectDetectionEvaluation

    def build(self):
        super().build()
        self._build_class_labels_to_names_mapping()
        self._build_od_evaluation()
        return self

    def process(self, *,
                matched_scores_per_class: List[np.ndarray],
                matched_true_pos_false_pos_per_class: List[np.ndarray],
                correctly_detected_classes: np.ndarray,
                number_of_groundtruth_objects_per_class: np.ndarray,
                number_of_samples_with_groundtruth_per_class: np.ndarray
                ) -> dict:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        self._add_data_to_od_evaluation(
            matched_scores_per_class, matched_true_pos_false_pos_per_class,
            correctly_detected_classes,
            number_of_groundtruth_objects_per_class,
            number_of_samples_with_groundtruth_per_class)
        (per_class_ap, mean_ap, per_class_precision, per_class_recall,
         per_class_corloc, mean_corloc) = self._od_evaluation.evaluate()
        kpi = self._format_od_metric_to_kpi(
            per_class_ap, mean_ap, per_class_precision, per_class_recall,
            per_class_corloc, mean_corloc)
        best_confidence_thresholds = self._get_best_confidence_scores()
        kpi.update(best_confidence_thresholds)
        return kpi

    def clear_state(self):
        super().clear_state()
        self._od_evaluation.clear_detections()
        self._od_evaluation.num_gt_imgs_per_class.fill(0)
        self._od_evaluation.num_gt_instances_per_class.fill(0)

    def _build_od_evaluation(self):
        # per image evaluation is not done inside of accumulator
        def _per_image_eval_class(**kwargs):  # pylint: disable=unused-argument
            return None

        self._od_evaluation = od_eval.ObjectDetectionEvaluation(
            num_groundtruth_classes=self.num_classes,
            label_id_offset=1,
            use_weighted_mean_ap=self.use_weighted_mean_ap,
            per_image_eval_class=_per_image_eval_class
        )

    def _build_class_labels_to_names_mapping(self):
        (self.num_classes, self.class_names_to_labels_mapping,
         self._class_labels_to_names_mapping
         ) = io_utils.build_class_labels_to_names_mapping(
             self.num_classes, self.class_names_to_labels_mapping,
             class_id_offset=1)

    def _add_data_to_od_evaluation(
            self, matched_scores_per_class,
            matched_true_pos_false_pos_per_class,
            correctly_detected_classes,
            number_of_groundtruth_objects_per_class,
            number_of_samples_with_groundtruth_per_class):
        for sample_scores, sample_tp_fp in zip(
                matched_scores_per_class, matched_true_pos_false_pos_per_class):
            for i in range(self.num_classes):
                if sample_scores[i].shape[0] > 0:
                    self._od_evaluation.scores_per_class[i].append(
                        sample_scores[i])
                    self._od_evaluation.tp_fp_labels_per_class[i].append(
                        sample_tp_fp[i])
        self._od_evaluation.num_images_correctly_detected_per_class = (
            np.sum(correctly_detected_classes, axis=0))

        self._od_evaluation.num_gt_imgs_per_class = np.sum(
            number_of_samples_with_groundtruth_per_class, axis=0)
        self._od_evaluation.num_gt_instances_per_class = np.sum(
            number_of_groundtruth_objects_per_class, axis=0)

    def _format_od_metric_to_kpi(self, per_class_ap, mean_ap,
                                 per_class_precision, per_class_recall,
                                 per_class_corloc, mean_corloc) -> dict:
        def _get_metric_name_for_class(metric_name, class_id):
            return "-".join([metric_name, "classwise",
                             self._class_labels_to_names_mapping[class_id]])

        kpi = {"mAP": mean_ap}
        kpi.update({
            _get_metric_name_for_class("AP", i + 1): class_metric
            for i, class_metric in enumerate(per_class_ap)
        })
        if self.evaluate_corlocs:
            kpi["meanCorLoc"] = mean_corloc
            kpi.update({
                _get_metric_name_for_class("CorLoc", i + 1): class_metric
                for i, class_metric in enumerate(per_class_corloc)
            })
        if self.evaluate_precision_recall:
            kpi.update({
                _get_metric_name_for_class("Precision", i + 1): class_metric
                for i, class_metric in enumerate(per_class_precision)
            })
            kpi.update({
                _get_metric_name_for_class("Recall", i + 1): class_metric
                for i, class_metric in enumerate(per_class_recall)
            })
        return kpi

    def _get_best_confidence_scores(self) -> dict:
        result = {}
        for each_f_beta in self.fscore_beta:
            result.update(_get_best_confidence_thresholds_for_fscore(
                self._od_evaluation, self.num_classes,
                self._class_labels_to_names_mapping, each_f_beta))
        return result


# pylint: enable=too-many-instance-attributes

def _select_valid_objects_by_class_id(
        object_classes, *other_object_properties, class_offset=1):
    valid_mask = np.greater_equal(object_classes, class_offset)
    valid_object_classes = object_classes[valid_mask]
    valid_other_object_properties = [
        each_item[valid_mask] for each_item in other_object_properties]
    return (valid_object_classes, *valid_other_object_properties)


def _get_best_confidence_thresholds_for_fscore(
        od_evaluation: od_eval.ObjectDetectionEvaluation,
        num_classes: int, class_labels_to_names_mapping: dict,
        f_beta: int = 1):
    # pylint: disable=too-many-locals
    scores = od_evaluation.scores_per_class
    precisions = od_evaluation.precisions_per_class
    recalls = od_evaluation.recalls_per_class
    best_thresholds_with_scores = {}
    for each_class_ind in range(num_classes):
        class_name = class_labels_to_names_mapping[each_class_ind + 1]
        scores_cl = scores[each_class_ind]
        if not scores_cl:
            continue
        scores_cl = np.concatenate(scores_cl)
        try:
            precisions_cl = precisions[each_class_ind]
            recalls_cl = recalls[each_class_ind]
        except IndexError:
            precisions_cl = np.ones([1])
            recalls_cl = np.zeros([1])
            scores_cl = scores_cl[:1]
        score_threshold, fscore = _get_best_confidence_for_fscore_single_class(
            precisions_cl, recalls_cl, scores_cl, f_beta)

        threshold_name = (
            'Best_confidence_for_f{}-score/class_id-{}-name-{}'.format(
                f_beta, each_class_ind + 1, class_name))
        score_name = (
            'Best_f{}-score/class_id-{}-name-{}'.format(
                f_beta, each_class_ind + 1, class_name))

        best_thresholds_with_scores[threshold_name] = score_threshold
        best_thresholds_with_scores[score_name] = fscore
    return best_thresholds_with_scores


def _get_best_confidence_for_fscore_single_class(precisions, recalls, scores,
                                                 f_beta=1):
    scores_sorted = np.sort(scores)[::-1]
    fscore = ((1 + f_beta ** 2) * (precisions * recalls)
              / ((f_beta ** 2) * precisions + recalls))
    fscore = np.where(np.isnan(fscore),
                      0,
                      fscore)
    fscore_best_index = np.argmax(fscore)
    best_score = scores_sorted[fscore_best_index]
    return best_score, fscore.max()
