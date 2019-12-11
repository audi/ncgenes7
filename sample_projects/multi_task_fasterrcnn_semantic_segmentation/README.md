Multi task Faster RCNN and semantic segmentation training on coco dataset
=========================================================================

- [Project description](#multi-coco-project-description)
- [Data preparation](#multi-coco-data-preparation)
- [Model description](#multi-coco-model-description)
- [Training](#multi-coco-training)
- [Inference](#multi-coco-inference)
- [KPI evaluation](#multi-coco-kpi-evaluation)

[coco_extraction_readme]: ../data_extraction/coco/README.md

## Project description <a name="multi-coco-project-description"></a>

In this project, the goal is to simultaneously train one neural network to
perform 2 different tasks - semantic segmentation and object detection using
Faster RCNN architecture. This is achieved by sharing the region proposal
network between 2 tasks, where it is used as an encoder for semantic
segmentation.

Here the same dataset (COCO) is used, where for each input image both labels
types exist. But to make the sample project more general, we use 2 different
datasets independently. Then the sample error is calculated only on the head
which has labels for this sample and so the gradients. Other loss is masked
out. The datasets during training phase are  sampled independently with
different probabilities and then combined to batch.

## Data preparation <a name="multi-coco-data-preparation"></a>

This project assumes that you have extracted the data to tfrecords format using
[coco extractor][coco_extraction_readme]. But you can apply it on your data in
tfrecords format with following fields:
- object detection tfrecords:
    - `images_PNG`: images with PNG encoding
    - `object_boxes`: normalized object boxes with shape [None, 4] and
    tf.float32 dtype
    - `object_classes`: 1-based object classes
- semantic segmentation tfrecords:
    - `images_PNG`: images with PNG encoding
    - `segmentation_classes_PNG`: images with segmentation classes with 1 channel
    and PNG encoding

The data must be inside of 
`ncgenes7/sample_projects/data_extraction/coco/train/extracted` for train
subset and `ncgenes7/sample_projects/data_extraction/coco/eval/extracted`
for eval subset and it assumes that tfrecord files have GZIP compression.

The only thing you need to modify is the mapping of class names to class ids
and only if you want to do so.

## Model description <a name="multi-coco-model-description"></a>

Model includes following nucleotides:

  * [datasets](training/configs/datasets.json):
    - read data from tfrecords for training and evaluation and resize images for
    both object detection
    (uses `ImageDataReaderTfRecords` and `ObjectDetectionReaderTfRecords`)
    and segmentation
    (uses `ImageDataReaderTfRecords` and `SemanticSegmentationReaderTfRecords`)
    subsets and sample with [0.7, 0.3] probabilities out of them.
    - apply augmentation like random contrast and brightness change,
    random horizontal flip, random crop and random cutout during training
  * [trainer](training/configs/trainer.json) - to specify the
  training parameters like batch size, number of iterations in one epoch,
  optimizer etc.
  * [Plugins](training/configs/plugins):
      * RPN feature extractor (densenet)
      * Faster RCNN first stage box predictor
      * ROI pooling
      * Second stage feature extractor (densenet)
      * Faster RCNN second stage predictor
      * semantic segmentation decoder with skip connections starting from
      rpn and predicting the logits
  * [Losses](training/configs/losses):
      * first stage loss
      * second stage loss
      * semantic segmentation softmax cross entropy loss
  * [Postprocessors](training/configs/postprocessors)
      * convert first stage predictions to absolute coordinates
      * format first stage predictions
      * filter second stage predictions by dimension of the bounding box
      * apply NMS on the filtered second stage predictions
      * format second stage predictions and increment the object_classes
      to have 1-based indexing and not 0-based
      * argmax on semantic segmentation logits to get the class
      * identity just to rename the argmax to prediction_segmentation_classes       
  * [Logger](training/configs/callbacks/base_logger.json)
  callback to print status for each iteration
  * [KPI Evaluator](training/configs/callbacks_eval/map_kpieval.json)
  to evaluation mean average precision during evaluation stage 
  * [Metrics](training/configs/metrics)
  to monitor the confusion matrix and IoU class wise and mean over training time
  during evaluation stage
  * [Summaries](training/configs/summaries): 
    - to draw objects from groundtruth, first and second stages, combine them
    together in one image and save it to tensorboard
    - to draw predicted segmentation classes together with groundtruth for further
    use on tensorboard

You can try following to visualize the model dna:

```bash
nc7-visualize_project_dna . -t train
```

## Training <a name="multi-coco-training"></a>

To start the training, use:

```bash
nc7-train .
```

## Inference <a name="multi-coco-inference"></a>

Inference will be done on the coco val images, which will be rescaled to be
[480, 640] and final predictions will be stored to json format and also bounding
boxes will be drawn on the images and saved as png
(inside of [inference run folder](inference/last_run/results)).

To start the inference, use:

```bash
nc7-infer . --batch_size {your batch size}
```

Inference configs are stored in [folder](inference/configs)

# KPI evaluation <a name="multi-coco-kpi-evaluation"></a>

To calculate KPI of the inferred results use:

```bash
nc7-evaluate_kpi . --batch_size {your batch size}
```

KPI will be saved to [kpi run folder](kpi_evaluation/last_run/results)
