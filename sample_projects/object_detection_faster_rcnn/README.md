Faster RCNN training on coco dataset
====================================

- [Data preparation](#fasterrcnn-coco-data-preparation)
- [Model description](#fasterrcnn-coco-model-description)
- [Training](#fasterrcnn-coco-training)
- [Inference](#fasterrcnn-coco-inference)
- [KPI evaluation](#fasterrcnn-coco-kpi-evaluation)

[coco_extraction_readme]: ../data_extraction/coco/README.md

## Data preparation <a name="fasterrcnn-coco-data-preparation"></a>

This project assumes that you have extracted the data to tfrecords format using
[coco extractor][coco_extraction_readme]. But you can apply it on your data in
tfrecords format with following fields:
- `images_PNG`: images with PNG encoding
- `object_boxes`: normalized object boxes with shape [None, 4] and tf.float32
dtype
- `object_classes`: 1-based object classes
    
The data must be inside of 
`ncgenes7/sample_projects/data_extraction/coco/train/extracted` for train
subset and `ncgenes7/sample_projects/data_extraction/coco/eval/extracted`
for eval subset and it assumes that tfrecord files have GZIP compression.

## Model description <a name="fasterrcnn-coco-model-description"></a>

Model includes following nucleotides:

  * [datasets](training/configs/datasets.json):
    - read data from tfrecords using `ImageDataReaderTfRecords` and
    `ObjectDetectionReaderTfRecords`
    - apply augmentation like random contrast and brightness change,
    random horizontal flip, random crop and random cutout during training
  * [trainer](training/configs/trainer.json) - to specify the
  training parameters like batch size, number of iterations in one epoch,
  optimizer etc.
  * [Plugins](training/configs/plugins):
      * RPN feature extractor (DenseNet121 from keras application pretrained on
      imagenet)
      * positive balanced sampler
      * Faster RCNN first stage box predictor
      * ROI pooling
      * Second stage feature extractor (custom densenet, random initialized)
      * Faster RCNN second stage predictor
  * [Losses](training/configs/losses):
      * first stage loss
      * second stage loss
  * [Postprocessors](training/configs/postprocessors)
      * convert first stage predictions to absolute coordinates
      * format first stage predictions
      * filter second stage predictions by dimension of the bounding box
      * apply NMS on the filtered second stage predictions
      * format second stage predictions and increment the object_classes
      to have 1-based indexing and not 0-based      
  * [Logger](training/configs/callbacks/base_logger.json)
  callback to print status for each iteration
  * [KPI Evaluator](training/configs/callbacks_eval/map_kpieval.json)
  to evaluation mean average precision during evaluation stage on different
  matching thresholds - 0.5 and 0.7 (will track also each class average
  precision)
  * [Summaries](training/configs/summaries): 
  to draw objects from groundtruth, first and second stages, combine them
  together in one image and save it to tensorboard  

You can try following to visualize the model dna:

```bash
nc7-visualize_project_dna . -t train
```

## Training <a name="fasterrcnn-coco-training"></a>

To start the training, use:

```bash
nc7-train .
```

## Inference <a name="fasterrcnn-coco-inference"></a>

Inference will be done on the coco val images, which will be rescaled to be
[480, 640] and final predictions will be stored to json format and also bounding
boxes will be drawn on the images and saved as png
(inside of [inference run folder](inference/last_run/results)).

To start the inference, use:

```bash
nc7-infer . --batch_size {your batch size}
```

Inference configs are stored in [folder](inference/configs)

# KPI evaluation <a name="fasterrcnn-coco-kpi-evaluation"></a>

To calculate KPI of the inferred results use:

```bash
nc7-evaluate_kpi . --batch_size {your batch size}
```

KPI will be saved to [kpi run folder](kpi_evaluation/last_run/results)
