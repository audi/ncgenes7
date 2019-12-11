Semantic segmentation on coco dataset
=====================================

- [Data preparation](#segm-coco-data-preparation)
- [Model description](#segm-coco-model-description)
- [Training](#segm-coco-training)
- [Inference](#segm-coco-inference)
- [KPI evaluation](#segm-coco-kpi-evaluation)

[coco_extraction_readme]: ../data_extraction/coco/README.md

## Data preparation <a name="segm-coco-data-preparation"></a>

This project assumes that you have extracted the data to tfrecords format using
[coco extractor][coco_extraction_readme]. But you can apply it on your data in
tfrecords format with following fields:
- `images_PNG`: images with PNG encoding
- `segmentation_classes_PNG`: images with segmentation classes with 1 channel
and PNG encoding

The data must be inside of 
`ncgenes7/sample_projects/data_extraction/coco/train/extracted` for train
subset and `ncgenes7/sample_projects/data_extraction/coco/eval/extracted`
for eval subset and it assumes that tfrecord files have GZIP compression.

## Model description <a name="segm-coco-model-description"></a>

Model includes following nucleotides:

  * [datasets](training/configs/datasets.json):
    - read data from tfrecords using `ImageDataReaderTfRecords` and
    `SemanticSegmentationReaderTfRecords`
    - apply augmentation like random contrast and brightness change,
    random horizontal flip and random crop during training
  * [trainer](training/configs/trainer.json) - to specify the
  training parameters like batch size, number of iterations in one epoch,
  optimizer etc.
  * [Plugins](training/configs/plugins):
      * encoder - deep CNN model using inception modules used by AEV on NIPS2017
      * decoder with skip connections with same like architecture as encoder
      * DUC module as alternative to conventional deconvolution on logits layer
  * [Losses](training/configs/losses):
      * softmax cross entropy loss
  * [Postprocessors](training/configs/postprocessors)
      * argmax on logits to get the class
      * identity just to rename the argmax to prediction_segmentation_classes
  * [Logger](training/configs/callbacks/base_logger.json)
  callback to print status for each iteration
  * [Metrics](training/configs/metrics)
  to monitor the confusion matrix and IoU class wise and mean over training time
  during evaluation stage 
  * [Summaries](training/configs/summaries): 
  to draw predicted segmentation classes together with groundtruth for further
  use on tensorboard
  * [Early stopping callback](training/configs/callbacks_eval/segmentation_early_stopping.json): 
  will stop the training if the segmentation metric did not improve more than
  0.5 in IOU in last 10 epochs 

You can try following to visualize the model dna:

```bash
nc7-visualize_project_dna . -t train
```

## Training <a name="segm-coco-training"></a>

To start the training, use:

```bash
nc7-train .
```

## Inference <a name="segm-coco-inference"></a>

Inference will be done on the coco val images, which will be rescaled to be
[480, 640] and final predictions will be stored as png images with every RGB
value representing the class id
(inside of [inference run folder](inference/last_run/results)).

To start the inference, use:

```bash
nc7-infer . --batch_size {your batch size}
```

Inference configs are stored in [folder](inference/configs)

# KPI evaluation <a name="segm-coco-kpi-evaluation"></a>

To calculate KPI of the inferred results use:

```bash
nc7-evaluate_kpi . --batch_size {your batch size}
```

KPI will be saved to [kpi run folder](kpi_evaluation/last_run/results)
