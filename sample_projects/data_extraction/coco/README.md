Prepare coco dataset for training
=================================

- [Download dataset](#coco-download-dataset)
- [Project description](#coco-project-description)
- [Generate tfrecords](#coco-generate-tfrecords)

[download_link]: http://cocodataset.org/#download

Coco dataset has multiple types of data. In this tutorial we will generate
tfrecords files with semantic segmentation, bounding boxes and keypoints
labels for training and validation sets.

## Download dataset <a name="coco-download-dataset"></a>

Refer to [coco page][download_link] for information how to download it and
extract archives. You will need following data:

* 2017 Train images
* 2017 Val images
* 2017 Train/Val annotations
* 2017 Panoptic Train/Val annotations

In later we assume that you extracted all archives to `raw_data` folder.

## Project description <a name="coco-project-description"></a>

We will save following data to tfrecord files:

- Images encoded as png inside of key "images_PNG" using `ImageDataReader`
following `ImageEncoder` processor
- Object data using `CocoObjectsReader`:
    - bounding boxes under "object_boxes" key
    - class id under "object_classes" key
    - instance id under "object_instance_ids" key
- Semantic segmentation out of panoptic annotations as images with png encoding
under "segmentation_classes" key using `CocoSemanticSegmentationReader`
following `ImageEncoder` processor.
- Persons keypoints data using `CocoPersonKeypointsReader`:
    - keypoints are saved under "object_keypoints" key
    - keypoints visibilities are saved under "object_keypoints_visibilities" key
    - bounding boxes under "object_boxes_from_keypoints" key
    - class id under "object_classes_from_keypoints" key
    - instance id under "object_instance_ids_from_keypoints" key

## Generate tfrecords <a name="coco-generate-tfrecords"></a>

To generate tfrecords files, we will use the `data_extraction` task for nucleus7
project and create 2 sets - train and eval.
Configs for it are inside of `data_extraction/configs` and the run
specific configs are inside of `data_extraction/train/configs` and
`data_extraction/eval/configs`.

All tfrecords files will be compressed using GZIP compression and for training
set, there will be up to 50 samples per file and for evaluation - up to 1000
samples per file.

Now we can generate the tfrecord files for training / validation using following
scripts:

```bash
# run from parent coco (this) folder, e.g. data_extraction

# 1. For training subset

nc7-extract_data coco --run_name train 

# 2. Extract evaluation subset

nc7-extract_data coco --run_name eval 
```

Now we have generated tfrecord files with semantic segmentation and 2D bounding
boxes and corresponding images for coco dataset inside of
`data_extraction/train/extracted/` and `data_extraction/eval/extracted/`
folders respectively.
