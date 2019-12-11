Prepare a2d2 dataset for training
=================================

- [Download dataset](#a2d2-download-dataset)
- [Project description](#a2d2-project-description)
- [Generate tfrecords](#a2d2-generate-tfrecords)

[download_link]: http://www.a2d2.audi

A2D2 has multiple types of data like images from different cameras,
semantic segmentation, 3D bounding boxes, LIDAR points etc.
In this tutorial we will generate
tfrecords files with semantic segmentation for training and validation sets.

## Download dataset <a name="a2d2-download-dataset"></a>

Refer to [a2d2][download_link] for information how to download the dataset. You
will need **Semantic Segmentation part**. Extract the downloaded archive to
`raw_data` folder (to get `raw_data/camera_lidar_semantic/*`).

## Project description <a name="a2d2-project-description"></a>

We will save following data to tfrecord files:

- Images encoded as png inside of key "images_PNG" using `ImageDataReader`
following `ImageEncoder` processor
- Semantic segmentation segmentation images with png encoding
under "segmentation_classes" key using `ImageDataReader`
following `ImageEncoder` processor.

## Generate tfrecords <a name="a2d2-generate-tfrecords"></a>

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
# run from parent a2d2 (this) folder, e.g. data_extraction

# 1. For training subset

nc7-extract_data a2d2 --run_name train 

# 2. Extract evaluation subset

nc7-extract_data a2d2 --run_name eval 
```

Now we have generated tfrecord files with semantic segmentation and
corresponding images for A2D2 dataset inside of
`data_extraction/train/extracted/` and `data_extraction/eval/extracted/`
folders respectively.
