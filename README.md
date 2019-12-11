<a>
    <img src="icon.png" alt="ncgenes7 logo" title="ncgenes7" align="right" height="120" />
</a>


ncgenes7
========

Welcome to **ncgenes7** - library with interfaces for different **nucleus7**
Nucleotides.

- [Installation](./INSTALL.md)
- [How to use](#how-to-use)
- [Nucleotide implementations](#nucleotide-implementations)
- Sample projects
    * [Data extraction COCO dataset](./sample_projects/data_extraction/coco/README.md)
    * [Semantic segmentation COCO](./sample_projects/semantic_segmentation_coco/README.md)
    * [Object detection](./sample_projects/object_detection_faster_rcnn/README.md)
    * [Multi tasking](./sample_projects/multi_task_fasterrcnn_semantic_segmentation/README.md)
- [Documentation](https://aev.github.io/ncgenes7/)
- [Contribution](./CONTRIBUTING.md)

## How to use <a name="how-to-use"></a>

Even if it is possible to use the ncgenes7 implementations as standalone
objects, the main goal is to use it inside of **nucleus7** projects.

To add any of the nucleotides to your project, just
[install / activate ncgenes7](./INSTALL.md)
and use `"class_name": "ncgehes7.package_name.module_name.ClassName"`, e.g.
to use DenseNetPlugin, use
`"class_name": "ncgenes7.plugins.cnns.densenet.DensenetPlugin"` inside of
plugin config.

## Nucleotide implementations <a name="nucleotide-implementations"></a>

**ncgenes7** have multiple sub-packages, where name of the package is the
nucleotide type, e.g. `plugins` refers to `ModelPlugin` and `kpis` refers
to kpi related nucleotides like `KPIPlugin` and `KPIAccumulator`. 
