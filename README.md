## How to run fruit project

* To install, See [INSTALL.md](INSTALL.md).
Note: You might need to run `python -u setup.py build develop --prefix=~/.local` instead of `python setup.py build develop` if you are not using a sudo account.

* Download pre-trained weights
[Faster R-CNN](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl), rename it as faster_rcnn_R_50_FPN_3x.pkl

[Mask R-CNN](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl), rename it as mask_rcnn_R_50_FPN_3x.pkl

* Train the model
Faster R-CNN `python -u tools/train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --model-weight faster_rcnn_R_50_FPN_3x.pkl --train-annotations /absolute-path/datasets/annotations/annotations_train.json --train-images /absolute-path/datasets/train_images_coco`

Faster R-CNN with ROI-Pooling `python -u tools/train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_roi_pooling.yaml --model-weight faster_rcnn_R_50_FPN_3x.pkl --train-annotations /absolute-path/datasets/annotations/annotations_train.json --train-images /absolute-path/datasets/train_images_coco`

Mask R-CNN `python -u tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --model-weight mask_rcnn_R_50_FPN_3x.pkl --train-annotations /absolute-path/datasets/annotations/annotations_train.json --train-images /absolute-path/datasets/train_images_coco`

* Test the model
Faster R-CNN `python -u tools/predict.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --test-annotations /absolute-path/datasets/annotations/annotations_test.json --test-images /absolute-path/datasets/test_images_coco`

Faster R-CNN with ROI-Pooling `python -u tools/predict.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_roi_pooling.yaml --test-annotations /absolute-path/datasets/annotations/annotations_test.json --test-images /absolute-path/datasets/test_images_coco`

Mask R-CNN `python -u tools/predict.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --test-annotations /absolute-path/datasets/annotations/annotations_test.json --test-images /absolute-path/datasets/test_images_coco`

Note: Please use absolute path for the paths of annotation files and image directories. The dataloader is called from the built installation and changing sys path in the code raises path errors from other functions.

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
