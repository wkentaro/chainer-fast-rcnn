# chainer-fast-rcnn

It aims at reproducing results of [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) using [Chainer](https://github.com/pfn/chainer). It can be run only with GPU because roi_pooling_2d layer has only GPU implementation.

## Requirements

- [OpenCV 2.4/3.0 with python bindings](http://opencv.org)
- [Chainer 1.8](https://github.com/pfn/chainer)
- [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) (optional)
- [dlib v18.18](https://github.com/davisking/dlib)

## Get Chainer models

You can skip below optional sections with below command:

```bash
bash scripts/download_chainermodels.sh
```

## Setup fast-rcnn (optional)

Create a symlink from the location of original [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) dir to this project's root dir. (The below line assumes a environment variable `$FRCN_ROOT` has a path to the `fast-rcnn` source dir.)

```
$ ln -s $FRCN_ROOT ./
```

Make sure that all steps written in the `Installation (sufficient for the demo)` section of `README.md` in [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) have been performed.

## Convert model (optional)

Convert caffemodel to chainermodel.

```
$ python scripts/load_net.py
```

## Test

First you should prepare a sample image, and then

```
$ python scripts/forward.py --img_fn sample.jpg --out_fn result.jpg
```

### Result

![](data/vgg16_forward_out/5891724192.jpg)

'Overstekend wild' St. Janskerkhof Den Bosch &copy; FaceMePLS (https://www.flickr.com/photos/faceme/5891724192)
