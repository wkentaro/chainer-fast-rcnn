#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'fast-rcnn/caffe-fast-rcnn/build/install/python')
import caffe
import chainer.serializers as S
from vgg16 import VGG16
import cPickle as pickle

param_dir = 'fast-rcnn/data/fast_rcnn_models'
param_fn = '%s/vgg16_fast_rcnn_iter_40000.caffemodel' % param_dir
model_dir = 'fast-rcnn/models/VGG16'
model_fn = '%s/test.prototxt' % model_dir

model = VGG16()
net = caffe.Net(model_fn, param_fn, caffe.TEST)
for name, param in net.params.iteritems():
    layer = getattr(model, name)

    print name, param[0].data.shape, param[1].data.shape,
    print layer.W.data.shape, layer.b.data.shape

    layer.W.data = param[0].data
    layer.b.data = param[1].data

S.save_hdf5('models/vgg16.chainermodel', model)
