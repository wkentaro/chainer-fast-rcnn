#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'models')
import caffe
import chainer.serializers as S
from caffenet import CaffeNet
import cPickle as pickle

param_dir = 'fast-rcnn/data/fast_rcnn_models'
param_fn = '%s/caffenet_fast_rcnn_iter_40000.caffemodel' % param_dir
model_dir = 'fast-rcnn/models/CaffeNet'
model_fn = '%s/test.prototxt' % model_dir

model = CaffeNet()
net = caffe.Net(model_fn, param_fn, caffe.TEST)
for name, param in net.params.iteritems():
    layer = getattr(model, name)

    print name, param[0].data.shape, param[1].data.shape,
    print layer.W.data.shape, layer.b.data.shape

    assert layer.W.data.shape == param[0].data.shape
    layer.W.data = param[0].data

    assert layer.b.data.shape == param[1].data.shape
    layer.b.data = param[1].data

S.save_hdf5('models/caffenet.chainermodel', model)
