#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'models')

import argparse

import caffe
import chainer.serializers as S
import cPickle as pickle

from caffenet import CaffeNet
from vgg16 import VGG16
from vgg_cnn_m_1024 import VGG_CNN_M_1024


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vgg_cnn_m_1024')
args = parser.parse_args()

model_name = args.model
if model_name == 'caffenet':
    model_name_capital = 'CaffeNet'
else:
    model_name_capital = model_name.upper()

param_dir = 'fast-rcnn/data/fast_rcnn_models'
param_fn = '%s/%s_fast_rcnn_iter_40000.caffemodel' % (param_dir, model_name)
model_dir = 'fast-rcnn/models/%s' % model_name_capital
model_fn = '%s/test.prototxt' % model_dir

if model_name == 'caffenet':
    model = CaffeNet()
elif model_name == 'vgg16':
    model = VGG16()
elif model_name == 'vgg_cnn_m_1024':
    model = VGG_CNN_M_1024()
else:
    raise ValueError('Unsupported model name: %s' % model_name)

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
