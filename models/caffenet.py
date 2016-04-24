#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'functions')

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from roi_pooling_2d import roi_pooling_2d


class CaffeNet(chainer.Chain):

    def __init__(self):
        super(CaffeNet, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4, pad=5),
            conv2=L.Convolution2D(48, 256, 5, pad=2),
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(192, 384, 3, pad=1),
            conv5=L.Convolution2D(192, 256, 3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            cls_score=L.Linear(4096, 21),
            bbox_pred=L.Linear(4096, 84)
        )
        self.train = False

    def __call__(self, x, rois):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = F.local_response_normalization(h, n=5, alpha=1e-4, beta=.75)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = F.local_response_normalization(h, n=5, alpha=1e-4, beta=.75)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))

        h = roi_pooling_2d(h, rois, 6, 6, 0.0625)
        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)

        cls_score = F.softmax(self.cls_score(h))
        bbox_pred = self.bbox_pred(h)

        return cls_score, bbox_pred
