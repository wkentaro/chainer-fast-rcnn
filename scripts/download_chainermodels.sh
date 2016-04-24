#!/bin/bash

_THIS_DIR=$(builtin cd "`dirname "${BASH_SOURCE[0]}"`" > /dev/null && pwd)
cd $_THIS_DIR/../models


url='https://drive.google.com/uc?id=0B9P1L--7Wd2vX015UzB4aC13cVk'
filename='vgg16.chainermodel'
if [ ! -e $filename ]; then
  gdown $url -O $filename
fi


url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZzJuaFRIdDMtLWc'
filename='vgg_cnn_m_1024.chainermodel'
if [ ! -e $filename ]; then
  gdown $url -O $filename
fi


url='https://drive.google.com/uc?id=0B9P1L--7Wd2vWFZtS3Zob2VTYmc'
filename='caffenet.chainermodel'
if [ ! -e $filename ]; then
  gdown $url -O $filename
fi
