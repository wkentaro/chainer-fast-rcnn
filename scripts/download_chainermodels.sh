#!/bin/bash

_THIS_DIR=$(builtin cd "`dirname "${BASH_SOURCE[0]}"`" > /dev/null && pwd)
cd $_THIS_DIR/../data


function check_md5_and_download () {
  local md5=$1
  local filename=$2
  local url=$3
  if [ -e $filename ]; then
    actual_md5=$(md5sum $filename | awk '{print $1}')
    echo "Checking md5: $actual_md5 <-> $md5"
    if [ "${actual_md5}" != "${md5}" ]; then
      gdown $url -O $filename
    fi
  else
    gdown $url -O $filename
  fi
}


url='https://drive.google.com/uc?id=0B9P1L--7Wd2vX015UzB4aC13cVk'
md5='5ae12288962e96124cce212fd3f18cad'
filename='vgg16.chainermodel'
check_md5_and_download $md5 $filename $url


url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZzJuaFRIdDMtLWc'
md5='eb33103e36f299b4433c63fcfc165cbd'
filename='vgg_cnn_m_1024.chainermodel'
check_md5_and_download $md5 $filename $url


url='https://drive.google.com/uc?id=0B9P1L--7Wd2vWFZtS3Zob2VTYmc'
md5='6056a4bf968e6a49fab2a568802ac254'
filename='caffenet.chainermodel'
check_md5_and_download $md5 $filename $url
