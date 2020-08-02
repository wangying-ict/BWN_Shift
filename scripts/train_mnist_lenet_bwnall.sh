#!/usr/bin/env bash

python ../examples/classifier_mnist/ttq_main.py ~/dataset/mnist \
    -a mnist_lenet_bwn_all -j 10 -b 128 -p 20 --epochs 100 \
    --gpu $1 --log-name $2 --lr $3 --cosine --debug --pretrained
