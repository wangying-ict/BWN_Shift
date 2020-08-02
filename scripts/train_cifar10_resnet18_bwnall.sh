#!/usr/bin/env bash

python ../examples/classifier_cifar10/ttq_main.py ~/datasets/data.cifar10 \
    -a cifar10_resnet18_bwn_all -j 10 -b 128 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr 1e-4 --cosine --debug --pretrained
