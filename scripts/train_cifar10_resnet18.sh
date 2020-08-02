#!/usr/bin/env bash
python ../examples/classifier_cifar10/ttq_main.py ~/datasets/data.cifar10 \
    -a cifar10_resnet18 -j 10 -b 128 -p 20 --epochs 400 \
    --gpu $1 --log-name $2 --lr 0.01 --debug --pretrain
# 95.28
