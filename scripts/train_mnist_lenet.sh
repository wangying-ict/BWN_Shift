#!/usr/bin/env bash

python ../examples/classifier_mnist/ttq_main.py ~/dataset/mnist \
    -a mnist_lenet -j 10 -b 128 -p 20 --epochs 100 \
    --gpu $1 --log-name $2 --lr 1e-4 --cosine --debug --pretrained
