#!/usr/bin/env bash

python ../examples/classifier_svhn/ttq_main.py ~/dataset/svhn \
    -a svhn_resnet18 -j 10 -b 128 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr 1e-4 --cosine --debug --pretrained
