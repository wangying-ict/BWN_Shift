#!/usr/bin/env bash

python ../examples/classifier_svhn/ttq_main.py ~/datasets/data.svhn \
    -a svhn_resnet18_bwn_all -j 10 -b 128 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr $3 --cosine --debug --pretrained
