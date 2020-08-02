# BWN_Shift

- Codes for “Towards State-Aware Computation in ReRAM Neural Networks,”, Yintao He, Ying Wang, Xiandong Zhao, Huawei Li, Xiaowei Li, in IEEE/ACM Proceedings of Design, Automation Conference (DAC), 2020.



Install
------------

```bash
git clone <.git>

cd BWN_Shift

export PYTHONPATH=$PYTHONPATH:`pwd`

pip install -r requirements.txt
```

Codes
-----
The folder contains three important parts:
- `utils/` contains scripts help us get started quickly
- `examples/` contains \*.py for training and validation of network
- `models/` contains files of BWN-Shift, BWN without Shift, and full-precision model structure

Run
-------

Run scripts with following (available dataset: "mnist", "cifar10", "svhn"):
```
cd scripts
./[scripts_name].sh [GPU_id] [log_name] [learning_rate]
```
1. Obtain full-precision model for follow-up binary training
```bash
./train_mnist_lenet.sh 0 baseline 1e-3
```
2. Train BWN-Shift model
```bash
./train_mnist_lenet_bwnshift.sh 0 test 1e-8
```
3. Train BWN without Shift to compare
```bash
./train_mnist_lenet_bwnall.sh 0 test 1e-5
```


* All the models will be stored in /scripts/logger
* All the layers has been binarized in this work
