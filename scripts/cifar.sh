#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=0
python cifar100.py --config config/cifar100.cfg --gpu ${GPU}
