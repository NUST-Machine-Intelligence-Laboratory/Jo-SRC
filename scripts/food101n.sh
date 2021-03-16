#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=0
python main.py --config config/food101n.cfg --gpu ${GPU}
