#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python ./pytorch/train.py train --config_path=./configs/CPFN/car/xyres_16.proto --model_dir=/home/aldno/pycharm-project/CPFN-main/model_dir/model_dir