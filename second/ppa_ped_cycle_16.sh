#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python ./pytorch/train.py train --config_path=./configs/CPFN/ped_cycle/xyres_16.proto --model_dir=/home/aldno/pycharm-project/CPFN-main/model_dir/ped_cycle_model_dir2  --refine_weight 2


