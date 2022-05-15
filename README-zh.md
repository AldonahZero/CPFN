# CPFN
[![](https://img.shields.io/badge/%E4%B8%BB%E9%A1%B5-ITcyx%2FChineseREADME-orange)](https://gitee.com/ITcyx/ChineseREADME)
README 文件通常是代码的第一个入口点。它应该告诉别人如何安装它，以及如何使用它。标准化编写 README 的方式可简化创建和维护你的 README 。

## 开始

This is a fork of [SECOND for KITTI object detection](https://github.com/traveller59/second.pytorch) and the relevant
subset of the original README is reproduced here.

### Code Support

ONLY supports python 3.8+, pytorch 1.4.1+. Code has only been tested on Ubuntu 18.04/20.04.

### Install

#### 1. Clone code

```bash
git clone https://github.com/AldonahZero/CPFN.git
```

#### 2. Install Python packages

It is recommend to use the Anaconda package manager.

First, use Anaconda to configure as many packages as possible.
```bash
conda create -n pointpillars python=3.7 anaconda
source activate pointpillars
conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision -c pytorch
conda install google-sparsehash -c bioconda
```

Then use pip for the packages missing from Anaconda.
```bash
pip install --upgrade pip
pip install fire tensorboardX
```

Finally, install SparseConvNet. This is not required for PointPillars, but the general SECOND code base expects this
to be correctly configured. 
```bash
git clone git@github.com:facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash build.sh
# NOTE: if bash build.sh fails, try bash develop.sh instead
```

Additionally, you may need to install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```


#### 3. Setup cuda for numba

You need to add following environment variables for numba to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

#### 4. PYTHONPATH

Add second.pytorch/ to your PYTHONPATH.

### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```


training目录是有标注的数据（标注数据目录是label_2） training和evaluate用的都是training目录下的数据

testing目录下面都是没有标注的数据，可以用来测测模型的检测效果（需要自己写可视化代码看看最终的检测结果）。

Note: PointPillar's protos use ```KITTI_DATASET_ROOT=/home/aldno/dataset/kitti_second/```.


#### 2. 创建info数据:

在工程根目录下cd second，通过下面几条命令创建info数据：

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

在KITTI_DATASET_ROOT目录下创建kitti_infos_train.pkl、kitti_infos_val.pkl、kitti_infos_trainval.pkl、kitti_infos_test.pkl四个bin文件

每个bin文件包含了图片的路径、calib中camera和lidar的标定参数等。
#### 3. 创建精简点云:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

在KITTI_DATASET_ROOT/training/velodyne_reduced和KITTI_DATASET_ROOT/testing/velodyne_reduced目录下分别创建它们同级velodyne目录中点云bin文件的reduce版本

去掉了点云数据中一些冗余的背景等数据，可以认为是经过裁剪的点云数据。


#### 4. Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

在KITTI_DATASET_ROOT目录下创建kitti_dbinfos_train.pkl数据，在训练代码中的input_cfg.database_sampler变量中用到。

#### 5. 修改配置文件

The config file needs to be edited to point to the above datasets:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```


### 训练

```bash
cd ~/second.pytorch/second
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* If you want to train a new model, make sure "/path/to/model_dir" doesn't exist.
* If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
* Training only supports a single GPU. 
* Training uses a batchsize=2 which should fit in memory on most standard GPUs.
* On a single 1080Ti, training xyres_16 requires approximately 20 hours for 160 epochs.


### 评估


```bash
cd ~/second.pytorch/second/
python pytorch/train.py evaluate --config_path= configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.

### 白宝典



### 常见错误
ImportError: Python version mismatch: module was compiled for Python 3.8, but the interpreter version is incompatible: 3.7.13 (default, Mar 29 2022, 02:18:16) 
[GCC 7.5.0].
ImportError: Python version mismatch: module was compiled for Python 3.7, but the interpreter version is incompatible: 3.8.13 (default, Mar 28 2022, 11:38:47) 
[GCC 7.5.0].
