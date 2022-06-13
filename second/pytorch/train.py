import os
import pathlib
import pickle
import shutil
import time
from functools import partial

import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar

from second.pytorch.models import fusion

# 压制警告
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted

# example 转化为 torch
# example_convert_to_torch函数把加载的example数据都转到gpu上，内容和变量example是一样的。
# exampLe包含下列信息
# voxels	[54786,5,4]	最大含有54786个voxels（54786是3个batch所有点的个数），每个点中最多5个点，每个点4个维度信息
# num_points	[54786,]	一个batch中所有点的个数
# coordinates	[54786,4]	每一个点对应的voxel坐标,4表示的是[bs,x,y,z]
# num_voxels	[3,1]	稀疏矩阵的参数[41,1280,1056]
# metrics	list类型，长度为3，[gen_time,prep_time]	衡量时间
# calib	dict类型，长度为3,[rect,Trv2c,P2]	二维到点云变换矩阵
# anchors	[3,168960,7]	3是bs,168960=1601328*4,7是回归维度
# gt_names	[67,]	这里的67的含义是在这个batch中有67个gt，names={car，cyslist,pedestrain,car}
# labels	[3,168960]	每一个anchor的lable
# reg_targes	[3,168960,7]	reg所对应的真实的gt
# importance	[3,1689660]
# metadata	list,长度为3
def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    '''
    convert example dict to tensor
    :param example:
    :param dtype:
    :param device:
    :return:
    '''
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2", "gt_boxes"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch


def build_inference_net(config_path,
                        model_dir,
                        result_path=None,
                        predict_test=False,
                        ckpt_path=None,
                        ref_detfile=None,
                        pickle_result=True,
                        measure_time=False,
                        batch_size=1):
    '''

    :param config_path:
    :param model_dir:
    :param result_path:
    :param predict_test:
    :param ckpt_path:
    :param ref_detfile:
    :param pickle_result:
    :param measure_time:
    :param batch_size:
    :return:
    '''
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    # detection_2d_path = config.train_config.detection_2d_path
    center_limit_range = model_cfg.post_center_limit_range
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    class_names = target_assigner.classes
    net = second_builder.build(
        model_cfg,
        voxel_generator,
        target_assigner,
        measure_time=measure_time)
    net.cuda()

    if ckpt_path is None:
        print("load existing model")
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or input_cfg.batch_size
    #batch_size = 1
    net.eval()
    return net


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pickle_result=True,
          refine_weight=2):
    '''
    训练函数
    :param config_path:配置文件路径
    :param model_dir:模型路径
    :param result_path:指定评估结果文件夹
    :param create_folder:是否新建文件夹
    :param display_step:每多少步数输出一次
    :param summary_step:
    :param pickle_result:
    :param refine_weight:优化权重
    :return:
    '''

    # 创建模型保存地址，读取预处理后的数据形式：
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    #  级联创建文件夹
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    # 输入cfg
    input_cfg = config.train_input_reader
    # 评估输入cfg
    eval_input_cfg = config.eval_input_reader
    # 模型cfg
    model_cfg = config.model.second
    # 训练cfg
    train_cfg = config.train_config
    # 2d识别结果存放
    # detection_2d_path = config.train_config.detection_2d_path
    # print("2d detection path:", detection_2d_path)
    center_limit_range = model_cfg.post_center_limit_range

    # 待识别类 car per cye
    class_names = list(input_cfg.class_names)
    ######################
    # BUILD VOXEL GENERATOR
    # 构建体素生成器
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    # 加载检测目标文件
    # 这里的分类目标包括有
    # [car, pedestrian, cyclist, van]
    # 同时的四类检测问题，以前分别对一种做检测时，需要设置同一种的anchor_size，然后只需要做二分类任务，如果是多类检测，应该需要如上四种不同类别的anchor和不同的size,在随后的检测时需要输出其对应的类别。
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    # 构建网络
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    # net_train = torch.nn.DataParallel(net).cuda()

    # 可训练参数数量
    print("num_trainable parameters:", len(list(net.parameters())))
    for n, p in net.named_parameters():
        print(n, p.shape)

    # 首先尝试从最近的checkpoints恢复网络。
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    # 回退一部全局步数
    gstep = net.get_global_step() - 1

    ######################
    # BUILD OPTIMIZER
    # 优化器
    # 设置优化器和损失函数
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    # 需要 global_step 来创建 lr_scheduler，所以 如果存在model_dir的话
    optimizer_cfg = train_cfg.optimizer

    # 单精度 or 混合精度
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    # 构建优化器
    optimizer = optimizer_builder.build(optimizer_cfg, net.parameters())
    # 如果是混合精度 就用混合精度的优化器
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    # must restore optimizer AFTER using MixedPrecisionWrapper
    # 必须在使用 MixedPrecisionWrapper 之后恢复优化器
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [mixed_optimizer])
    # 需要 global_step 来创建 lr_scheduler
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, gstep)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################
    # PREPARE INPUT
    # 准备输入
    ######################

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    # 创建数据加载器
    # dataset：（数据类型 dataset）
    # batch_size：（int）批训练数据量的大小默认：1）PyTorch训练模型时调用数据不是一行一行进行的，而是一捆一捆来的。这里就是定义每次喂给神经网络多少行数据，如果设置成1，那就是一行一行进行）
    # shuffle：（数据类型 bool）是否打乱数据，默认为False）
    # num_workers：（int）数据加载器的线程数，默认为0）
    # pin_memory：（数据类型 bool）是否将数据加载到GPU上，默认为False）
    # collate_fn：（数据类型 callable，没见过的类型）将一小段数据合并成数据列表，默认设置是False。如果设置成True，系统会在返回前会将张量数据（Tensors）复制到CUDA内存中。
    # worker_init_fn：（数据类型
    # callable，没见过的类型）每个线程的初始化函数，默认设置是None。子进程导入模式，默认为Noun。在数据导入前和步长结束后，根据工作子进程的ID逐个按顺序导入数据。
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)
    data_iter = iter(dataloader)

    ######################
    # TRAINING
    # 训练
    ######################
    # 日志文件
    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a')
    # 向日志里写入config配置内容
    logf.write(proto_str)
    logf.write("\n")
    # summary dir
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    # 剩余的步数
    remain_steps = train_cfg.steps - net.get_global_step()

    t = time.time()
    ckpt_start_time = t

    #  整除得到总循环次数
    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    # 每epoch是否清除指标
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    # 到了评估阶段
    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()
    try:
        for _ in range(total_loop):
            if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
                steps = train_cfg.steps % train_cfg.steps_per_eval
            else:
                steps = train_cfg.steps_per_eval
            for step in range(steps):
                lr_scheduler.step()
                try:
                    # 从迭代器返回下一项。 如果给出默认值并且迭代器用尽，它引发StopIteration错误
                    example = next(data_iter)
                except StopIteration:
                    print("end epoch")
                    # 如果清除指标
                    if clear_metrics_every_epoch:
                        net.clear_metrics()

                    # 从对象中获取迭代器。 在第一种形式中，参数必须提供它自己的迭代器，或者是一个序列。在第二种形式中，调用 callable 直到它返回哨兵。
                    # 这里是第一种形式
                    data_iter = iter(dataloader)

                    # example包含如下信息：
                    # voxels	[54786,5,4]	最大含有54786个voxels（54786是3个batch所有点的个数），每个点中最多5个点，每个点4个维度信息
                    # num_points	[54786,]	一个batch中所有点的个数
                    # coordinates	[54786,4]	每一个点对应的voxel坐标,4表示的是[bs,x,y,z]
                    # num_voxels	[3,1]	稀疏矩阵的参数[41,1280,1056]
                    # metrics	list类型，长度为3，[gen_time,prep_time]	衡量时间
                    # calib	dict类型，长度为3,[rect,Trv2c,P2]	二维到点云变换矩阵
                    # anchors	[3,168960,7]	3是bs,168960=1601328*4,7是回归维度
                    # 最终经过中间层后提取的feature_map 大小是[ 3 , 8 , 130 , 132 ] [3,8,130,132][3,8,130,132]这里的8表示两个方向*4个类别
                    # 然后每一种anchor预测得到一个回归值和一个分类值，然后也就是8 × 130 × 132 = 168960
                    # gt_names	[67,]	这里的67的含义是在这个batch中有67个gt，names={car，cyslist,pedestrain,car}
                    # labels	[3,168960]	每一个anchor的lable
                    # reg_targes	[3,168960,7]	reg所对应的真实的gt
                    # importance	[3,1689660]
                    # metadata	list,长度为3
                    example = next(data_iter)

                # example_convert_to_torch函数把加载的example数据都转到gpu上，内容和变量example是一样的。
                example_torch = example_convert_to_torch(example, float_dtype)

                batch_size = example["anchors"].shape[0]

                # 过了一次这个网络。得到了ret_dict
                ret_dict = net(example_torch, refine_weight)

                # box_preds = ret_dict["box_preds"]

                # cls_preds	[3,8,160,132,4]	160 ×132 ×8=168960，表示每一个分类的得分
                cls_preds = ret_dict["cls_preds"]
                # loss	[,]	一个float，总损失的和
                loss = ret_dict["loss"].mean()
                # cls_loss_reduced	[,]	总损失/batch_size
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                # loc_loss_reduced	[,]
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                # cls_pos_loss	[,]	pos的anchor的损失
                cls_pos_loss = ret_dict["cls_pos_loss"]
                # cls_neg_loss	[,]	neg的anchor的损失
                cls_neg_loss = ret_dict["cls_neg_loss"]
                # loc_loss	[3,168960,7]	anchor的回归损失，回归定位损失
                loc_loss = ret_dict["loc_loss"]
                # cls_loss	[3,168960,4]	每一个anchor的预测分类损失，一共有4类
                cls_loss = ret_dict["cls_loss"]
                # dir_loss_reduced	[,]	方向预测损失
                dir_loss_reduced = ret_dict["dir_loss_reduced"]
                # cared	[3.168960]	猜测是被判定为pos的anchor
                cared = ret_dict["cared"]
                # 取labels，shape[3,168960]对应着每一个anchor的label，和gt存在大于阀值的IOU就会被认为是label为1
                labels = example_torch["labels"]
                if train_cfg.enable_mixed_precision:
                    loss *= loss_scale

                # 反向传播。
                loss.backward()
                # 使得最小的梯度至少都是10.采用optimizer.step()进行参数更新，
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                mixed_optimizer.step()
                mixed_optimizer.zero_grad()
                net.update_global_step()

                # loss和准确率
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                #  每训练一个step的毫秒值
                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                # 目标数量
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                # 背景数量
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step()
                if global_step % display_step == 0:
                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["step"] = global_step
                    metrics["steptime"] = step_time
                    metrics.update(net_metrics)
                    metrics["loss"] = {}
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())

                    ########################################
                    if model_cfg.rpn.module_class_name == "PSA" or model_cfg.rpn.module_class_name == "RefineDet":
                        coarse_loss = ret_dict["coarse_loss"]
                        refine_loss = ret_dict["refine_loss"]
                        metrics["coarse_loss"] = float(
                            coarse_loss.detach().cpu().numpy())
                        metrics["refine_loss"] = float(
                            refine_loss.detach().cpu().numpy())
                    ########################################
                    # if unlabeled_training:
                    #     metrics["loss"]["diff_rt"] = float(
                    #         diff_loc_loss_reduced.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())
                    metrics["num_vox"] = int(example_torch["voxels"].shape[0])
                    metrics["num_pos"] = int(num_pos)
                    metrics["num_neg"] = int(num_neg)
                    metrics["num_anchors"] = int(num_anchors)
                    metrics["lr"] = float(
                        mixed_optimizer.param_groups[0]['lr'])
                    metrics["image_idx"] = example['image_idx'][0]
                    flatted_metrics = flat_nested_json_dict(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, "/")
                    for k, v in flatted_summarys.items():
                        if isinstance(v, (list, tuple)):
                            v = {str(i): e for i, e in enumerate(v)}
                            writer.add_scalars(k, v, global_step)
                        else:
                            writer.add_scalar(k, v, global_step)
                    metrics_str_list = []
                    for k, v in flatted_metrics.items():
                        if isinstance(v, float):
                            metrics_str_list.append(f"{k}={v:.3}")
                        elif isinstance(v, (list, tuple)):
                            if v and isinstance(v[0], float):
                                v_str = ', '.join([f"{e:.3}" for e in v])
                                metrics_str_list.append(f"{k}=[{v_str}]")
                            else:
                                metrics_str_list.append(f"{k}={v}")
                        else:
                            metrics_str_list.append(f"{k}={v}")
                    log_str = ', '.join(metrics_str_list)
                    print(log_str, file=logf)
                    print(log_str)
                ckpt_elasped_time = time.time() - ckpt_start_time
                if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                    torchplus.train.save_models(model_dir, [net, optimizer],
                                                net.get_global_step())
                    ckpt_start_time = time.time()
            total_step_elapsed += steps
            torchplus.train.save_models(model_dir, [net, optimizer],
                                        net.get_global_step())

            # Ensure that all evaluation points are saved forever
            torchplus.train.save_models(
                eval_checkpoint_dir, [
                    net, optimizer], net.get_global_step(), max_to_keep=100)

            net.eval()
            result_path_step = result_path / f"step_{net.get_global_step()}"
            result_path_step.mkdir(parents=True, exist_ok=True)
            print("#################################")
            print("#################################", file=logf)
            print("# EVAL")
            print("# EVAL", file=logf)
            print("#################################")
            print("#################################", file=logf)
            print("Generate output labels...")
            print("Generate output labels...", file=logf)
            t = time.time()
            if model_cfg.rpn.module_class_name == "PSA" or model_cfg.rpn.module_class_name == "RefineDet":
                dt_annos_coarse = []
                dt_annos_refine = []
                prog_bar = ProgressBar()
                prog_bar.start(
                    len(eval_dataset) //
                    eval_input_cfg.batch_size +
                    1)
                for example in iter(eval_dataloader):
                    example = example_convert_to_torch(example, float_dtype)
                    if pickle_result:
                        coarse, refine = predict_kitti_to_anno(
                            net, example, class_names, center_limit_range,
                            model_cfg.lidar_input, use_coarse_to_fine=True)
                        dt_annos_coarse += coarse
                        dt_annos_refine += refine
                    else:
                        _predict_kitti_to_file(
                            net,
                            example,
                            result_path_step,
                            class_names,
                            center_limit_range,
                            model_cfg.lidar_input,
                            use_coarse_to_fine=True)
                    prog_bar.print_bar()
            else:
                dt_annos = []
                prog_bar = ProgressBar()
                prog_bar.start(
                    len(eval_dataset) //
                    eval_input_cfg.batch_size +
                    1)
                for example in iter(eval_dataloader):
                    example = example_convert_to_torch(example, float_dtype)
                    if pickle_result:
                        dt_annos += predict_kitti_to_anno(
                            net, example, class_names, center_limit_range,
                            model_cfg.lidar_input, use_coarse_to_fine=False)
                    else:
                        _predict_kitti_to_file(
                            net,
                            example,
                            result_path_step,
                            class_names,
                            center_limit_range,
                            model_cfg.lidar_input,
                            use_coarse_to_fine=False)

                    prog_bar.print_bar()

            sec_per_ex = len(eval_dataset) / (time.time() - t)
            print(f"avg forward time per example: {net.avg_forward_time:.3f}")
            print(
                f"avg postprocess time per example: {net.avg_postprocess_time:.3f}"
            )

            net.clear_time_metrics()
            print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
            print(
                f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                file=logf)
            gt_annos = [
                info["annos"] for info in eval_dataset.dataset.kitti_infos
            ]
            if not pickle_result:
                dt_annos = kitti.get_label_annos(result_path_step)

            if model_cfg.rpn.module_class_name == "PSA" or model_cfg.rpn.module_class_name == "RefineDet":

                print('Before Fusion:')
                result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(
                    gt_annos, dt_annos_coarse, class_names, return_data=True)
                print(result, file=logf)
                print(result)
                writer.add_text('eval_result', result, global_step)

                print('After Fusion::')
                result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(
                    gt_annos, dt_annos_refine, class_names, return_data=True)
                dt_annos = dt_annos_refine
            else:
                result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(
                    gt_annos, dt_annos, class_names, return_data=True)
            print(result, file=logf)
            print(result)
            writer.add_text('eval_result', result, global_step)

            for i, class_name in enumerate(class_names):
                writer.add_scalar('bev_ap:{}'.format(
                    class_name), mAPbev[i, 1, 0], global_step)
                writer.add_scalar('3d_ap:{}'.format(
                    class_name), mAP3d[i, 1, 0], global_step)
                writer.add_scalar('aos_ap:{}'.format(
                    class_name), mAPaos[i, 1, 0], global_step)
            writer.add_scalar('bev_map', np.mean(mAPbev[:, 1, 0]), global_step)
            writer.add_scalar('3d_map', np.mean(mAP3d[:, 1, 0]), global_step)
            writer.add_scalar('aos_map', np.mean(mAPaos[:, 1, 0]), global_step)

            result = get_coco_eval_result(gt_annos, dt_annos, class_names)
            print(result, file=logf)
            print(result)
            if pickle_result:
                with open(result_path_step / "result.pkl", 'wb') as f:
                    pickle.dump(dt_annos, f)
            writer.add_text('eval_result', result, global_step)
            net.train()
    except Exception as e:
        # 报错退出前保存模型
        torchplus.train.save_models(model_dir, [net, optimizer],
                                    net.get_global_step())
        logf.close()
        raise e
    # save model before exit
    # 退出前保存模型
    torchplus.train.save_models(model_dir, [net, optimizer],
                                net.get_global_step())
    logf.close()


def comput_kitti_output(predictions_dicts, batch_image_shape,
                        lidar_input, center_limit_range, class_names,
                        global_set):
    '''

    :param predictions_dicts:
    :param batch_image_shape:
    :param lidar_input:
    :param center_limit_range:
    :param class_names:
    :param global_set:
    :return:
    '''
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)

    return annos


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False, use_coarse_to_fine=True,
                          global_set=None):
    '''

    :param net:
    :param example:
    :param class_names:
    :param center_limit_range:
    :param lidar_input:
    :param use_coarse_to_fine:
    :param global_set:
    :return:
    '''
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']

    if use_coarse_to_fine:
        predictions_dicts_coarse, predictions_dicts_refine = net(example)
        # t = time.time()
        annos_coarse = comput_kitti_output(
            predictions_dicts_coarse,
            batch_image_shape,
            lidar_input,
            center_limit_range,
            class_names,
            global_set)
        annos_refine = comput_kitti_output(
            predictions_dicts_refine,
            batch_image_shape,
            lidar_input,
            center_limit_range,
            class_names,
            global_set)
        return annos_coarse, annos_refine
    else:
        predictions_dicts_coarse = net(example)
        annos_coarse = comput_kitti_output(
            predictions_dicts_coarse,
            batch_image_shape,
            lidar_input,
            center_limit_range,
            class_names,
            global_set)

        return annos_coarse


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False, use_coarse_to_fine=True):
    '''

    :param net:
    :param example:
    :param result_save_path:
    :param class_names:
    :param center_limit_range:
    :param lidar_input:
    :param use_coarse_to_fine:
    :return:
    '''
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    if use_coarse_to_fine:
        _, predictions_dicts_refine = net(example)
        predictions_dicts = predictions_dicts_refine
    else:
        predictions_dicts = net(example)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):
    '''

    :param config_path:配置文件路径
    :param model_dir:模型文件路径
    :param result_path:
    :param predict_test:
    :param ckpt_path:
    :param ref_detfile:
    :param pickle_result:
    :return:
    '''
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()

    if model_cfg.rpn.module_class_name == "PSA" or model_cfg.rpn.module_class_name == "RefineDet":
        dt_annos_coarse = []
        dt_annos_refine = []
        print("Generate output labels...")
        bar = ProgressBar()
        bar.start(len(eval_dataset) // input_cfg.batch_size + 1)
        for example in iter(eval_dataloader):
            example = example_convert_to_torch(example, float_dtype)
            if pickle_result:
                coarse, refine = predict_kitti_to_anno(
                    net, example, class_names, center_limit_range,
                    model_cfg.lidar_input, use_coarse_to_fine=True, global_set=None)
                dt_annos_coarse += coarse
                dt_annos_refine += refine
            else:
                _predict_kitti_to_file(
                    net,
                    example,
                    result_path_step,
                    class_names,
                    center_limit_range,
                    model_cfg.lidar_input,
                    use_coarse_to_fine=True)
            bar.print_bar()
    else:
        dt_annos = []
        print("Generate output labels...")
        bar = ProgressBar()
        bar.start(len(eval_dataset) // input_cfg.batch_size + 1)
        for example in iter(eval_dataloader):
            example = example_convert_to_torch(example, float_dtype)
            if pickle_result:
                dt_annos += predict_kitti_to_anno(net,
                                                  example,
                                                  class_names,
                                                  center_limit_range,
                                                  model_cfg.lidar_input,
                                                  use_coarse_to_fine=False,
                                                  global_set=None)
            else:
                _predict_kitti_to_file(
                    net,
                    example,
                    result_path_step,
                    class_names,
                    center_limit_range,
                    model_cfg.lidar_input,
                    use_coarse_to_fine=False)
            bar.print_bar()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

    print(f"avg forward time per example: {net.avg_forward_time:.3f}")
    print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)

        if model_cfg.rpn.module_class_name == "PSA" or model_cfg.rpn.module_class_name == "RefineDet":
            print('Before Fusion:')
            result_coarse = get_official_eval_result(
                gt_annos, dt_annos_coarse, class_names)
            print(result_coarse)

            print('After Fusion::')
            result_refine = get_official_eval_result(
                gt_annos, dt_annos_refine, class_names)
            print(result_refine)
            result = get_coco_eval_result(
                gt_annos, dt_annos_refine, class_names)
            dt_annos = dt_annos_refine
            print(result)
        else:
            result = get_official_eval_result(gt_annos, dt_annos, class_names)
            print(result)

        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)


if __name__ == '__main__':
    # 获取子任务
    fire.Fire()
