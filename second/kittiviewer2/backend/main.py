"""This backend now only support lidar. camera is no longer supported.
"""

import base64
import datetime
import io as sysio
import json
import pickle
import time
from pathlib import Path

import fire
import torch
import numpy as np
import skimage
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.protobuf import text_format
from skimage import io

from second.core import box_np_ops
from second.data import kitti_common as kitti
from second.data.all_dataset import get_dataset_class
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.inference import TorchInferenceContext
from second.pytorch.train import example_convert_to_torch

from second.builder import target_assigner_builder, voxel_builder

app = Flask("CPMNet")
CORS(app)

class SecondBackend:
    def __init__(self):
        self.root_path = None
        self.image_idxes = None
        self.dt_annos = None
        self.dataset = None
        self.net = None
        self.device = None
        self.info_path = None
        self.kitti_infos = None
        self.inference_ctx = None


BACKEND = SecondBackend()

def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    print("[ERROR]" + msg)
    return response


@app.route('/api/readinfo', methods=['POST'])
def readinfo():
    global BACKEND
    instance = request.json
    root_path = Path(instance["root_path"])
    response = {"status": "normal"}
    if not (root_path / "training").exists():
        response["status"] = "error"
        response["message"] = "ERROR: your root path is incorrect."
        print("ERROR: your root path is incorrect.")
        return response
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])
    if not info_path.exists():
        response["status"] = "error"
        response["message"] = "ERROR: info file not exist."
        print("ERROR: your root path is incorrect.")
        return response
    BACKEND.info_path = info_path

    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    BACKEND.kitti_infos = kitti_infos
    BACKEND.image_idxes = [info["image_idx"] for info in kitti_infos]
    response["image_indexes"] = BACKEND.image_idxes

    dataset_class_name = instance["dataset_class_name"]
    BACKEND.dataset = get_dataset_class(dataset_class_name)(root_path=root_path, info_path=info_path)
    BACKEND.image_idxes = list(range(len(BACKEND.dataset)))
    response["image_indexes"] = BACKEND.image_idxes
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/read_detection', methods=['POST'])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if Path(det_path).is_file():
        with open(det_path, "rb") as f:
            dt_annos = pickle.load(f)
    else:
        dt_annos = kitti.get_label_annos(det_path)
    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    enable_int16 = instance["enable_int16"]

    idx = BACKEND.image_idxes.index(image_idx)
    sensor_data = BACKEND.dataset.get_sensor_data(idx)

    # img_shape = image_info["image_shape"] # hw
    if 'annotations' in sensor_data["lidar"]:
        annos = sensor_data["lidar"]['annotations']
        gt_boxes = annos["boxes"].copy()
        response["locs"] = gt_boxes[:, :3].tolist()
        response["dims"] = gt_boxes[:, 3:6].tolist()
        rots = np.concatenate([np.zeros([gt_boxes.shape[0], 2], dtype=np.float32), -gt_boxes[:, 6:7]], axis=1)
        response["rots"] = rots.tolist()
        response["labels"] = annos["names"].tolist()
    # response["num_features"] = sensor_data["lidar"]["points"].shape[1]
    response["num_features"] = 3
    points = sensor_data["lidar"]["points"][:, :3]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")

    # if "score" in annos:
    #     response["score"] = score.tolist()
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("send response with size {}!".format(len(pc_str)))
    return response

@app.route('/api/get_image', methods=['POST'])
def get_image():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    query = {
        "lidar": {
            "idx": idx
        },
        "cam": {}
    }
    sensor_data = BACKEND.dataset.get_sensor_data(query)
    if "cam" in sensor_data and "data" in sensor_data["cam"] and sensor_data["cam"]["data"] is not None:
        image_str = sensor_data["cam"]["data"]
        response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
        response["image_b64"] = 'data:image/{};base64,'.format(sensor_data["cam"]["datatype"]) + response["image_b64"]
        print("send an image with size {}!".format(len(response["image_b64"])))
    else:
        response["image_b64"] = ""
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/build_network', methods=['POST'])
def build_network():
    global BACKEND
    instance = request.json
    cfg_path = Path(instance["config_path"])
    ckpt_path = Path(instance["checkpoint_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    if not cfg_path.exists():
        return error_response("config file not exist.")
    if not ckpt_path.exists():
        return error_response("ckpt file not exist.")
    BACKEND.inference_ctx = TorchInferenceContext()
    BACKEND.inference_ctx.build(str(cfg_path))
    BACKEND.inference_ctx.restore(str(ckpt_path))
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("build_network successful!")
    return response


@app.route('/api/inference_by_idx', methods=['POST'])
def inference_by_idx():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    if BACKEND.inference_ctx is None:
        return error_response("inference_ctx is not loaded")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    kitti_info = BACKEND.kitti_infos[idx]

    v_path = str(Path(BACKEND.root_path) / kitti_info['velodyne_path'])
    num_features = 4
    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_features])
    rect = kitti_info['calib/R0_rect']
    P2 = kitti_info['calib/P2']
    Trv2c = kitti_info['calib/Tr_velo_to_cam']
    if 'img_shape' in kitti_info:
        image_shape = kitti_info['img_shape']
        points = box_np_ops.remove_outside_points(
            points, rect, Trv2c, P2, image_shape)
        print(points.shape[0])
    img_shape = kitti_info["img_shape"] # hw
    wh = np.array(img_shape[::-1])
    whwh = np.tile(wh, 2)

    t = time.time()
    inputs = BACKEND.inference_ctx.get_inference_input_dict(
        kitti_info, points)
    print("input preparation time:", time.time() - t)
    t = time.time()
    with BACKEND.inference_ctx.ctx():
        dt_annos = BACKEND.inference_ctx.inference(inputs)[0]
    print("detection time:", time.time() - t)

    dt_annos = dt_annos[0]
    print(dt_annos)

    dims = dt_annos['dimensions']
    num_obj = dims.shape[0]
    loc = dt_annos['location']
    rots = dt_annos['rotation_y']
    labels = dt_annos['name']
    bbox = dt_annos['bbox'] / whwh

    dt_boxes_camera = np.concatenate(
        [loc, dims, rots[..., np.newaxis]], axis=1)
    dt_boxes = box_np_ops.box_camera_to_lidar(
        dt_boxes_camera, rect, Trv2c)
    box_np_ops.change_box3d_center_(dt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    locs = dt_boxes[:, :3]
    dims = dt_boxes[:, 3:6]
    rots = np.concatenate([np.zeros([num_obj, 2], dtype=np.float32), -dt_boxes[:, 6:7]], axis=1)
    response["dt_locs"] = locs.tolist()
    response["dt_dims"] = dims.tolist()
    response["dt_rots"] = rots.tolist()
    response["dt_labels"] = labels.tolist()
    response["dt_scores"] = dt_annos["score"].tolist()
    response["dt_bbox"] = bbox.tolist()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
