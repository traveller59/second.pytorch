"""Run inference on CSI dataset"""
from pathlib import Path
import re

import cv2
from google.protobuf import text_format
import lupa
import numpy as np
import torch

from second.data.kitti_dataset import _homogeneous_coords
from second.data import image_utils
from second.protos import pipeline_pb2
from second.pytorch.train import build_network
from second.utils import config_tool

LUA_RUNTIME = lupa.LuaRuntime()
LUA_REGEX = re.compile("^return (.*)$", re.DOTALL)


def parse_lua(text):
    return LUA_RUNTIME.eval(re.fullmatch(LUA_REGEX, text)[1])


def read_rig_calibration(cal_lua: Path):
    cal = parse_lua(cal_lua.read_text())
    baseline = cal.baseline or cal.baseline_m
    cam_intrincis = cal.c or cal.cam_intrinsics
    fx = cam_intrincis.fx
    fy = cam_intrincis.fy
    cx = cam_intrincis.cx
    cy = cam_intrincis.cy
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline}


# tweaked version of kitti_dataset._create_reduced_point_cloud
def point_cloud_from_image_and_disparity(rgb, disp, fx, cx, cy, baseline):
    """Compute point cloud from image and disparity"""
    h, w = disp.shape
    xs, ys = _homogeneous_coords(
        w, h, focal_length=fx, optical_center_x=cx, optical_center_y=cy,
    )
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    valid_idx = disp > 0
    zs = fx * baseline / disp[valid_idx]
    xs = xs[valid_idx] * zs
    ys = ys[valid_idx] * zs

    # individual color channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    # make reflections visible in viewer
    colors = np.hstack(
        [
            b[valid_idx].reshape((-1, 1)),
            g[valid_idx].reshape((-1, 1)),
            r[valid_idx].reshape((-1, 1)),
        ]
    )
    points = np.hstack([xs.reshape((-1, 1)), ys.reshape((-1, 1)), zs.reshape((-1, 1))])
    points = np.hstack([points, colors]).astype(np.float32)

    # downsample point clouds density
    points = points[0::4, ...]
    return points


def detect_bboxes(csi_path: Path, checkpoint_path: Path, config_path: Path):
    # read calibration
    cal_file = csi_path / "calibration/rectified_calibration.lua"
    cal = read_rig_calibration(cal_file)

    # create csi image stack readers
    disp_reader = image_utils.RawReader(
        image_utils.find_image_stack(
            csi_path / "training", "_train_x_disp.left.scale16"
        )
    )
    rgb_reader = image_utils.RawReader(
        image_utils.find_image_stack(csi_path / "training", "_train_rgb.left.sqrt")
    )

    # create network
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    config_tool.change_detection_range(model_cfg, [-50, -50, 50, 50])
    device = torch.device("cuda")
    net = build_network(model_cfg).to(device).eval()
    net.load_state_dict(torch.load(checkpoint_path))
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    # generate anchors
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)

    # detect bboxes
    num_frames = disp_reader.total_frames
    for ix in range(num_frames):
        print(f"Processing frame {ix} of {num_frames}...")

        # load frame data
        rgb = cv2.resize(
            np.squeeze(rgb_reader.get_frames(ix, 1).astype(np.float32) * (1.0 / 255.0)),
            (disp_reader.w, disp_reader.h),
        )
        disp = np.squeeze(disp_reader.get_frames(ix, 1).astype(np.float32))

        # process point cloud
        points = point_cloud_from_image_and_disparity(
            rgb, disp, cal["fx"], cal["cx"], cal["cy"], cal["baseline"]
        )

        # generate voxels
        voxel_dict = voxel_generator.generate(points, max_voxels=90000)
        voxels = voxel_dict["voxels"]
        coords = voxel_dict["coordinates"]
        num_points = voxel_dict["num_points_per_voxel"]
        coords = np.pad(coords, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=device)

        # make detection
        example = {
            "anchors": anchors,
            "voxels": voxels,
            "num_points": num_points,
            "coordinates": coords,
        }
        with torch.no_grad():
            prediction = net(example)[0]

        boxes = prediction["box3d_lidar"].detach().cpu().numpy()
        print(boxes)


if __name__ == "__main__":
    detect_bboxes(
        Path("/host/data/csi-stacks/csi-200306-414905"),
        Path("/host/data/second/pp_disp_rgb_3classes_model/voxelnet-296960.tckpt"),
        Path("/host/data/second/pp_disp_rgb_3classes_model/pipeline.config"),
    )
