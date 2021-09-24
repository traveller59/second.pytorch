"""project 3D kitti boxes to 2D"""
import cv2
import dataclasses
import fire
import numpy as np
import pickle
import math
from scipy.spatial.transform import Rotation


@dataclasses.dataclass
class BBox3D:
    label: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    height: float
    width: float
    length: float
    det_score: float = -1

def apply_camera_matrix(camera_matrix, pts):
    if camera_matrix.shape[1] == 4:
        pts = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1)
    pts = np.matmul(camera_matrix, pts.T).T
    pts = np.stack([pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]], 1)
    return pts


def bbox3d_to_cv_points(bbox_3d, camera_matrix):
    pts_3d = []
    for i in range(8):
        h = -bbox_3d.height if i % 2 == 0 else 0  # bbox_3d.height
        w = -bbox_3d.width / 2 if (i // 2) % 2 == 0 else bbox_3d.width / 2
        l = -bbox_3d.length / 2 if (i // 4) % 2 == 0 else bbox_3d.length / 2
        pts_3d.append(np.array([w, h, l]))

    pts_3d = np.stack(pts_3d)
    R = Rotation.from_euler("YXZ", [bbox_3d.yaw, bbox_3d.pitch, bbox_3d.roll])
    R = R.as_matrix()
    T = np.array([bbox_3d.x, bbox_3d.y, bbox_3d.z])
    pts_3d = np.matmul(R, pts_3d.T).T + T.reshape([1, -1])
    pts = apply_camera_matrix(camera_matrix, pts_3d)
    center = apply_camera_matrix(camera_matrix, T.reshape([1, 3]))
    return pts, center[0]

def draw_bbox3d(img, bbox_3d, K, taxonomy=None):
    if taxonomy is not None and taxonomy.valid_class(bbox_3d.label):
        color = taxonomy.label_color(bbox_3d.label)
    else:
        color = (255, 0, 0)
    corner_pts, center_pt = bbox3d_to_cv_points(bbox_3d, K)
    corner_pts = np.round(corner_pts).astype(np.int32)
    for i in range(8):
        for j in range(8):
            if i < j and i ^ j in [1, 2, 4]:
                img = cv2.line(
                    img, corner_pts[i], corner_pts[j], color=color, thickness=1
                )

    center_pt = np.round(center_pt).astype(np.int32)
    img = cv2.circle(
        img, (center_pt[0], center_pt[1]), radius=1, color=color, thickness=-1
    )
    return img


def read_calib(calib_path):
    with open(calib_path, "r") as f:
        calib = f.readlines()
    P2 = calib[2].split()
    assert P2[0] == "P2:"
    P2 = [float(x) for x in P2[1:]]
    P3 = calib[3].split()
    assert P3[0] == "P3:"
    P3 = [float(x) for x in P3[1:]]
    K = np.array(P2).reshape([3, 4])
    baseline = (P2[3] - P3[3]) / K[0, 0]
    return K, baseline

def plot_box(bbox_3ds, rgb, K, output_path):
    for bbox_3d in bbox_3ds:
        rgb = draw_bbox3d(rgb, bbox_3d, K)
    cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    print("output to ", output_path)

def run():
    pkl_path = "/host/ssd/quan/data/tmp/kitti3d_second/disp_pp_xyres_20_models_rgb/results/step_296960/result.pkl"
    rgb_path = "/host/ssd/quan/data/tmp/kitti3d_second/training/image_2/000008.png"
    calib_path = "/host/ssd/quan/data/tmp/kitti3d_second/training/calib/000008.txt"
    output_path = "/host/tmp/0000008_boxes.png"
    xyz_hwl_r = [ 8.2277,  1.1708, -0.8107,  1.5976,  3.8149,  1.5407, -1.2763]
    with open(pkl_path, "rb") as fp:
        d = pickle.load(fp)
    bboxes = d[5]

    bbox_3ds = []
    for xyz_hwl_r, score in zip(bboxes["box3d_lidar"], bboxes["scores"]):
        if score < 0.25:
            continue
        bbox_3d = BBox3D(
                    label="car",
                    height=float(xyz_hwl_r[3]),
                    width=float(xyz_hwl_r[5]),
                    length=float(xyz_hwl_r[4]),
                    x=-float(xyz_hwl_r[1]),
                    y=-float(xyz_hwl_r[2])+float(xyz_hwl_r[3])/2,
                    z=float(xyz_hwl_r[0]),
                    yaw=float(xyz_hwl_r[6]) + math.pi / 2,
                    pitch=0,
                    roll=0,
        )
        bbox_3ds.append(bbox_3d)

    K, baseline = read_calib(calib_path)
    bgr = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plot_box(bbox_3ds, rgb, K, output_path)

if __name__ == '__main__':
    run()
