import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, max_sweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    if max_sweeps == 0:
        create_groundtruth_database("NuScenesDataset", root_path, Path(root_path) / "infos_train.pkl")
    else:
        print("WARNING: ground truth database will be disabled because sweeps don't support this.")

if __name__ == '__main__':
    # root_path = "/media/yy/My Passport/datasets/nuscene/v1.0-mini"
    # root_path = "/media/yy/software/datasets/v1.0-mini"
    # out_path = "/media/yy/960evo/datasets/nuscene/v1.0-mini"
    # nuscenes_data_prep(root_path, "v1.0-mini")
    fire.Fire()
