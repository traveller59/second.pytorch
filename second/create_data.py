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

def nuscenes_data_prep(root_path, version):
    nu_ds.create_nuscenes_infos(root_path, version=version)
    create_groundtruth_database("NuScenesDataset", root_path, Path(root_path) / "infos_train.pkl")

if __name__ == '__main__':
    fire.Fire()
