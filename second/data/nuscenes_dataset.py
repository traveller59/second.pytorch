from pathlib import Path
import pickle
import time
from functools import partial
from copy import deepcopy
import numpy as np
import fire
import json

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.data.dataset import Dataset
from second.utils.progress_bar import progress_bar_iter as prog_bar


class NuScenesDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            self._nusc_infos = pickle.load(f)
        self._num_point_features = 4
        self._class_names = class_names
        self._prep_func = prep_func
        self._name_mapping = {
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.wheelchair': 'ignore',
            'human.pedestrian.stroller': 'ignore',
            'human.pedestrian.personal_mobility': 'ignore',
            'human.pedestrian.police_officer': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'animal': 'ignore',
            'vehicle.car': 'car',
            'vehicle.motorcycle': 'motorcycle',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.truck': 'truck',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.emergency.ambulance': 'ignore',
            'vehicle.emergency.police': 'ignore',
            'vehicle.trailer': 'trailer',
            'movable_object.barrier': 'barrier',
            'movable_object.trafficcone': 'traffic_cone',
            'movable_object.pushable_pullable': 'ignore',
            'movable_object.debris': 'ignore',
            'static_object.bicycle_rack': 'ignore',
        }
        self._kitti_name_mapping = {}
        for k, v in self._name_mapping.items():
            if v.lower() in ["car", "pedestrian"
                             ]:  # we only eval these classes in kitti
                self._kitti_name_mapping[k] = v
        self.version = "v1.0-trainval"
        self.eval_version = "cvpr_2019"

    def __len__(self):
        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        from nuscenes.eval.detection.config import eval_detection_configs
        cls_range_map = eval_detection_configs[self.
                                               eval_version]["class_range"]
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = info["gt_names"]
            gt_boxes = info["gt_boxes"]
            mask = np.array([n in self._kitti_name_mapping for n in gt_names],
                            dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            gt_names_mapped = [self._kitti_name_mapping[n] for n in gt_names]
            det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append({
                "bbox":
                np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha":
                np.full(N, -10),
                "occluded":
                np.zeros(N),
                "truncated":
                np.zeros(N),
                "name":
                gt_names[mask],
                "location":
                gt_boxes[mask][:, :3],
                "dimensions":
                gt_boxes[mask][:, 3:6],
                "rotation_y":
                gt_boxes[mask][:, 6],
            })
        return gt_annos

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        assert isinstance(query, int)
        info = self._nusc_infos[query]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }

        lidar_path = Path(info['lidar_path'])
        points = np.fromfile(
            str(lidar_path), dtype=np.float32, count=-1).reshape([-1,
                                                                  5])[:, :4]
        points[:, -1] /= 255
        # mask = box_np_ops.points_in_rbbox(points, info["gt_boxes"]).any(-1)
        # points = points[~mask]
        res["lidar"]["points"] = points
        if 'gt_boxes' in info:
            res["lidar"]["annotations"] = {
                'boxes': info["gt_boxes"],
                'names': info["gt_names"],
            }
        return res

    def evaluation_kitti(self, detections, output_dir):
        """eval by kitti evaluation tool
        """
        class_names = self._class_names
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        gt_annos = deepcopy(gt_annos)
        detections = deepcopy(detections)
        dt_annos = []
        for det in detections:
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 50, 50]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_lidar[j, 3:6])
                anno["location"].append(box3d_lidar[j, :3])
                anno["rotation_y"].append(box3d_lidar[j, 6])
                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                dt_annos.append(anno)
            else:
                dt_annos.append(kitti.empty_result_anno())
            num_example = dt_annos[-1]["name"].shape[0]
            dt_annos[-1]["metadata"] = det["metadata"]

        for anno in gt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self._name_mapping:
                    mapped_names.append(self._name_mapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        for anno in dt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self._name_mapping:
                    mapped_names.append(self._name_mapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        z_axis = 2
        z_center = 0.5
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
            mapped_class_names,
            z_axis=z_axis,
            z_center=z_center)
        result_coco = get_coco_eval_result(
            gt_annos,
            dt_annos,
            mapped_class_names,
            z_axis=z_axis,
            z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco["result"],
            },
            "detail": {
                "official": result_official_dict["detail"],
                "coco": result_coco["detail"],
            },
        }

    def evaluation_nusc(self, detections, output_dir):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        nusc_annos = {}
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version=version, dataroot=str(self._root_path), verbose=False)
        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in detections:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes,
                                              det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": [0.0, 0.0],
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": '',
                }
                annos.append(nusc_anno)
            nusc_annos[det["metadata"]["token"]] = annos
        res_path = str(Path(output_dir) / "results_nusc.json")
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        from nuscenes.eval.detection.evaluate import main as eval_main
        eval_main(
            res_path,
            output_dir,
            eval_set=eval_set_map[version],
            dataroot=str(self._root_path),
            version=version,
            verbose=False,
            config_name=self.eval_version,
            plot_examples=0)
        with open(Path(output_dir) / "metrics.json", "r") as f:
            metrics = json.load(f)
        detail = {}
        result = f"Nusc {version} Evaluation\n"
        for name in mapped_class_names:
            detail[name] = {}
            for k, v in metrics["label_aps"][name].items():
                detail[name][f"dist@{k}"] = v
            threshs = ', '.join(list(metrics["label_aps"][name].keys()))
            scores = list(metrics["label_aps"][name].values())
            scores = ', '.join([f"{s * 100:.2f}" for s in scores])
            result += f"{name} Nusc dist AP@{threshs}\n"
            result += scores
            result += "\n"
        return {
            "results": {
                "nusc": result
            },
            "detail": {
                "nusc": detail
            },
        }

    def evaluation(self, detections, output_dir):
        res_kitti = self.evaluation_kitti(detections, output_dir)

        res_nusc = self.evaluation_nusc(detections, output_dir)
        res = {
            "results": {
                "nusc": res_nusc["results"]["nusc"],
                "kitti.official": res_kitti["results"]["official"],
                "kitti.coco": res_kitti["results"]["coco"],
            },
            "detail": {
                "eval.nusc": res_nusc["detail"]["nusc"],
                "eval.kitti": {
                    "official": res_kitti["detail"]["official"],
                    "coco": res_kitti["detail"]["coco"],
                },
            },
        }
        return res


def _second_det_to_nusc_box(detection):
    from nuscenes.utils.data_classes import Box
    import pyquaternion
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i])
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    import pyquaternion
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record["data"]["LIDAR_TOP"]
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False):
    train_nusc_infos = []
    val_nusc_infos = []
    for sample in prog_bar(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path, boxes, cam_intrinsic = nusc.get_sample_data(lidar_token)
        if Path(lidar_path).exists():
            info = {
                "lidar_path": lidar_path,
                "token": sample["token"],
            }
            if not test:
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array(
                    [b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                        -1, 1)
                names = np.array([b.name for b in boxes])
                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2],
                                          axis=1)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = names
            if sample["scene_token"] in train_scenes:
                train_nusc_infos.append(info)
            else:
                val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos


def create_nuscenes_infos(root_path, version="v1.0-trainval"):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test)
    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(root_path / "infos_test.pkl", 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(root_path / "infos_train.pkl", 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(root_path / "infos_val.pkl", 'wb') as f:
            pickle.dump(val_nusc_infos, f)


def create_nuscenes_infos_custom(
        root_path,
        version="v1.0-trainval",
        split_rate=0.82353,  # 700 / 850
        test=False):
    """Don't use this because official evaluation tool don't support custom
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    num_train_scene = np.round(split_rate * len(available_scenes)).astype(
        np.int64)
    train_scenes = set(
        [s["token"] for s in available_scenes[:num_train_scene]])
    val_scenes = set([s["token"] for s in available_scenes[num_train_scene:]])
    if test:
        train_scenes = set([s["token"] for s in nusc.scene])
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes)
    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(root_path / "infos_test.pkl", 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(root_path / "infos_train.pkl", 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(root_path / "infos_val.pkl", 'wb') as f:
            pickle.dump(val_nusc_infos, f)


def get_box_mean(info_path, class_name="vehicle.car"):
    with open(info_path, 'rb') as f:
        nusc_infos = pickle.load(f)

    gt_boxes_list = []
    for info in nusc_infos:
        mask = np.array([s == class_name for s in info["gt_names"]],
                        dtype=np.bool_)
        gt_boxes_list.append(info["gt_boxes"][mask].reshape(-1, 7))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    print(gt_boxes_list.mean(0))


if __name__ == "__main__":
    # create_nuscenes_infos("/media/yy/My Passport/datasets/nuscene/v1.0-mini",
    #                       "v1.0-mini")
    fire.Fire()