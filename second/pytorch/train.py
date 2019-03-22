import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import json 
import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter
import copy
import torchplus
from second.core import box_np_ops
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar
from second.utils.log_tool import metric_to_str, flat_nested_json_dict

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


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        else:
            example_torch[k] = v
    return example_torch

def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    class_names = target_assigner.classes
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pickle_result=True,
          resume=False):
    """train a VoxelNet model specified by a config file.
    """
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = pathlib.Path(model_dir)
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used 
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg).cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    class_names = target_assigner.classes
    
    # net_train = torch.nn.DataParallel(net).cuda()
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)
    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    mixed_optimizer = optimizer_builder.build(optimizer_cfg, net, mixed=train_cfg.enable_mixed_precision, loss_scale=loss_scale)
    optimizer = mixed_optimizer
    center_limit_range = model_cfg.post_center_limit_range
    """
    if train_cfg.enable_mixed_precision:
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    """
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [mixed_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, train_cfg.steps)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################
    # PREPARE INPUT
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
    ######################
    training_detail = []
    log_path = model_dir / 'log.txt'
    training_detail_path = model_dir / 'log.json'
    if training_detail_path.exists():
        with open(training_detail_path, 'r') as f:
            training_detail = json.load(f)
    logf = open(log_path, 'a')
    logf.write(proto_str)
    logf.write("\n")
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    # total_loop = remain_steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

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
                lr_scheduler.step(net.get_global_step())
                try:
                    example = next(data_iter)
                except StopIteration:
                    print("end epoch")
                    if clear_metrics_every_epoch:
                        net.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)
                example_torch = example_convert_to_torch(example, float_dtype)

                batch_size = example["anchors"].shape[0]

                ret_dict = net(example_torch)

                # box_preds = ret_dict["box_preds"]
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"]
                cls_neg_loss = ret_dict["cls_neg_loss"]
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                dir_loss_reduced = ret_dict["dir_loss_reduced"]
                cared = ret_dict["cared"]
                labels = example_torch["labels"]
                if train_cfg.enable_mixed_precision:
                    loss *= loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                mixed_optimizer.step()
                mixed_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
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
                    metrics["type"] = "step_info"
                    metrics["step"] = global_step
                    metrics["steptime"] = step_time
                    metrics.update(net_metrics)
                    metrics["loss"] = {}
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())
                    metrics["num_vox"] = int(example_torch["voxels"].shape[0])
                    metrics["num_pos"] = int(num_pos)
                    metrics["num_neg"] = int(num_neg)
                    metrics["num_anchors"] = int(num_anchors)
                    # metrics["lr"] = float(
                    #     mixed_optimizer.param_groups[0]['lr'])
                    metrics["lr"] = float(
                        optimizer.lr)
                    if "image_info" in example['metadata'][0]:
                        metrics["image_idx"] = example['metadata'][0]["image_info"]['image_idx']
                    training_detail.append(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, "/")
                    """
                    for k, v in flatted_summarys.items():
                        if isinstance(v, (list, tuple)):
                            v = {str(i): e for i, e in enumerate(v)}
                            writer.add_scalars(k, v, global_step)
                        else:
                            writer.add_scalar(k, v, global_step)
                    """
                    log_str = metric_to_str(metrics)
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
            dt_annos = []
            prog_bar = ProgressBar()
            net.clear_timer()
            prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1) // eval_input_cfg.batch_size)
            for example in iter(eval_dataloader):
                example = example_convert_to_torch(example, float_dtype)
                dt_annos += predict_to_kitti_label(
                    net, example, class_names, center_limit_range,
                    model_cfg.lidar_input)
                prog_bar.print_bar()

            sec_per_ex = len(eval_dataset) / (time.time() - t)
            
            print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
            print(
                f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                file=logf)
            result_official, result_coco = eval_dataset.dataset.evaluation(dt_annos)
            print(result_official)
            print(result_official, file=logf)
            print(result_coco)
            print(result_coco, file=logf)
            if pickle_result:
                with open(result_path_step / "result.pkl", 'wb') as f:
                    pickle.dump(dt_annos, f)
            else:
                kitti_anno_to_label_file(dt_annos, result_path_step)
            writer.add_text('eval_result', result_official, global_step)
            writer.add_text('eval_result coco', result_coco, global_step)
            net.train()
    except Exception as e:
        torchplus.train.save_models(model_dir, [net, optimizer],
                                    net.get_global_step())
        logf.close()
        raise e
    # save model before exit
    torchplus.train.save_models(model_dir, [net, optimizer],
                                net.get_global_step())
    logf.close()


def predict_to_kitti_label(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False):
    predictions_dicts = net(example)
    limit_range = None
    if center_limit_range is not None:
        limit_range = np.array(center_limit_range)
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        box3d_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
        box3d_camera = None
        scores = preds_dict["scores"].detach().cpu().numpy()
        label_preds = preds_dict["label_preds"].detach().cpu().numpy()
        if "box3d_camera" in preds_dict:
            box3d_camera = preds_dict["box3d_camera"].detach().cpu().numpy()
        bbox = None
        if "bbox" in preds_dict:
            bbox = preds_dict["bbox"].detach().cpu().numpy()
        anno = kitti.get_start_result_anno()
        num_example = 0
        for j in range(box3d_lidar.shape[0]):
            if limit_range is not None:
                if (np.any(box3d_lidar[j, :3] < limit_range[:3])
                        or np.any(box3d_lidar[j, :3] > limit_range[3:])):
                    continue
            if "bbox" in preds_dict:
                assert "image_shape" in preds_dict["metadata"]["image"]
                image_shape = preds_dict["metadata"]["image"]["image_shape"]
                if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                    continue
                if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                    continue
                bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                anno["bbox"].append(bbox[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(-np.arctan2(-box3d_lidar[j, 1], box3d_lidar[j, 0]) +
                                    box3d_camera[j, 6])
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])
            else:
                # bbox's height must higher than 25, otherwise filtered during eval
                anno["bbox"].append(np.array([0, 0, 50, 50]))
                # note that if you use raw lidar data to eval,
                # you will get strange performance because
                # in standard KITTI eval, instance with small bbox height
                # will be filtered. but it is impossible to filter
                # boxes when using raw data.
                anno["alpha"].append(0.0)
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
            annos.append(anno)
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["metadata"] = preds_dict["metadata"]
    return annos


def kitti_anno_to_label_file(annos, folder):
    folder = pathlib.Path(folder)
    for anno in annos:
        image_idx = anno["metadata"]["image"]["image_idx"]
        label_lines = []
        for j in range(anno["bbox"].shape[0]):
            label_dict = {
                'name': anno["name"][j],
                'alpha': anno["alpha"][j],
                'bbox': anno["bbox"][j],
                'location': anno["location"][j],
                'dimensions': anno["dimensions"][j],
                'rotation_y': anno["rotation_y"][j],
                'score': anno["score"][j],
            }
            label_line = kitti.kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f"{kitti.get_image_index_str(image_idx)}.txt"
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True,
             measure_time=False,
             batch_size=None):
    result_name = 'eval_results'
    if result_path is None:
        model_dir = pathlib.Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used 
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    net = build_network(model_cfg, measure_time=measure_time).cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    class_names = target_assigner.classes

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,# input_cfg.num_workers,
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
    dt_annos = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            t1 = time.time()
            torch.cuda.synchronize()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        dt_annos += predict_to_kitti_label(
                net, example, class_names, center_limit_range,
                model_cfg.lidar_input)
        # print(json.dumps(net.middle_feature_extractor.middle_conv.sparity_dict))
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms")
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    if pickle_result:
        with open(result_path_step / "result.pkl", 'wb') as f:
            pickle.dump(dt_annos, f)
    else:
        kitti_anno_to_label_file(dt_annos, result_path_step)

    result_official, result_coco = eval_dataset.dataset.evaluation(dt_annos)
    if result_official is not None:
        print(result_official)
        print(result_coco)


def save_config(config_path, save_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    ret = text_format.MessageToString(config, indent=2)
    with open(save_path, 'w') as f:
        f.write(ret)


if __name__ == '__main__':
    fire.Fire()
