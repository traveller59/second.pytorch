import fire 

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

def eval_main(root_path, version, eval_version, res_path, eval_set, output_dir):
    nusc = NuScenes(
        version=version, dataroot=str(root_path), verbose=False)

    cfg = config_factory(eval_version)
    nusc_eval = NuScenesEval(nusc, config=cfg, result_path=res_path, eval_set=eval_set, 
                            output_dir=output_dir,
                            verbose=False)
    nusc_eval.main(render_curves=False)

if __name__ == "__main__":
    fire.Fire(eval_main)