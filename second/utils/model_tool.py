from pathlib import Path 
import re 
import shutil

def rm_invalid_model_dir(directory, step_thresh=200):
    directory = Path(directory).resolve()
    ckpt_re = r"[a-zA-Z0-9_]+\-([0-9]+)\.tckpt"
    ckpt_re = re.compile(ckpt_re)
    for path in directory.rglob("*"):
        if path.is_dir():
            pipeline_path = (path / "pipeline.config")
            log_path = (path / "log.txt")
            summary_path = (path / "summary")
            must_exists = [pipeline_path, log_path, summary_path]
            if not all([e.exists() for e in must_exists]):
                continue
            ckpts = []
            for subpath in path.iterdir():
                match = ckpt_re.search(subpath.name)
                if match is not None:
                    ckpts.append(int(match.group(1)))
            if len(ckpts) == 0 or all([e < step_thresh for e in ckpts]):
                shutil.rmtree(str(path))
