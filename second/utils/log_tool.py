import numpy as np 
from tensorboardX import SummaryWriter
import json 
from pathlib import Path 

def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + str(k))
        else:
            flatted[start + sep + str(k)] = v

def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, str(k))
        else:
            flatted[str(k)] = v
    return flatted

def metric_to_str(metrics, sep='.'):
    flatted_metrics = flat_nested_json_dict(metrics, sep)
    metrics_str_list = []
    for k, v in flatted_metrics.items():
        if isinstance(v, float):
            metrics_str_list.append(f"{k}={v:.4}")
        elif isinstance(v, (list, tuple)):
            if v and isinstance(v[0], float):
                v_str = ', '.join([f"{e:.4}" for e in v])
                metrics_str_list.append(f"{k}=[{v_str}]")
            else:
                metrics_str_list.append(f"{k}={v}")
        else:
            metrics_str_list.append(f"{k}={v}")
    return ', '.join(metrics_str_list)

class SimpleModelLog:
    """For simple log.
    generate 4 kinds of log: 
    1. simple log.txt, all metric dicts are flattened to produce
    readable results.
    2. TensorBoard scalars and texts
    3. multi-line json file log.json.lst
    4. tensorboard_scalars.json, all scalars are stored in this file
        in tensorboard json format.
    """
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.log_file = None 
        self.log_mjson_file = None
        self.summary_writter = None
        self.metrics = []
        self._text_current_gstep = -1
        self._tb_texts = []

    def open(self):
        model_dir = self.model_dir
        assert model_dir.exists()
        summary_dir = model_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)

        log_mjson_file_path = model_dir / f'log.json.lst'
        if log_mjson_file_path.exists():
            with open(log_mjson_file_path, 'r') as f:
                for line in f.readlines():
                    self.metrics.append(json.loads(line))
        log_file_path = model_dir / f'log.txt'
        self.log_mjson_file = open(log_mjson_file_path, 'a')
        self.log_file = open(log_file_path, 'a')
        self.summary_writter = SummaryWriter(str(summary_dir))
        return self

    def close(self):
        assert self.summary_writter is not None
        self.log_mjson_file.close()
        self.log_file.close()
        tb_json_path = str(self.model_dir / "tensorboard_scalars.json")
        self.summary_writter.export_scalars_to_json(tb_json_path)
        self.summary_writter.close()
        self.log_mjson_file = None 
        self.log_file = None 
        self.summary_writter = None

    def log_text(self, text, step, tag="regular log"):
        """This function only add text to log.txt and tensorboard texts
        """
        print(text)
        print(text, file=self.log_file)
        if step > self._text_current_gstep and self._text_current_gstep != -1:
            total_text = '\n'.join(self._tb_texts)
            self.summary_writter.add_text(tag, total_text, global_step=step)
            self._tb_texts = []
            self._text_current_gstep = step
        else:
            self._tb_texts.append(text)
        if self._text_current_gstep == -1:
            self._text_current_gstep = step


    def log_metrics(self, metrics: dict, step):
        flatted_summarys = flat_nested_json_dict(metrics, "/")
        for k, v in flatted_summarys.items():
            if isinstance(v, (list, tuple)):
                if any([isinstance(e, str) for e in v]):
                    continue
                v_dict = {str(i): e for i, e in enumerate(v)}
                for k1, v1 in v_dict.items():
                    self.summary_writter.add_scalar(k + "/" + k1, v1, step)
            else:
                if isinstance(v, str):
                    continue
                self.summary_writter.add_scalar(k, v, step)
        log_str = metric_to_str(metrics)
        print(log_str)
        print(log_str, file=self.log_file)
        print(json.dumps(metrics), file=self.log_mjson_file)

