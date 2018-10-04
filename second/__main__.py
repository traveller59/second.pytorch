import os
import sys
from importlib import import_module

import fire
from google.protobuf import json_format, text_format

from codeai.tools.file_ops import scan

VOXELNET_CONFIG_PROTOS = "./protos"


def update_config(path, field, new_value):
    pass


def clean_config(path):
    pass


if __name__ == "__main__":
    method_name = sys.argv[1]
    module_name = ".".join(method_name.split(".")[:-1])
    obj = import_module(module_name, "second")
    fire.Fire(getattr(obj, (method_name.split(".")[-1])), command=sys.argv[2:])
