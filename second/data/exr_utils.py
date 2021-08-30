"""Utility functions for working with OpenEXR images."""

import pathlib
from typing import Sequence

import Imath
import numpy as np
import OpenEXR


DTYPE_MAP = {
    np.float16: Imath.PixelType(Imath.PixelType.HALF),
    np.float32: Imath.PixelType(Imath.PixelType.FLOAT),
    np.uint32: Imath.PixelType(Imath.PixelType.UINT),
}

MAP_DTYPE = {str(v):k for k, v in DTYPE_MAP.items()}

def save(
    path: pathlib.Path,
    img: np.ndarray,
    dtype: np.dtype = None,
    channel_names: Sequence = None,
    compression: Imath.Compression = Imath.Compression.NO_COMPRESSION,
):
    """Save image as an OpenEXR file"""
    if dtype is None:
        dtype = img.dtype.type
    # input image is of shape [height, width, nch]
    header = OpenEXR.Header(img.shape[1], img.shape[0])
    if len(img.shape) == 2:
        channel_names = ["Y"]
        img.shape = [img.shape[0], img.shape[1], 1]
    elif channel_names is None:
        channel_names = [str(k) for k in range(img.shape[2])]
    header["channels"] = {
        name: Imath.Channel(DTYPE_MAP[dtype]) for name in channel_names
    }
    header["compression"] = Imath.Compression(compression)
    data = {
        name: (img[:, :, k].astype(dtype)).tostring()
        for k, name in enumerate(channel_names)
    }

    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels(data)
    out.close()
    if len(img.shape) == 3 and img.shape[2] == 1:
        img.shape = img.shape[:2]

def load(path: pathlib.Path, dtype: np.dtype = np.float32):
    """Loads the specified OpenEXR image and returns it in numpy format.

    Note: Legacy images were saved as RGB with data only in R channel, so this function
    does not currently support RGB images.
    """
    infile = OpenEXR.InputFile(str(path))
    header = infile.header()
    # load width and height
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    # channel precisions
    channel_precision = {c: v.type for c, v in header["channels"].items()}
    # load channels
    channels = header["channels"]
    if "R" in channels:
        channel_names = ["R"]
        num_chs = 1
    else:
        channel_names = list(channels.keys())
        num_chs = len(channels)

    # assume all channels are of the same dtype
    dtype = MAP_DTYPE[str(channel_precision[channel_names[0]])]

    if num_chs == 1:
        img_bytes = infile.channel(channel_names[0], DTYPE_MAP[dtype])
        img = np.frombuffer(img_bytes, dtype=dtype)
        img.shape = (height, width)
    else:
        img = np.zeros((height, width, num_chs), dtype=dtype)
        strings = infile.channels(channel_names)
        for i, string in enumerate(strings):
            img[:, :, i] = np.frombuffer(string, dtype=dtype).reshape(height, width)

    infile.close()
    return img
