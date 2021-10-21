"""Utilities for reading image stack data
"""
import os
import glob
import pathlib
import warnings
import numpy as np
from matplotlib import pyplot as plt

IMAGE_UTILS_TYPE_SELECT = {
    "float": np.float32,
    "ushort": np.uint16,
    "short": np.int16,
    "uchar": np.uint8,
    "char": np.int8,
}


def scale16_tform(x):
    x = x.astype(float) * (1.0 / 16.0)
    return x


IMAGE_UTILS_TFORM_SELECT = {
    "id": None,  # the identity function (the "do nothing" transform)
    "left": None,  # take no action, left label is not a transform
    "right": None,  # take no action, right label is not a transform
    "sqrt": None,  # take no action, sqrt label transform is handled seperately (for now)
    "scale16": scale16_tform,
    # TODO: make all the changes in the ML pipeline to allow turning this on
    # "sqrt": lambda x: np.square(x.astype(float) * (1.0 / 255.0)),
}


def prefix_file_find(folder, prefix):
    """find files by prefix"""
    hits = sorted(glob.glob(folder + "/**/" + prefix + "*", recursive=True))
    unique_parts = []
    for hit in hits:
        unique_parts.append(
            hit[hit.find(folder) + len(folder) : hit.rfind(prefix)].rstrip("/")
        )
    return hits, unique_parts


def find_image_stack(
    directory: pathlib.Path, image_prefix: str, recursive: bool = False
) -> str:
    """find image stack"""
    if recursive:
        match = glob.glob(str(directory / "**" / image_prefix) + "*")
    else:
        match = glob.glob(str(directory / image_prefix) + "*")
    match = [x for x in match if not x.endswith(".gz")]  # ignore zipped files
    if len(match) == 0:
        raise Exception(
            f"Could not find image with prefix {image_prefix} in {directory}"
        )
    elif len(match) > 1:
        raise Exception(
            f"Found multiple images with prefix {image_prefix} in {directory}, "
            "please remove old images"
        )
    return match[0]


class RawReader:
    """Reads images from an image stack file

    Args:
        fname: name of image stack file with a format like:
            <name>.[<tform>].<width>.<height>.<colors>.<type>
            E.g: some_image.512.256.3.uchar, or img.sqrt.512.256.3.uchar

            `tform` can be any from `IMAGE_UTILS_TFORM_SELECT` (or missing)

            `type` can be any from `IMAGE_UTILS_TYPE_SELECT`

            TODO: use null instead of ""

        mask_value: if provided, get_frames will return an array masked with
            x == mask_value

        first_frame: a frame offset to add to subsequent `get_` calls

    Note: if the image stack filename contains six or more tokens separated by
    `.`, and the token in the `<tform>` slot does not match a valid `tform` name in
    `IMAGE_UTILS_TFORM_SELECT`, a warning will be raised. You may consider adding `id`
    to the image stack name to get the same behavior but suppress the warning.
    """

    def __init__(self, fname="", mask_value=None, first_frame=0):
        # determine offsets for underlying raw data
        self.fname = str(fname)
        s = self.fname.split(".")
        self.w = int(s[-4])
        self.h = int(s[-3])
        self.nchan = int(s[-2])
        self.type = IMAGE_UTILS_TYPE_SELECT[s[-1]]
        self.bytes_per_pix = np.dtype(self.type).itemsize
        self.pix_per_frame = self.w * self.h * self.nchan
        total_bytes = os.path.getsize(self.fname)
        self.total_frames = (
            total_bytes // (self.bytes_per_pix * self.pix_per_frame) - first_frame
        )
        self.mask_value = mask_value
        self.first_frame = first_frame

        # get the post-processing transform
        self.tform = None
        if len(s) > 5:
            tform_id = s[-5]
            if tform_id in IMAGE_UTILS_TFORM_SELECT.keys():
                self.tform = IMAGE_UTILS_TFORM_SELECT[tform_id]
            else:
                msg = f"Unrecognized transform ID: {tform_id}. Ignoring it."
                warnings.warn(msg)

    def get_frames(self, start, nframes):
        """read raw data from file"""
        nframes = min(nframes, self.total_frames - start)
        with open(self.fname) as f:
            f.seek(self.pix_per_frame * self.bytes_per_pix * (start + self.first_frame))
            frames = np.fromfile(f, dtype=self.type, count=self.pix_per_frame * nframes)
        # don't use reshape; it should be an error to copy data here
        frames.shape = (nframes, self.h, self.w, self.nchan)

        if self.mask_value is not None:
            # make mask (needs to be done before possible transformation)
            mask = frames == self.mask_value

        if self.tform is not None:
            # apply transformation
            frames = self.tform(frames)

        if self.mask_value is not None:
            # apply mask
            frames = np.ma.masked_where(mask, frames)

        return frames


class TensorReader(RawReader):
    """Reads tensors from a tensor stack file

    Individual tensor channels can be accessed through the RawReader interfaces.

    Args:
        fname: name of tensor stack file with a format like:
            <name>.<channels>.<width>.<height>.1.<type>
            E.g: _some_tensor.32.256.128.1.float
            `type` can be any from `IMAGE_UTILS_TYPE_SELECT`

            Note: the `.1.` for "colors" allows each tensor channel to be accessed
            as frames of the RawReader superclass

        first_frame: a frame offset to add to subsequent `get_` calls

    Attributes:
        tensor_channels (int): the number of channels per tensor. Each channel
            is an individual frame
        total_tensors (int): the total number of tensors available in the file
    """

    def __init__(self, fname="", first_frame=0):
        super().__init__(fname, first_frame=first_frame)
        self.tensor_channels = int(self.fname.split(".")[-5])
        self.total_tensors = self.total_frames // self.tensor_channels

    def get_shape(self, ntensors=None):
        """Returns the shape of the tensor returned by get_tensors.

        (tensors, channels, height, width)

        Args:
            ntensors: number of tensors (defaults to total number in the file)
        """

        shape = (
            ntensors or self.total_tensors,
            self.tensor_channels,
            self.h,
            self.w,
        )
        return shape

    def get_tensors(self, start=0, ntensors=None):
        """Reads tensors from the file

        Args:
            start: tensor offset from first_frame (see __init__)
            ntensors: number of tensors to retrieve

        Returns:
            an array of data read from the file with shape
            `self.get_shape(ntensors)`
        """
        ntensors = ntensors or self.total_tensors
        tensors = super().get_frames(
            start * self.tensor_channels, ntensors * self.tensor_channels
        )
        tensors.shape = self.get_shape(ntensors)
        return tensors


def write_raw_data(file_without_ext, data_array, data_type=None, append=False):
    """
    write the data array into binary format as those from reconstructed batch
    Args:
      file_without_ext:  output file without final extension
      data_array: np array of shape (n, h, w, c) or (n, h, w) to dump
      data_type: one of IMAGE_UTILS_TYPE_SELECT
    Returns:
      None
    """
    # reverse mapping from np types to key str
    if data_type is None:
        data_type2keys = {
            IMAGE_UTILS_TYPE_SELECT[key]: key for key in IMAGE_UTILS_TYPE_SELECT
        }
        array_item_type = type(data_array.flat[0])
        if array_item_type not in data_type2keys:
            raise NotImplementedError(
                f"array type {array_item_type} is not in the list {str(data_type2keys)}"
            )
        data_type = data_type2keys[array_item_type]
    # assume it is with at least three dimensions and at most four dimensions
    assert (
        len(data_array.shape) <= 4 and len(data_array.shape) >= 3
    ), "input data array needs to be (n,h,w,c) or (n,h,w)"
    w = data_array.shape[2]
    h = data_array.shape[1]
    if len(data_array.shape) <= 3:
        nchan = 1
    else:
        nchan = data_array.shape[3]
    if data_type not in IMAGE_UTILS_TYPE_SELECT:
        raise NotImplementedError(
            f"data_type {data_type} is not in the list {str(IMAGE_UTILS_TYPE_SELECT)}"
        )

    output_file_name = f"{file_without_ext}.{w}.{h}.{nchan}.{data_type}"
    with open(output_file_name, "ab" if append else "wb") as f:
        f.write(data_array.copy(order="C"))


# unpack raw12 into raw16
def raw12to16(x):
    """expand 12-bit packed samples to 16 bits"""
    y = np.zeros((len(x) * 2 // 3,), dtype=np.uint16)

    a_high = x[0::3]
    a_low = x[1::3]
    a_low = np.left_shift(a_low, 4)
    y[0::2] = a_high * 256 + a_low

    b_high = x[2::3]
    b_low = x[1::3]
    b_low = np.left_shift(np.right_shift(b_low, 4), 4)
    y[1::2] = b_high * 256 + b_low
    return y


class Raw12Reader:
    """class to read images from a raw stack of bayer12
    frame output dimension is [nframes, h/2, w/2, 4(rgbg)]
    """

    def __init__(self, fname="", w=1280, h=512, pad=0, rggb=None):
        self.fname = fname
        self.w = w
        self.h = h
        self.pad = pad
        if rggb is None:
            self.rggb = [0, 1, 2, 3]
        else:
            self.rggb = rggb
        self.bytes_per_frame = (self.w * self.h * 3) // 2
        total_bytes = os.path.getsize(fname)
        self.total_frames = total_bytes // (self.bytes_per_frame + self.pad)

    def get_frames(self, start, nframes):
        """get frames"""
        with open(self.fname) as f:
            f.seek((self.bytes_per_frame + self.pad) * start)
            x = np.fromfile(
                f, dtype=np.uint8, count=(self.bytes_per_frame + self.pad) * nframes
            )

        if self.pad > 0:
            x = np.reshape(x, (nframes, (self.bytes_per_frame + self.pad)))
            x = x[:, self.pad : :]
            x = np.reshape(x, (np.size(x),))

        x = raw12to16(x)
        x = np.reshape(x, (nframes, self.h, self.w))
        im = np.zeros((nframes, 4, self.h // 2, self.w // 2), dtype=np.float32)

        r_ind = np.unravel_index(self.rggb[0], (2, 2))
        im[:, 0, :, :] = x[:, r_ind[0] :: 2, r_ind[1] :: 2]

        g_ind = np.unravel_index(self.rggb[1], (2, 2))
        im[:, 1, :, :] = x[:, g_ind[0] :: 2, g_ind[1] :: 2]

        b_ind = np.unravel_index(self.rggb[3], (2, 2))
        im[:, 2, :, :] = x[:, b_ind[0] :: 2, b_ind[1] :: 2]

        g_ind = np.unravel_index(self.rggb[2], (2, 2))
        im[:, 3, :, :] = x[:, g_ind[0] :: 2, g_ind[1] :: 2]

        return np.transpose(im, (0, 2, 3, 1)) * (1 / 2 ** 16)


class ImageStackPlot(object):
    """display for image stack

    x must be size [nimages, h, w, ncolors]
    or x can be a Reader class with method "get_frames(start, nframes)"
    by passing a list of ImageStackPlot into friends you can have multiple
    synchronized image views

    - mousewheel to step through image stack (+shift key to step faster)
    - ctrl + mousewheel to zoom
    - click and drag mouse to pan

    TODO: use None instead of ""
    """

    def __init__(
        self, x, title="", wintitle="", cmap="", vmin=None, vmax=None, friends=None
    ):
        # image stack
        self.x = x
        self.reader = "numpy" not in type(x).__module__
        if self.reader:
            self.nframes = self.x.total_frames
            title = title or self.x.fname
            colors = self.x.nchan
        else:
            self.nframes = x.shape[0]
            colors = x.shape[3]

        wintitle = wintitle or title

        # ui variables
        self.ctrl = False  # control key is pressed
        self.shift = False  # shift key is pressed
        self.mouse_position = None
        self.zoom = 1.0
        self.ind = 0

        [self.fig, self.ax] = plt.subplots(1, 1, figsize=(15, 6))
        self.ax.set_title(title, fontsize=12)
        self.fig.canvas.set_window_title(wintitle)

        first = self.get_frame()
        cmap = cmap if cmap != "" else None
        self.vmin = vmin
        self.vmax = vmax
        self.im = plt.imshow(first, vmin=vmin, vmax=vmax, cmap=cmap)

        if colors == 1:
            plt.colorbar()

        self.update_frame()

        xlim = self.ax.get_xlim()
        self.width = abs(xlim[1] - xlim[0])
        ylim = self.ax.get_ylim()
        self.height = abs(ylim[1] - ylim[0])

        # ui callbacks
        self.fig.canvas.mpl_connect("scroll_event", self.scroll_event)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.key_release)
        self.fig.canvas.mpl_connect("button_press_event", self.mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.mouse_move)
        self.fig.canvas.mpl_connect("close_event", self.leave_friends_behind)
        self.friends = []

        # synchronized ImageStackPlots (same zoom and frame #)
        if friends is None:
            friends = []
        friends.append(self)
        friends = list(set(friends))  # ensure each friend is only listed once
        for plot in friends:
            plot.friends = friends

    def scroll_event(self, event):
        """scroll event"""
        if self.ctrl and event.xdata is not None:
            zoom = self.zoom * (0.5 + 1.5 * (event.button == "up"))
            zoom = min(max(zoom, 1), 64)
            self.update_zoom(
                zoom, event.xdata / self.width, event.ydata / self.height, True
            )
        else:
            step = -1 + 2 * (event.button == "up")
            ind = self.ind + step * (1 + 15 * self.shift)
            self.update_frame(ind, True)

    def key_press(self, event):
        """key press"""
        if event.key == "control":
            for plot in self.friends:
                plot.ctrl = True
        if event.key == "shift":
            for plot in self.friends:
                plot.shift = True

    def key_release(self, event):
        """key release"""
        if event.key == "control":
            for plot in self.friends:
                plot.ctrl = False
        if event.key == "shift":
            for plot in self.friends:
                plot.shift = False

    def mouse_press(self, event):
        """mouse press"""
        if event.inaxes == self.ax:
            self.mouse_position = [event.xdata, event.ydata]

    def mouse_release(self, event):
        """mouse release"""
        self.mouse_position = None

    def mouse_move(self, event):
        """mouse move"""
        if self.mouse_position is not None and event.xdata is not None:
            dx = (self.mouse_position[0] - event.xdata) / self.width
            dy = (self.mouse_position[1] - event.ydata) / self.height
            self.update_pan(dx, dy, True)

    def leave_friends_behind(self, event):
        """remove the callback references from other figs if user closes the plot"""
        self.friends.remove(self)
        for plot in self.friends:
            plot.friends = self.friends

    def update_pan(self, dx, dy, master=False):
        """update pan"""
        if master:
            for plot in self.friends:
                plot.update_pan(dx, dy)
        else:
            xlim = self.ax.get_xlim()
            self.ax.set_xlim([xlim[0] + dx * self.width, xlim[1] + dx * self.width])
            ylim = self.ax.get_ylim()
            self.ax.set_ylim([ylim[0] + dy * self.height, ylim[1] + dy * self.height])
            self.im.axes.figure.canvas.draw()

    def update_zoom(self, zoom, x, y, master=False):
        """update zoom"""
        if master:
            for plot in self.friends:
                plot.update_zoom(zoom, x, y)
        else:
            self.zoom = zoom
            if self.zoom == 1:
                x = 0.5
                y = 0.5
            w2 = 0.5 * self.width / self.zoom
            h2 = 0.5 * self.height / self.zoom
            self.ax.set_xlim([x * self.width - w2, x * self.width + w2])
            self.ax.set_ylim([y * self.height + h2, y * self.height - h2])
            self.im.axes.figure.canvas.draw()

    def update_frame(self, ind=0, master=False):
        """update frame"""
        ind = min(max(ind, 0), self.nframes - 1)
        if master:
            for plot in self.friends:
                plot.update_frame(ind)
        else:
            if ind != self.ind:
                self.ind = ind
                data = self.get_frame()
                self.im.set_data(data)
                if self.vmin is None or self.vmax is None:
                    vmin = self.vmin or np.min(data)
                    vmax = self.vmax or np.max(data)
                    self.im.set_clim(vmin=vmin, vmax=vmax)
                self.ax.set_xlabel("Frame " + str(self.ind))
                self.im.axes.figure.canvas.draw()

    def get_frame(self):
        """get frame"""
        if self.reader:
            x = self.x.get_frames(start=self.ind, nframes=1)
        else:
            x = self.x[self.ind, :, :, :]
            x = np.expand_dims(x, axis=0)
        if x.shape[3] > 3:
            x = x[:, :, :, 0:3]
        return np.squeeze(x)


# example image display
if __name__ == "__main__":
    R1 = Raw12Reader("/mnt/nvme/datasets/csi/csi-200102-214954/Left.Bayer12")
    R2 = Raw12Reader("/mnt/nvme/datasets/csi/csi-200102-214954/Right.Bayer12")
    # R2 = Raw12Reader(
    #   fname = "/mnt/nvme/datasets/censors/Censors-200102-154806/left.tbayer12",
    #   w = 2048, h = 800, pad = 32)

    p1 = ImageStackPlot(R1)
    p2 = ImageStackPlot(R2, friends=[p1])

    R = RawReader("/mnt/nvme/datasets/example/rgb.left.640.256.3.float")
    p3 = ImageStackPlot(R)

    plt.show()
