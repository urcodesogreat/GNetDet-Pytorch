'''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
from ctypes import (
    byref, cast, CDLL, POINTER, Structure,
    c_uint8, c_float, c_ulonglong, c_char_p, c_int, c_void_p
)
import os
import platform
import numpy as np

if platform.system() != "Linux":
    raise NotImplementedError("Windows support is currently limited")
libgtisdk = CDLL(os.path.join(os.path.dirname(__file__), "libGTILibrary.so"))
libgtisdk4 = CDLL(os.path.join(os.path.dirname(__file__), "libGTILibrary.so.4.5.1"))


class GtiTensor(Structure):
    pass


GtiTensor._fields_ = [
    ("width", c_int),
    ("height", c_int),
    ("depth", c_int),
    ("stride", c_int),
    ("buffer", c_void_p),
    ("customerBuffer", c_void_p),
    ("size", c_int),  # buffer size
    ("format", c_int),  # tensor format
    ("tag", c_void_p),
    ("next", POINTER(GtiTensor))
]


class GtiModel(object):
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("{} layers file does not exist".format(model_path))
        libgtisdk.GtiCreateModel.argtypes = [c_char_p]
        libgtisdk.GtiCreateModel.restype = c_ulonglong
        self.obj = libgtisdk.GtiCreateModel(model_path.encode('ascii'))
        if self.obj == 0:
            print("Fatal error creating GtiModel.  Does python have"
                  "permission to access the chip?")
            exit(-1)

    def evaluate(self, numpy_array, activation_bits=5):
        """Evaluate tensor on GTI device for chip layers only.

        Args:
            numpy_array: 3D or 4D array in [(batch,) height, width, channel]
                order. If present, batch must be 1.
        Returns:
            4D numpy float32 array in [batch, height, width, channel] order
        """
        if len(numpy_array.shape) == 4:  # squeeze batch dimension
            numpy_array = numpy_array.squeeze(axis=0)
        if len(numpy_array.shape) != 3:
            raise ValueError("Input dimension must be HWC or NHWC")

        # transform chip input tensor
        # 1. split tensor by depth/channels, e.g. BGR channels = 3
        # 2. vertically stack channels
        in_height, in_width, in_channels = numpy_array.shape
        numpy_array = np.vstack(np.dsplit(numpy_array, in_channels))
        in_tensor = GtiTensor(
            in_width,
            in_height,
            in_channels,
            0,      # stride = 0, irrelevant for this use case
            numpy_array.ctypes.data,  # input buffer
            None,   # customerBuffer
            in_channels * in_height * in_width,  # input buffer size
            0,      # tensor format = 0, binary format,
            None,   # tag
            None    # next
        )

        libgtisdk.GtiEvaluate.argtypes = [c_ulonglong, POINTER(GtiTensor)]
        libgtisdk.GtiEvaluate.restype = POINTER(GtiTensor)
        out_tensor = libgtisdk.GtiEvaluate(self.obj, byref(in_tensor))

        # transform chip output tensor
        out_width = out_tensor.contents.width
        out_height = out_tensor.contents.height
        out_channels = out_tensor.contents.depth
        # output tensor is [channel, height, width] order
        out_shape = (1, out_channels, out_height, out_width)  # add 1 as batch dimension
        pointer_type = POINTER(c_float) if activation_bits > 5 else POINTER(c_uint8)
        # for this use case, output tensor is floating point
        out_buffer = cast(out_tensor.contents.buffer, pointer_type)
        result = (
            np.ctypeslib.as_array(out_buffer, shape=(np.prod(out_shape),))
                .reshape(out_shape)  # reshape buffer to 4D tensor
        )
        return result.astype(np.float32)

    def release(self):
        if self.obj is not None:
            libgtisdk.GtiDestroyModel.argtypes = [c_ulonglong]
            libgtisdk.GtiDestroyModel.restype = c_int
            destroyed = libgtisdk.GtiDestroyModel(self.obj)
            if not destroyed:
                raise Exception("Unable to release sources for GTI driver layers")
            self.obj = None


def compose_model(json_file, model_file):
    libgtisdk4.GtiComposeModelFile.argtypes = [c_char_p, c_char_p]
    return libgtisdk4.GtiComposeModelFile(json_file.encode('ascii'), model_file.encode('ascii'))
