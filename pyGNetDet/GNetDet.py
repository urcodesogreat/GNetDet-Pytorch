#!/usr/bin/env python3
import platform
if platform.system() != "Linux":
    raise NotImplementedError("SDK on Linux ONLY!")
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
import numpy as np
from itertools import zip_longest

from ctypes import (byref, cast, CDLL, POINTER, Structure, c_uint8, c_float, c_ulonglong, c_char_p, c_int, c_void_p)
from config import get_config


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
    def __init__(self, cfg):
        self.libgtisdk = CDLL(cfg.LIBGTISDK)
        self.libgtisdk.GtiCreateModel.argtypes = [c_char_p]
        self.libgtisdk.GtiCreateModel.restype = c_ulonglong
        self.obj = self.libgtisdk.GtiCreateModel(cfg.MODEL.PATH.encode('ascii'))
        if self.obj == 0:
            print("[ERROR] Fatal error creating GtiModel.  Does python have permission to access the chip?")
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
            0,  # stride = 0, irrelevant for this use case
            numpy_array.ctypes.data,  # input buffer
            None,  # customerBuffer
            in_channels * in_height * in_width,  # input buffer size
            0,  # tensor format = 0, binary format,
            None,  # tag
            None  # next
        )

        self.libgtisdk.GtiEvaluate.argtypes = [c_ulonglong, POINTER(GtiTensor)]
        self.libgtisdk.GtiEvaluate.restype = POINTER(GtiTensor)
        out_tensor = self.libgtisdk.GtiEvaluate(self.obj, byref(in_tensor))

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
            self.libgtisdk.GtiDestroyModel.argtypes = [c_ulonglong]
            self.libgtisdk.GtiDestroyModel.restype = c_int
            destroyed = self.libgtisdk.GtiDestroyModel(self.obj)
            if not destroyed:
                raise Exception("Unable to release sources for GTI driver net.")
            self.obj = None


class ChipInfer(object):

    def __init__(self, cfg):
        print(f"Initiate `{cfg.MODEL.NAME}` on chip `{cfg.CHIP}`")
        self.chip_model = GtiModel(cfg)
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.class_names = cfg.DATA.CLASS_NAMES
        self.input_format = cfg.MODEL.INPUT.FORMAT
        self.input_size = cfg.MODEL.INPUT.SIZE
        self.grid_size = 14
        self.model_out_size = 10 + self.num_classes
        self.chip_out_size = self.get_chip_out_size()

        self.cap = cfg.MODEL.OUTPUT.CAP
        self.conf_thresh = cfg.OPTS.CONF_THRESH
        self.prob_thresh = cfg.OPTS.PROB_THRESH
        self.nms_thresh = cfg.OPTS.NMS_THRESH
        self.color = cfg.OPTS.COLOR
        self.fancy = cfg.FANCY

    def get_chip_out_size(self):
        chip_out = 0
        for n in range(20):
            thresh = self.model_out_size / np.power(2, n)
            if thresh < 1:
                chip_out = np.power(2, n)
                break
        return chip_out

    def preprocess(self, img: np.ndarray):
        if self.input_format == 1:  # Y
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        return img

    def process_frame(self, frame: np.ndarray):
        h, w, *_ = frame.shape
        frame_infer = self.preprocess(frame)
        chip_start_time = time.perf_counter()
        pred = self.chip_model.evaluate(frame_infer)[:, :, :, :self.chip_out_size]
        pred = pred.transpose((0, 2, 3, 1))[:, :, :, :self.model_out_size]
        chip_end_time = time.perf_counter()

        pred *= (self.cap / 31.)
        boxes, cls_indexs, probs = self.decoder(pred)

        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)

            cls_index = cls_indexs[i]
            cls_index = int(cls_index)
            prob = probs[i]
            prob = float(prob)

            class_name = self.class_names[cls_index]
            color = self.color[cls_index + 1]

            if self.fancy:
                self.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame, '%s' % class_name + ' %.2f' % prob,
                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5,
                color, 1, 8)

        return frame, chip_end_time - chip_start_time

    def decoder(self, pred: np.ndarray):
        boxes = []
        cls_indexs = []
        probs = []
        cell_size = 1. / self.grid_size

        pred = np.squeeze(pred)
        contain1 = pred[:, :, 4][:, :, np.newaxis]
        contain2 = pred[:, :, 9][:, :, np.newaxis]
        contain = np.concatenate((contain1, contain2), axis=-1)
        mask1 = contain > self.conf_thresh
        mask2 = (contain == contain.max())
        mask = ((mask1 + mask2) > 0).astype(np.int32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for b in range(2):
                    if mask[i, j, b] == 1:
                        box = pred[i, j, b * 5:b * 5 + 4]
                        contain_prob = pred[i, j, b * 5 + 4]
                        xy = np.array([j, i]) * cell_size
                        box[:2] = box[:2] * cell_size + xy
                        box_xy = np.zeros_like(box)
                        box_xy[:2] = box[:2] - 0.5 * box[2:]
                        box_xy[2:] = box[:2] + 0.5 * box[2:]
                        max_prob = np.max(pred[i, j, 10:])
                        cls_index = np.argmax(pred[i, j, 10:])
                        if float(contain_prob * max_prob) > self.prob_thresh:
                            boxes.append(box_xy)
                            cls_indexs.append(cls_index)
                            probs.append(contain_prob * max_prob)
        if len(boxes) == 0:
            boxes = np.zeros((1, 4))
            probs = np.zeros(1)
            cls_indexs = np.zeros(1)
        else:
            boxes = np.array(boxes)
            probs = np.array(probs)
            cls_indexs = np.array(cls_indexs)
        keep = self.nms(boxes, probs, threshold=self.nms_thresh)
        return boxes[keep], cls_indexs[keep], probs[keep]

    @staticmethod
    def nms(bboxes, scores, threshold=0.5):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            xx1 = x1[order[1:]].clip(min=x1[i])
            yy1 = y1[order[1:]].clip(min=y1[i])
            xx2 = x2[order[1:]].clip(max=x2[i])
            yy2 = y2[order[1:]].clip(max=y2[i])

            w = (xx2 - xx1).clip(min=0)
            h = (yy2 - yy1).clip(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= threshold).nonzero()[0]
            if len(ids) == 0:
                break
            order = order[ids + 1]
        return np.array(keep)

    @staticmethod
    def rectangle(img, pt1, pt2, color, thickness):
        x1, y1 = pt1
        x2, y2 = pt2
        r = 2
        dx, dy = int(max(4, abs(x2 - x1) * 0.04)), int(max(2, abs(y2 - y1) * 0.08))

        # Top leftq
        cv2.line(img, (x1 + r, y1), (x1 + r + dx, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + dy), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - dx, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + dy), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + dx, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - dy), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - dx, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - dy), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def on_camara(self):
        return self._on_stream(0)

    def on_video(self, file):
        return self._on_stream(file)

    def _on_stream(self, file):
        cv2.namedWindow('GnetDet', 0)
        cap = cv2.VideoCapture(file)
        count = 0

        try:
            while True:
                start = time.time()
                flag, frame = cap.read()
                if not flag or frame is None:
                    break
                pre_frame, chip_time = self.process_frame(frame)
                end = time.time()
                timestamp = end - start

                cv2.putText(pre_frame, "chip: {0:.3f} fps".format(1 / chip_time), (10, 20), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 1)
                cv2.putText(pre_frame, "display: {0:.3f} fps".format(1 / timestamp), (10, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 1)
                count += 1

                cv2.imshow('GnetDet', pre_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    while True:
                        key = cv2.waitKey(0)
                        if key == ord(' '):
                            break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def on_image(self, file):
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        pre_frame, chip_time = self.process_frame(img)
        result = cv2.putText(pre_frame, "chip: {0:.3f} fps".format(1 / chip_time), (10, 20),
                             cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
        cv2.imshow('GnetDet', pre_frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return result


def main():
    cfg = get_config()
    print(cfg)
    inference = ChipInfer(cfg)

    if cfg.TYPE == "camara":
        inference.on_camara()
    elif cfg.TYPE == "video":
        if cfg.OUTPUT_PATH:
            print("[INFO] Save result as video is currently not supported.")
        for path in cfg.INPUT_PATH.split(','):
            inference.on_video(path)
    elif cfg.TYPE == "image":
        for inp_path, out_path in zip_longest(cfg.INPUT_PATH.split(','), cfg.OUTPUT_PATH.split(',')):
            img = inference.on_image(inp_path)
            if out_path:
                cv2.imwrite(out_path, img)
                print(f"[INFO] Detection result has already saved: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
