#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnetmdk.gti.chip.driver import GtiModel
from configs import get_config
from gnetmdk.utils.experiment import silent

with silent():
    cfg = get_config()

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128], [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def decoder(pred):
    grid_num = cfg.grid_size
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = np.squeeze(pred)
    contain1 = pred[:, :, 4][:, :, np.newaxis]
    contain2 = pred[:, :, 9][:, :, np.newaxis]
    contain = np.concatenate((contain1, contain2), axis=-1)
    mask1 = contain > cfg.conf_thresh
    mask2 = (contain == contain.max())
    mask = ((mask1 + mask2) > 0).astype(np.int32)
    for i in range(grid_num):
        for j in range(grid_num):
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
                    if float(contain_prob * max_prob) > cfg.prob_thresh:
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
    keep = nms(boxes, probs, cfg.nms_thresh)
    return boxes[keep], cls_indexs[keep], probs[keep]


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


class Function_:
    def __init__(self, model):
        self.image_size = cfg.image_size
        self.chip_layer = GtiModel(model)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        frame_inp = cv2.resize(frame, (self.image_size, self.image_size),
                               interpolation=cv2.INTER_LINEAR)   # cv2.INTER_CUBIC
        chip_start_time = time.time()
        pred = self.chip_layer.evaluate(frame_inp)[:, :, :, :int(cfg.chip_depth)]
        pred = pred.transpose((0, 2, 3, 1))[:, :, :, :cfg.grid_depth]
        chip_end_time = time.time()

        pred *= (cfg.scale / 31.)
        boxes, cls_indexs, probs = decoder(pred)

        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)

            cls_index = cls_indexs[i]
            cls_index = int(cls_index)
            prob = probs[i]
            prob = float(prob)

            class_name = cfg.VOC_CLASSES[cls_index]
            # color = Color[VOC_CLASSES.index(class_name)+1]
            color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # self.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                frame, '%s' % class_name +
                       '%.2f' % prob,
                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (255, 255, 255), 1, 8)

        return frame, chip_end_time - chip_start_time

    @staticmethod
    def rectangle(img, pt1, pt2, color, thickness):
        x1, y1 = pt1
        x2, y2 = pt2
        r = 2
        dx, dy = int(max(4, abs(x2 - x1) * 0.02)), int(max(2, abs(y2 - y1) * 0.04))

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

    def evaluate_video(self, video_file):
        cv2.namedWindow('GnetDet', 0)
        cap = cv2.VideoCapture(video_file)
        count = 0
        try:
            while True:
                start = time.time()
                flag, frame = cap.read()
                if not flag or frame is None:
                    break
                print("process frame:", count)
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
        # print(1/(sum_time/count))


if __name__ == '__main__':
    if cfg.Display == 'camera':
        Function_(cfg.model_path).evaluate_video(0)
    elif cfg.Display == 'mp4':
        Function_(cfg.model_path).evaluate_video(cfg.mp4_path)
