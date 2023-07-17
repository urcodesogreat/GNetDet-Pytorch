#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from itertools import zip_longest

from intersect import Point, Line
from pyGNetDet.GNetDet import ChipInfer
from pyGNetDet.config import get_config


WIN_NAME = "Count-Person"
source: np.ndarray = None
dummy: np.ndarray = None


class PeopleCount(ChipInfer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.crowded_thresh = cfg.OPTS.CROWDED_THRESH
        self.polygon = []
        cv2.namedWindow(WIN_NAME, 0)
        cv2.setMouseCallback(WIN_NAME, self._mouse_handler)

    def create_polygon(self, frame):
        global source, dummy
        dummy = frame.copy()
        while True:
            source = dummy.copy()
            self._rend_polygon(source, False)
            cv2.imshow(WIN_NAME, source)
            k = cv2.waitKey(20)
            if k == ord('a') and len(self.polygon) > 2:
                break
            elif k == ord('d'):
                self.polygon.clear()

    def _rend_polygon(self, image, crowded):
        n = len(self.polygon)
        for i in range(n):
            p1 = self.polygon[i]
            p2 = self.polygon[(i+1) % n]
            p1.rend(image)
            side = Line(p1, p2, )
            side.rend(image, (0, 0, 255) if crowded else (0, 255, 0))
        return image

    def _mouse_handler(self, action, x, y, flags, userdata):
        if action == cv2.EVENT_LBUTTONDOWN:
            self.polygon.append(Point(x, y))

    def process_frame(self, frame: np.ndarray):
        h, w, *_ = frame.shape
        frame_infer = self.preprocess(frame)
        pred = self.chip_model.evaluate(frame_infer)[:, :, :, :self.chip_out_size]
        pred = pred.transpose((0, 2, 3, 1))[:, :, :, :self.model_out_size]
        pred *= (self.cap / 31.)
        boxes, cls_indexs, probs = self.decoder(pred)

        count_total = 0
        count_inside = 0
        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            cx = round((x1 + x2) / 2)
            cy = round((y1 + y2) / 2)

            cls_index = int(cls_indexs[i])
            prob = float(probs[i])
            class_name = self.class_names[cls_index]
            color = self.color[cls_index + 1]

            if Point(cx, cy).inside_polygon(self.polygon):
                color = (225, 225, 0)
                count_inside += 1
            count_total += 1

            if self.fancy:
                self.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame, '%s' % class_name + ' %.2f' % prob,
                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5,
                color, 1, 8)

        return frame, count_total, count_inside

    def _on_stream(self, file):
        cap = cv2.VideoCapture(file)

        status, frame = cap.read()
        if not status:
            sys.stderr.write(f"error: unable to open stream: {file}\n")
            cap.release()
            cv2.destroyAllWindows()
            return
        try:
            self.create_polygon(frame)
            cv2.setMouseCallback(WIN_NAME, lambda *args: None)
            while True:
                status, frame = cap.read()
                if not status:
                    break
                pre_frame, count_total, count_inside = self.process_frame(frame)
                is_crowd = (count_inside >= self.crowded_thresh)
                color = (0, 0, 255) if is_crowd else (255, 0, 0)
                cv2.putText(pre_frame, f"People:{count_total}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 1)
                cv2.putText(pre_frame, f"Queue:{count_inside}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 1)
                cv2.putText(pre_frame, f"Crowed: {'YES' if is_crowd else 'NO'}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 1)
                self._rend_polygon(pre_frame, is_crowd)
                cv2.imshow(WIN_NAME, pre_frame)
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
        self.create_polygon(img)
        pre_frame, count_total, count_inside = self.process_frame(img)
        is_crowd = (count_inside >= self.crowded_thresh)
        color = (0, 0, 255) if is_crowd else (255, 0, 0)
        cv2.putText(pre_frame, f"People:{count_total}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 1)
        cv2.putText(pre_frame, f"Queue:{count_inside}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 1)
        cv2.putText(pre_frame, f"Crowed: {'YES' if is_crowd else 'NO'}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 1)
        result = self._rend_polygon(pre_frame, is_crowd)
        cv2.imshow(WIN_NAME, result)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return result


def main(cfg):
    inference = PeopleCount(cfg)

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
    cfg = get_config()
    print(cfg)
    main(cfg)
