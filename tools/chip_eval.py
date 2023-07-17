#!/usr/bin/env python3
from tools.predict import *
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnetmdk.gti.chip.driver import GtiModel
from configs import get_config


cfg = get_config()


Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
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


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap calculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(preds, target, recs, VOC_CLASSES, threshold=0.5, use_07_metric=False, ):
    aps = []
    f = open(cfg.valid_txt_path)
    lines = f.readlines()
    imagenames = []
    for line in lines:
        splited = line.strip().split()
        imagenames.append(splited)
    f.close()
    for i, class_ in enumerate(VOC_CLASSES):
        class_recs = {}
        npos = 0
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # if get nothing
            ap = -1
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
            print("--------------------------------")
            break

        for t in imagenames:
            imagename = t[0]
            if recs[imagename] is not None:
                R = [obj for obj in recs[imagename] if obj['name'] == class_]
                bbox = np.array([x['bbox'] for x in R])
                difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
                det = [False] * len(R)
                npos = npos + sum(~difficult)
                class_recs[imagename] = {'bbox': bbox,
                                         'difficult': difficult,
                                         'det': det}

        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > threshold:
                if not R['det'][jmax]:
                    tp[d] = 1
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_, ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))


def prepare_target(target):
    image_list = []  # image path list
    f = open(cfg.valid_txt_path)
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()
    recs = {}
    print('---prepare target---')
    for index, image_file in enumerate(file_list):
        image_id = image_file[0]
        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        objects = []
        for i in range(num_obj):
            obj_struct = {}
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])
            class_name = cfg.VOC_CLASSES[c]
            target[(image_id, class_name)].append([x1, y1, x2, y2])
            obj_struct['name'] = class_name
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [x1, y1, x2, y2]
            objects.append(obj_struct)
        recs[image_id] = objects

    return image_list, target, recs


def chip_eval():
    target = defaultdict(list)
    preds = defaultdict(list)
    # image_list, target = prepare_target(target)
    image_list, target, recs = prepare_target(target)
    print('---start test---')

    model_path = cfg.out_model_path
    model = GtiModel(model_path)

    for image_path in tqdm(image_list):
        result = predict_chip(model, image_path, root_path=cfg.image_dir_path)
        for (x1, y1), (x2, y2), class_name, image_id, prob in result:
            preds[class_name].append([image_id, prob, x1, y1, x2, y2])
    print('---start evaluate---')
    voc_eval(preds, target, recs, VOC_CLASSES=cfg.VOC_CLASSES)


if __name__ == "__main__":
    chip_eval()
