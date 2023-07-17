# encoding:utf-8
import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.ops import nms as nms_v2
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import get_config


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


def decoder(pred, cfg=None):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    if cfg is None:
        cfg = get_config()

    pred = pred.clone()
    grid_num = cfg.grid_size
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # 7x7x30
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = (contain > 0.1).to(torch.float32)  # 0.1 threshold
    mask2 = (contain == contain.max()).to(torch.float32)  # Always select the best contain_prob when>0.9
    mask = (mask1 + mask2).gt(0.)

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    # print(i,j,b)
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # up left of cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)

                    cls_index = cls_index.unsqueeze(0)

                    if float((contain_prob * max_prob)[0]) > - 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)

    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indexs = torch.cat(cls_indexs, 0)  # (n,)

    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def decoder_v2(preds):
    """
    Args:
        preds: [B, H, W', 5A+C]
    """
    preds = preds.clone()
    B, h_amap, w_amap = preds.shape[:3]
    A = 2
    dtype = preds.dtype
    device = preds.device

    preds_bbox = preds[..., :10].view(B, h_amap, w_amap, A, 5)  # [B, H, W, A, 5]
    preds_conf = preds_bbox[..., 4].contiguous()  # [B, H, W, A]
    preds_bbox = preds_bbox[..., :4].contiguous()  # [B, H, W, A, 4]
    preds_prob = preds[..., 10:].contiguous()  # [B, H, W, C]
    preds_prob = preds_prob.unsqueeze(3).repeat(1, 1, 1, A, 1)  # [B, H, W, A, C]

    # Get valid boxes: (confs > 0.1) or (with max conf)
    mask_contain = (preds_conf > 0.1)
    max_conf_per_batch = torch.max(preds_conf.view(B, -1), dim=-1)[0]
    mask_max_conf = (preds_conf == max_conf_per_batch.view(B, 1, 1, 1))
    mask_contain |= mask_max_conf  # [B, H, W, A]
    valid_idxs = torch.nonzero(mask_contain.view(-1)).squeeze(-1)  # [M]
    preds_bbox = preds_bbox.view(-1, 4)[valid_idxs]  # [M, 4]
    preds_conf = preds_conf.view(-1)[valid_idxs]  # [M]
    preds_prob = preds_prob.view(-1, preds_prob.shape[-1])[valid_idxs]  # [M, C]

    # offsets of each grid
    shift_ys = torch.arange(h_amap, dtype=dtype, device=device)
    shift_xs = torch.arange(w_amap, dtype=dtype, device=device)
    shift_ys, shift_xs = torch.meshgrid([shift_ys, shift_xs])
    offsets = torch.stack([shift_xs, shift_ys], dim=-1)  # [H, W, 2]
    offsets = offsets.view(1, h_amap, w_amap, 1, 2).repeat(B, 1, 1, A, 1)  # [B, H, W, A, 2]
    offsets = offsets.view(-1, 2)[valid_idxs]  # [M, 2]

    # Get center coords of predicted bboxes
    # NOTE: 1. preds_bbox[:, :2] are deltas-distances against top-left corner, in feature map scale.
    #       2. preds_bbox[:, 2:] are width-height in normalized scale
    #       3. Need to normalize preds_bbox[:, :2]
    preds_bbox[:, :2] = (preds_bbox[:, :2] + offsets)
    preds_bbox[:, :2] /= torch.tensor([w_amap, h_amap], dtype=dtype, device=device)

    # Change xy-wh to xy-xy
    preds_bbox_xyxy = torch.zeros_like(preds_bbox)
    preds_bbox_xyxy[:, :2] = preds_bbox[:, :2] - 0.5 * preds_bbox[:, 2:]
    preds_bbox_xyxy[:, 2:] = preds_bbox[:, :2] + 0.5 * preds_bbox[:, 2:]

    # Confidence thresholding by (conf * max-prob > 0.1)
    max_probs, preds_idxs = torch.max(preds_prob, dim=1)
    preds_prob = preds_conf * max_probs  # [M]
    mask_filter = (preds_prob > 0.1)  # [M]

    final_bboxes, final_probs, final_idxs = [], [], []
    # This function aims to handling multiple batches, hence we need to tell which batch
    # the predictions belongs to. We then apply nms per batch bboxes.
    num_ancs_per_batch = h_amap * w_amap * A
    for b in range(B):
        mask_valid_idxs_per_batch = (valid_idxs >= b * num_ancs_per_batch) & (valid_idxs < (b + 1) * num_ancs_per_batch)
        mask_filter_per_batch = mask_filter & mask_valid_idxs_per_batch
        preds_bbox_per_batch = preds_bbox_xyxy[mask_filter_per_batch]
        preds_prob_per_batch = preds_prob[mask_filter_per_batch]
        preds_idxs_per_batch = preds_idxs[mask_filter_per_batch]

        # NMS thresholding
        # keep = nms(preds_bbox_per_batch, preds_prob_per_batch)
        keep = nms_v2(preds_bbox_per_batch, preds_prob_per_batch, iou_threshold=0.5)
        final_bboxes.append(preds_bbox_per_batch[keep])
        final_probs.append(preds_prob_per_batch[keep])
        final_idxs.append(preds_idxs_per_batch[keep])
    return list(zip(final_bboxes, final_idxs, final_probs))


def nms(bboxes, scores, threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
        if order.numel() == 1:
            order = order.unsqueeze(0)
    return torch.LongTensor(keep)


#
# start predict one image
#


def quant_data(x):
    x = x.astype(np.uint8)
    x = np.right_shift(x, np.ones_like(x) * 2)
    x += 1
    x = np.right_shift(x, np.ones_like(x)).astype(np.float32)
    x = np.clip(x, 0, 31)
    return x


def predict_gpu(model, image_name, root_path='', cfg=None):
    if cfg is None:
        cfg = get_config()

    result = []
    image = cv2.imread(os.path.join(root_path, image_name))
    h, w, _ = image.shape
    img = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)

    img = quant_data(img)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img).unsqueeze(0)
    img = img.cuda()

    with torch.no_grad():
        pred = model(img)  # 1x7x7x30
    pred = pred.cpu()

    # Old Version
    # boxes, cls_indexs, probs = decoder(pred)

    # New Version
    boxes, cls_indexs, probs = decoder_v2(pred)[0]

    # # ---- TEST SPEEDUP (Approx 3~30x faster)----
    # t1 = time.time()
    # decoder(pred)
    # t2 = time.time()
    # decoder_v2(pred)
    # t3 = time.time()
    # print(f"Old version: {t2 - t1:.3f} s,  New Version: {t3-t2:.3f}s,   SPEEDUP:{(t2-t1)/(t3-t2):.2f}x")
    # # --------------------------------------------

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), cfg.VOC_CLASSES[cls_index], image_name, prob])
    return result, pred


def predict_chip(chip_model, image_name, root_path='', cfg=None):
    if cfg is None:
        cfg = get_config()

    result = []
    image = cv2.imread(os.path.join(root_path, image_name))
    h, w, _ = image.shape
    img = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)

    pred = chip_model.evaluate(img)[:, :30, :, :]

    pred = pred * (cfg.scale / 31)
    pred = torch.tensor(pred)

    pred = pred.permute(0, 2, 3, 1)
    pred = pred.cpu()

    boxes, cls_indexs, probs = decoder(pred)  ##decoder
    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), cfg.VOC_CLASSES[cls_index], image_name, prob])
    return result
