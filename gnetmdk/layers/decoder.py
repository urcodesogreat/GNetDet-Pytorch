import torch
from torchvision.ops import nms


def decoder_gpu_gnetdet(preds):
    """
    Args:
        preds: [B, H, W, 5A+C]
    """
    preds = preds.clone()
    B, h_amap, w_amap = preds.shape[:3]
    A = 2
    dtype = preds.dtype
    device = preds.device

    preds_bbox = preds[..., :10].view(B, h_amap, w_amap, A, 5)  # [B, H, W, A, 5]
    preds_conf = preds_bbox[..., 4].contiguous()                # [B, H, W, A]
    preds_bbox = preds_bbox[..., :4].contiguous()               # [B, H, W, A, 4]
    preds_prob = preds[..., 10:].contiguous()                   # [B, H, W, C]
    preds_prob = preds_prob.unsqueeze(3).repeat(1, 1, 1, A, 1)  # [B, H, W, A, C]

    # Get valid boxes: (confs > 0.1) or (with max conf)
    mask_contain = (preds_conf > 0.1)                           # [B, H, W, A]
    max_conf_per_batch = torch.max(preds_conf.view(B, -1), dim=-1)[0]
    mask_max_conf = (preds_conf == max_conf_per_batch.view(B, 1, 1, 1))
    mask_contain |= mask_max_conf                                       # [B, H, W, A]
    valid_idxs = torch.nonzero(mask_contain.view(-1)).squeeze(-1)       # [M]
    preds_bbox = preds_bbox.view(-1, 4)[valid_idxs]                     # [M, 4]
    preds_conf = preds_conf.view(-1)[valid_idxs]                        # [M]
    preds_prob = preds_prob.view(-1, preds_prob.shape[-1])[valid_idxs]  # [M, C]

    # offsets of each grid
    shift_ys = torch.arange(h_amap, dtype=dtype, device=device)
    shift_xs = torch.arange(w_amap, dtype=dtype, device=device)
    shift_ys, shift_xs = torch.meshgrid([shift_ys, shift_xs])
    offsets = torch.stack([shift_xs, shift_ys], dim=-1)                     # [H, W, 2]
    offsets = offsets.view(1, h_amap, w_amap, 1, 2).repeat(B, 1, 1, A, 1)   # [B, H, W, A, 2]
    offsets = offsets.view(-1, 2)[valid_idxs]                               # [M, 2]

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
    preds_prob = preds_conf * max_probs             # [M]
    mask_filter = (preds_prob > 0.1)                # [M]

    final_bboxes, final_probs, final_idxs = [], [], []
    # This function aims to handling batch images, hence we need to tell which image
    # the predictions belongs to. We then apply nms per batch bboxes.
    num_ancs_per_batch = h_amap * w_amap * A
    for b in range(B):
        mask_valid_idxs_per_batch = (valid_idxs >= b * num_ancs_per_batch) & (valid_idxs < (b + 1) * num_ancs_per_batch)
        mask_filter_per_batch = mask_filter & mask_valid_idxs_per_batch
        preds_bbox_per_batch = preds_bbox_xyxy[mask_filter_per_batch]
        preds_prob_per_batch = preds_prob[mask_filter_per_batch]
        preds_idxs_per_batch = preds_idxs[mask_filter_per_batch]

        # NMS thresholding
        keep = nms(preds_bbox_per_batch, preds_prob_per_batch, iou_threshold=0.5)
        final_bboxes.append(preds_bbox_per_batch[keep])
        final_probs.append(preds_prob_per_batch[keep])
        final_idxs.append(preds_idxs_per_batch[keep])
    return list(zip(final_bboxes, final_idxs, final_probs))
