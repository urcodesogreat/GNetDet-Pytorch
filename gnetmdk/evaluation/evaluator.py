import abc
import torch
import numpy as np
import torch.distributed as dist

from typing import Optional, Union, List

from gnetmdk.dist import comm
from gnetmdk.layers import decoder_gpu_gnetdet
from gnetmdk.utils.experiment import global_average


class _BaseEvaluator(object):
    """
    Base class of Evaluator.
    """

    def __init__(self, iou_thresholds):
        self.iou_thresholds = iou_thresholds
        self.mAP = 0.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report()
        self.reset()

    @abc.abstractmethod
    def report(self):
        """Report mAP."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset state."""
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, *args, **kwargs):  # real signature unknown
        """Evaluate during each iteration."""
        raise NotImplementedError


class Evaluator(_BaseEvaluator):
    """
    Second edition of mAP Evaluator that runs faster than Evaluator.
    This version is able to calculate Mean-Average Precision over a set of iou-thresholds.

    NOTE:
        The final mAP (map@IoU=50) compared with the first version is different, but they are
        close. The first version calculate AP over predictions of all images for each class,
        but this version calculates mAPs over predictions of all classes for each image
        then take the mean value of mAPs.
    """

    def __init__(self, iou_thresholds: Optional[Union[float, List[float]]] = None):
        if isinstance(iou_thresholds, float):
            iou_thresholds = [iou_thresholds]
        iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05).tolist()
        assert isinstance(iou_thresholds, list), type(iou_thresholds)
        super(Evaluator, self).__init__(iou_thresholds)
        self.maps = {iou_thresh: 0. for iou_thresh in self.iou_thresholds}
        self.last = {iou_thresh: 0. for iou_thresh in self.iou_thresholds}
        self.avg_fns = [global_average() for _ in self.iou_thresholds]

    def eval(self, preds, gt_bboxes_list: List[torch.Tensor], whs, fnames):
        """
        Match preds with gt_bboxes. Each predicted bbox will be matched one or zero gt bbox.
        Then calculate AP for each image, and the final mAP is the mean over all individual APs.
        Args:
            preds:      [B, H, W, 10+C]
            gt_bboxes_list: List[[N, 5]]
        """
        # Useless arguments
        del whs, fnames

        # In case cuda device
        if preds.device != gt_bboxes_list[0].device:
            gt_bboxes_list = [gt_bboxes.to(preds.device) for gt_bboxes in gt_bboxes_list]

        # Evaluate mAP
        preds = decoder_gpu_gnetdet(preds)
        for b in range(len(preds)):
            preds_bbox, preds_idxs, preds_prob = preds[b]  # [M]
            gt_bboxes = gt_bboxes_list[b]  # [N]

            # sort by probs
            sortedIdx = torch.argsort(preds_prob, descending=True)
            preds_bbox = preds_bbox[sortedIdx]
            preds_idxs = preds_idxs[sortedIdx]

            # compute iou matrix
            iou_mat = self.iou(preds_bbox, gt_bboxes).cpu().numpy()  # [M, N]

            # calculate mAP per each iou-threshold
            for i, iou_thresh in enumerate(self.iou_thresholds):
                preds_match, gt_match = self._match(preds_bbox, preds_idxs, gt_bboxes, iou_mat, iou_thresh)
                mAP = self._calc_ap(preds_match, gt_match)
                self.maps[iou_thresh] = self.avg_fns[i](mAP)

    @staticmethod
    def _match(preds_bbox, preds_idxs, gt_bboxes, iou_mat, iou_threshold):
        """
        Match predicted box with gt boxes.
        Args:
            preds_bbox: [M, 4]
            preds_idxs: [M]
            gt_bboxes:  [N, 5]
            iou_mat:    [M, N]
            iou_threshold: float
        Returns:
            preds_match: [N]
            gt_match:    [M]
        """
        M, N = preds_bbox.shape[0], gt_bboxes.shape[0]

        # Loop through predictions and find matching gt boxes
        match_count = 0
        pred_match = -1 * np.ones([M])
        gt_match = -1 * np.ones([N])
        for i in range(M):
            # Find best matching gt box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(iou_mat[i])[::-1]

            # 2. Remove low scores
            low_score_idx = np.where(iou_mat[i, sorted_ixs] < iou_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]

            # 3. Find the match
            for j in sorted_ixs:
                # If gt box is already matched, go to next one
                if gt_match[j] > -1:
                    continue
                # If reach IoU smaller than the threshold, end the loop
                iou = iou_mat[i, j]
                if iou < iou_threshold:
                    break
                # Do we have a match?
                if int(preds_idxs[i]) == int(gt_bboxes[j, -1]):
                    match_count += 1
                    gt_match[j] = i
                    pred_match[i] = j
                    break
        return pred_match, gt_match

    @staticmethod
    def _calc_ap(preds_match, gt_match):
        """
        Calculate AP given matches.
        Args:
            preds_match: [M]
            gt_match:    [N]
        Returns:
            mAP
        """
        # Compute precision and recall at each prediction box step
        precisions = np.cumsum(preds_match > -1) / (np.arange(len(preds_match)) + 1)
        recalls = np.cumsum(preds_match > -1).astype(np.float32) / len(gt_match)

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # Compute mean AP over recall range
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        return mAP.item()

    @staticmethod
    def iou(bboxes1, bboxes2):
        """
        Calculate pairwise iou.
        Args:
            bboxes1: [M, 4]
            bboxes2: [N, 4]

        Returns:
            iou_mat: [M, N]
        """
        bboxes1 = bboxes1.unsqueeze(-2).clamp(min=0., max=1.)   # [M, 1, 4]
        bboxes2 = bboxes2.unsqueeze(-3)                         # [1, N, 4]
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        inter_xy_tl = torch.maximum(bboxes1[..., :2], bboxes2[..., :2])
        inter_xy_br = torch.minimum(bboxes1[..., 2:4], bboxes2[..., 2:4])
        inter_wh = (inter_xy_br - inter_xy_tl).clamp(min=0.)
        area_inter = inter_wh[..., 0] * inter_wh[..., 1]
        return area_inter / (area1 + area2 - area_inter)

    def reset(self):
        self.last = self.maps.copy()
        self.maps = {iou_thresh: 0. for iou_thresh in self.iou_thresholds}
        self.avg_fns = [global_average() for _ in self.iou_thresholds]

    def report(self):
        """
        Print mAP @ IoU table.
        """
        print("=" * 80)
        print(f"  mAP @ IoU\t\tValue\t\tImproved\t\tValue")
        print("-" * 80)
        formater = "{iou:^9d}\t\t{mAP:<.6f}\t\t{imp:^6}\t\t{delta:<.4f}"

        for i, iou_thresh in enumerate(self.iou_thresholds):
            curr_mAP = self.maps[iou_thresh]
            last_mAp = self.last[iou_thresh]
            delta = curr_mAP - last_mAp
            if curr_mAP > last_mAp:
                imp = '+'
            elif curr_mAP < last_mAp:
                imp = '-'
            else:
                imp = '='
            print(formater.format(iou=int(iou_thresh * 100), mAP=curr_mAP, imp=imp, delta=delta))

        self.mAP = np.mean(list(self.maps.values())).item()
        print("-" * 80)
        print(f"  Mean   \t\t{self.mAP:<.6f}")
        print("=" * 80)


class DistEvaluator(Evaluator):
    """Evaluator for distributed training."""

    def __init__(self, iou_thresholds: Optional[Union[float, List[float]]] = None):
        super(DistEvaluator, self).__init__(iou_thresholds)
        self.local_rank = comm.get_local_rank()
        self.world_size = comm.get_world_size()
        self.device = torch.device(self.local_rank)

    def __exit__(self, exc_type, exc_val, exc_tb):
        res_maps_list = [torch.zeros(len(self.iou_thresholds), device=self.device) for _ in range(self.world_size)]
        curr_maps = torch.tensor([self.maps[iou] for iou in self.iou_thresholds], device=self.device)
        dist.all_gather(res_maps_list, curr_maps)
        res_maps = torch.stack(res_maps_list, dim=0).mean(dim=0).cpu()
        for idx, iou in enumerate(self.iou_thresholds):
            self.maps[iou] = res_maps[idx].item()
        self.report()
        self.reset()

    def report(self):
        if self.local_rank == 0:
            super(DistEvaluator, self).report()
        else:
            self.mAP = np.mean(list(self.maps.values())).item()
