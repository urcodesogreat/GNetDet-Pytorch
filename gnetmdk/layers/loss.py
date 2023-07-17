# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnetmdk.config import configurable


class GNetDetLoss(nn.Module):

    @configurable
    def __init__(self, l_coord, l_noobj, l_norml, l_momtm, grid_num):
        super(GNetDetLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_norml = l_norml
        self.l_momtm = l_momtm
        self.grid_num = grid_num

    def pairwise_iou(self, bbox1, bbox2):
        """
        Args:
            bbox1: [AN, 4]
            bbox2: [AN, 4]

        Returns:
            iou:   [AN]
        """
        assert bbox1[..., :4].shape == bbox2[..., :4].shape

        bbox1_xyxy = torch.zeros_like(bbox1[..., :4])
        bbox1_xyxy[..., :2] = bbox1[..., :2] / self.grid_num - 0.5 * bbox1[..., 2:4]
        bbox1_xyxy[..., 2:4] = bbox1[..., :2] / self.grid_num + 0.5 * bbox1[..., 2:4]
        area_bbox1 = (bbox1_xyxy[..., 2] - bbox1_xyxy[..., 0]) * (bbox1_xyxy[..., 3] - bbox1_xyxy[..., 1])

        bbox2_xyxy = torch.zeros_like(bbox2[..., :4])
        bbox2_xyxy[..., :2] = bbox2[..., :2] / self.grid_num - 0.5 * bbox2[..., 2:4]
        bbox2_xyxy[..., 2:4] = bbox2[..., :2] / self.grid_num + 0.5 * bbox2[..., 2:4]
        area_bbox2 = (bbox2_xyxy[..., 2] - bbox2_xyxy[..., 0]) * (bbox2_xyxy[..., 3] - bbox2_xyxy[..., 1])

        inter_xytl = torch.maximum(bbox1_xyxy[..., :2], bbox2_xyxy[..., :2])
        inter_xybr = torch.minimum(bbox1_xyxy[..., 2:4], bbox2_xyxy[..., 2:4])
        inter_wh = (inter_xybr - inter_xytl).clamp(min=0.)
        area_inter = inter_wh[..., 0] * inter_wh[..., 1]

        iou = area_inter / (area_bbox1 + area_bbox2 - area_inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        pred_tensor: [B, H, W, 10+C]
        target_tensor: [B, H, W, 10+C]
        """
        pred_tensor = pred_tensor + 1e-8
        coo_mask = target_tensor[..., 4] > 0.
        noo_mask = torch.logical_not(coo_mask)

        # 1. not contain obj loss
        noo_pred = pred_tensor[noo_mask]  # [M, 10+C]
        noo_targ = target_tensor[noo_mask]  # [M, 10+C]
        noo_pred_conf = noo_pred[..., [4, 9]].contiguous().view(-1)  # [2M]
        noo_targ_conf = noo_targ[..., [4, 9]].contiguous().view(-1)  # [2M]
        noo_obj_loss = F.smooth_l1_loss(noo_pred_conf, noo_targ_conf, reduction="sum")

        # 2. response loss
        coo_pred = pred_tensor[coo_mask]  # [N, 10+C]
        coo_targ = target_tensor[coo_mask]  # [N, 10+C]
        box_pred = coo_pred[..., :10].reshape(-1, 5)  # [2N, 5]
        box_targ = coo_targ[..., :10].reshape(-1, 5)  # [2N, 5]

        ious = self.pairwise_iou(box_pred, box_targ).view(-1, 2)  # [N, 2]
        max_ious, max_iou_idx = ious.max(dim=1)  # [N]
        n = torch.arange(ious.shape[0])
        box_pred_response = box_pred.view(-1, 2, 5)[n, max_iou_idx, :]  # [N, 5]
        box_targ_response = box_targ.view(-1, 2, 5)[n, max_iou_idx, :]  # [N, 5]

        contain_loss = F.smooth_l1_loss(box_pred_response[:, 4], max_ious, reduction="sum")
        loc_loss = F.smooth_l1_loss(box_pred_response[:, :2], box_targ_response[:, :2], reduction="sum") + \
                   F.smooth_l1_loss(box_pred_response[:, 2:4].sqrt(), box_targ_response[:, 2:4].sqrt(), reduction="sum")

        # 3. not response loss
        min_iou_idx = torch.ones_like(max_iou_idx) - max_iou_idx
        box_pred_not_response = box_pred.view(-1, 2, 5)[n, min_iou_idx, :]
        box_targ_not_response = box_targ.view(-1, 2, 5)[n, min_iou_idx, :]
        box_targ_not_response[:, 4] = 0.
        not_contain_loss = F.smooth_l1_loss(box_pred_not_response[:, 4], box_targ_not_response[:, 4], reduction="sum")

        # 4. class loss
        cls_pred = coo_pred[..., 10:]
        cls_targ = coo_targ[..., 10:]
        cls_loss = F.smooth_l1_loss(cls_pred, cls_targ, reduction="sum")

        self.l_norml = self.l_momtm * self.l_norml + (1 - self.l_momtm) * max(n.numel(), 1)
        losses = {
            "loc_loss": self.l_coord * loc_loss / self.l_norml,
            "contain_loss": 2 * contain_loss / self.l_norml,
            "not_contain_loss": not_contain_loss / self.l_norml,
            "noobj_loss": self.l_noobj * noo_obj_loss / self.l_norml,
            "cls_loss": cls_loss / self.l_norml,
        }
        # number positive/negative ratio
        # print("+/-: %.2f" % (coo_pred.shape[0] / noo_pred.shape[0]))
        return losses

    @classmethod
    def from_config(cls, cfg):
        return {
            "l_coord": cfg.loss_weights[0],
            "l_noobj": cfg.loss_weights[1],
            "l_norml": cfg.loss_normalizer,
            "l_momtm": cfg.loss_normalizer_momentum,
            "grid_num": cfg.grid_size,
        }


class RegLoss(object):
    """
    Compute L1, L2 or Elastic(L1+L2) regularization loss.

    NOTE:
        If use this loss, one must set `weight_decay = 0` in optimizer where l2 reg loss is computed.
    """

    def __init__(self, params, weight_decay: float = 1e-5, reg_loss: str = "l2"):
        reg_loss = reg_loss.lower()
        _REG_LOSS = {"l1", "l2", "elastic"}
        assert reg_loss in _REG_LOSS, f"reg_loss must be one of: {_REG_LOSS}"
        params = list(params)
        for p in params:
            assert isinstance(p, nn.Parameter), f"Unknown parameter type: {type(p)}"
        self.params = params
        if reg_loss == "elastic":
            assert (
                isinstance(weight_decay, (list, tuple)) and len(weight_decay) == 2
            ), f"`weight_decay` must be a tuple of length 2"
            lambda1, lambda2 = weight_decay
            l1 = l2 = True
        elif reg_loss == "l1":
            lambda1, lambda2 = weight_decay, 0.
            l1 = True
            l2 = False
        else:
            lambda1, lambda2 = 0., weight_decay
            l1 = False
            l2 = True
        self.l1 = l1
        self.l2 = l2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self):
        l1_loss = 0.
        l2_loss = 0.
        for p in self.params:
            if not p.requires_grad:
                continue
            if self.l1:
                l1_loss = l1_loss + self.lambda1 * torch.abs(p).sum()
            if self.l2:
                l2_loss = l2_loss + self.lambda2 * torch.square(p).sum()

        loss = {}
        if self.l1:
            loss.update(l1_loss=l1_loss)
        if self.l2:
            loss.update(l2_loss=l2_loss)
        return loss

    def __call__(self):
        return self.forward()


class GNetDetWithRegLoss(object):

    def __init__(self, cfg, model, reg_loss="l2"):
        self._data_loss = GNetDetLoss(cfg)
        self._reg_loss = RegLoss(
            filter(lambda p: p.requires_grad, model.parameters()),
            weight_decay=cfg.weight_decay, reg_loss=reg_loss,
        )

    def compute_data_loss(self, inputs, labels):
        return self._data_loss(inputs, labels)

    def compute_reg_loss(self):
        return self._reg_loss()

    def forward(self, inputs, labels):
        data_loss = self.compute_data_loss(inputs, labels)
        reg_loss = self.compute_reg_loss()
        return data_loss, reg_loss

    def __call__(self, inputs, labels):
        return self.forward(inputs, labels)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    [Link](http://ydwen.github.io/papers/WenECCV16.pdf)

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes: int = 10, feat_dim: int = 2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


if __name__ == '__main__':
    # x = torch.randn([4, 2])
    # y = torch.tensor([0, 1, 1, 0])
    # l = CenterLoss(5, 2)
    #
    # z = l(x, y)
    # print(z)
    pass