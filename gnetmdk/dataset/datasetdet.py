import os
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as trans

from gnetmdk.config.configurable import configurable
from gnetmdk.dataset.augments import Augmentation


class DatasetDet(data.Dataset):

    @configurable
    def __init__(self,
                 root,
                 list_file,
                 train,
                 image_format="BGR",
                 input_size=448,
                 grid_num=14,
                 grid_dim=30,
                 transforms=None):
        self.root = root
        self.train = train
        self.image_format = image_format
        self.input_size = input_size
        self.grid_dim = grid_dim
        self.grid_num = grid_num

        self.aug = Augmentation()
        if transforms is None:
            self.transforms = [trans.ToTensor()]

        if isinstance(list_file, list):
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()

        self.fnames = []
        self.boxes = []
        self.labels = []
        # self.mean = (123, 117, 104)  # RGB

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # img = self.trans.up_noise(img)
        if self.train:
            # img = self.random_bright(img)
            img, boxes = self.aug.random_flip(img, boxes)
            img, boxes = self.aug.randomScale(img, boxes)
            img = self.aug.randomBlur(img)
            img = self.aug.RandomBrightness(img)
            img = self.aug.RandomHue(img)
            img = self.aug.RandomSaturation(img)
            img, boxes, labels = self.aug.randomShift(img, boxes, labels)
            img, boxes, labels = self.aug.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        # img = self.trans.BGR2RGB(img) # because pytorch pretrained layers use RGB
        # img = self.trans.subMean(img,self.mean)

        img = cv2.resize(img, (self.input_size, self.input_size))

        if self.image_format == "YUV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif self.image_format == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.aug.quant_data(img)
        for transform in self.transforms:
            img = transform(img)

        target = self.encoder(boxes, labels)
        gt_boxes = torch.cat([boxes, labels.view(-1, 1) - 1], dim=-1)
        wh = torch.tensor([w, h])
        return img, target, gt_boxes, wh, fname

    def encoder(self, bboxs, labels):
        """Vectorized version of encoder.

        Args:
            bboxs: [N, 4], 4 indicates (x_tl, y_tl, x_br, y_br)
            labels: [N]
        Returns:
            target: [H, W, 10+C]
        """
        h_amap = w_amap = self.grid_num

        # Normalized scale
        cxcy = (bboxs[:, 2:] + bboxs[:, :2]) / 2  # [N, 2]
        wh = bboxs[:, 2:] - bboxs[:, :2]  # [N, 2]

        # Rescale coords to feature map
        cxcy = cxcy * torch.tensor([w_amap, h_amap])  # [N, 2]

        # deltas from box centers to grids centers
        grids = generate_grids(h_amap, w_amap)
        deltas = cxcy - grids.view(-1, 1, 2)  # [HW, N, 2]

        # Calculate l1 distance and get activated grids
        mah_dist = torch.abs(deltas).sum(dim=-1)  # [HW, N]
        activated_grid_idxs = mah_dist.min(dim=0)[1]  # [N]

        # deltas to top-left corner
        delta_xy = cxcy - grids.view(-1, 2)[activated_grid_idxs] + 0.5  # [N, 2]

        # Fill target
        target = torch.zeros([h_amap * w_amap, self.grid_dim], dtype=torch.float32, device=bboxs.device)
        target[activated_grid_idxs, 4] = 1.
        target[activated_grid_idxs, 9] = 1.
        target[activated_grid_idxs, :2] = delta_xy
        target[activated_grid_idxs, 2:4] = wh
        target[activated_grid_idxs, 5:7] = delta_xy
        target[activated_grid_idxs, 7:9] = wh
        target[activated_grid_idxs, 9 + labels] = 1.
        return target.view(h_amap, w_amap, self.grid_dim)

    @classmethod
    def from_config(cls, cfg, train=False, transforms=None):
        return {
            "root":  cfg.image_dir_path,
            "list_file": cfg.train_txt_path if train else cfg.valid_txt_path,
            "train": train,
            "image_format": cfg.image_format,
            "input_size": cfg.image_size,
            "grid_num": cfg.grid_size,
            "grid_dim": cfg.grid_depth,
            "transforms": transforms,
        }


def generate_grids(h_amap, w_amap, device="cpu"):
    """Generate grid centers on feature map.

    Returns:
        grids: [H, W, 2]
    """
    w_range = torch.arange(w_amap, dtype=torch.float32, device=device) + 0.5
    h_range = torch.arange(h_amap, dtype=torch.float32, device=device) + 0.5
    girds_xs = w_range.view(1, -1).expand(h_amap, -1)
    grids_ys = h_range.view(-1, 1).expand(-1, w_amap)
    grids = torch.stack([girds_xs, grids_ys], dim=-1)
    return grids


def collate_fn(batch_list: list):
    batch_imgs = []
    batch_targets = []
    batch_gt_boxes = []
    batch_wh = []
    batch_fnames = []
    for b in range(len(batch_list)):
        img, target, gt_boxes, wh, fname = batch_list[b]
        batch_imgs.append(img)
        batch_targets.append(target)
        batch_gt_boxes.append(gt_boxes)
        batch_wh.append(wh)
        batch_fnames.append(fname)

    batch_imgs = torch.stack(batch_imgs, dim=0)
    batch_targets = torch.stack(batch_targets, dim=0)
    batch_wh = torch.stack(batch_wh, dim=0)
    return batch_imgs, batch_targets, batch_gt_boxes, batch_wh, batch_fnames
