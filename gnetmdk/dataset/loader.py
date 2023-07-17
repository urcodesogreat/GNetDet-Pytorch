from torch.utils.data import DataLoader, DistributedSampler

from gnetmdk.config import configurable
from gnetmdk.dist.comm import get_world_size
from gnetmdk.dataset import DatasetDet, collate_fn


def get_dataloader_from_config(cfg, train=False):
    return {
        "dataset": DatasetDet(cfg, train=train),
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "train": train,
    }


@configurable(from_config=get_dataloader_from_config)
def get_gnetdet_dataloader(dataset, train=False, batch_size=16, num_workers=0):
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=train, drop_last=True)
    else:
        sampler = None

    # TODO: [W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
    #       When pin_memory=True
    return sampler, DataLoader(dataset,
                               batch_size,
                               shuffle=(train and get_world_size() <= 1),
                               sampler=sampler,
                               num_workers=num_workers,
                               collate_fn=collate_fn,
                               pin_memory=False,
                               drop_last=True)
