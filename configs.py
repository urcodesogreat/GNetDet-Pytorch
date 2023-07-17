from gnetmdk.config import BaseConfig

_cfg = None
DEFAULT_STEP = 1


def get_config(step=None, cal=False):
    global _cfg
    if _cfg is None:
        if step is None:
            step = DEFAULT_STEP
        _cfg = Config(step, cal)
    return _cfg


class Config(BaseConfig):

    def __init__(self, step=1, cal=False):
        super().__init__()

        # Training step
        self.step = step  # IMPORTANT
        self.cal = cal    # IMPORTANT

        # Basic configs
        self.image_size = 448       # 448 or 224
        self.image_format = "BGR"   # BGR or RGB or YUV
        self.VOC_CLASSES = [
            "head",
        ]
        self.DS_NAME = "HT21"        # Dataset directory name under `data` directory
        self.checkpoint_path = "checkpoint/step0/best.pth"
        self.batch_norm = False

        # Training configs
        self.loss_weights = [5., 0.005]      # [loc_loss, noobj_loss]
        self.learning_rate = 1e-3
        self.batch_size = 16
        self.num_epochs = 200
        self.weight_decay = 3e-6
        self.lr_momentum = 0.9
        self.num_workers = 4
        self.lr_scheduler = "StepLR"
        self.step_size = 20
        self.gamma = 0.75
        self.warmup_epochs: int = 2

        # Test configs
        self.Display = "mp4"                # "mp4" or "camera"
        self.mp4_path = "data/test.mp4"
        self.model_path = "model/out.model"
        self.conf_thresh = 0.48             # TC
        self.prob_thresh = 0.4              # TP
        self.nms_thresh = 0.4
