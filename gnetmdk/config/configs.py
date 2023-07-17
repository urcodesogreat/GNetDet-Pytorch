import os
import math

from gnetmdk import ROOT
from gnetmdk.config.configurable import PostInit
from gnetmdk.dist import comm
from gnetmdk.utils.experiment import silent

STEP_SETTINGS = {
    #  quant_31, quant_r, quant_w
    1: (0,      0,      0),
    2: (0,      0,      1),
    3: (0,      1,      1),
    4: (1,      1,      1),
}


class BaseConfig(metaclass=PostInit):

    #  Dont' touch these values!!
    MDK_ROOT: str = ROOT
    VOC_CLASSES: list = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    DS_NAME: str = "VOC0712"    # Dataset directory name under `MDK_DIR/data` directory

    def __init__(self):

        # training
        self.step: int = 1
        self.batch_size: int = 16
        self.num_epochs: int = 128
        self.num_workers: int = 0
        self.loss_weights: list = [5., 0.005]
        self.loss_normalizer: float = 100.
        self.loss_normalizer_momentum: float = 0.9
        self.lr_scheduler: str = "StepLR"   # The default scheduler used by GNetDet MDK
        self.step_size: int = 10
        self.gamma: float = 0.8

        # distributed
        self.world_size: int = 1
        self.local_rank: int = 0
        self.global_rank: int = 0
        self.gpus: int = 1
        self.device = "cuda:0"

        # switches
        self.quant_w: bool = False    # Turns on/off the quantized convolution training
        self.quant_r: bool = False    # Turns on/off the quantized activation training
        self.quant_31: bool = False   # Checks if the relu_cap is in a range of [0, 31] for each activation layer
        self.cal: bool = False        # Turns on/off relu_cap calculation
        self.edit_gain: bool = False  # Used in Step4, True if load ckpt from step3, else False

        # convert
        self.chip: str = "5801"
        self.dat_json_path = "gnetmdk/gti/json/5801_gnetdet_dat.json"
        self.model_json_path = "gnetmdk/gti/json/5801_gnetdet_model.json"
        self.dat_json_448_path = "gnetmdk/gti/json/5801_gnetdet_dat_448.json"
        self.model_json_448_path = "gnetmdk/gti/json/5801_gnetdet_model_448.json"
        self.cap_txt_path = "model/relu_cap.txt"
        self.convert_model_path = "model/convert_checkpoint.pth"
        self.best_ckpt_path = "checkpoint/step4/best.pth"
        self.out_model_path = "model/out.model"
        self.dump_mode: bool = False
        self.batch_norm: bool = False

        # bit match
        self.bit_match_model_path = "checkpoint/best/best.pth"

        # layers
        self.image_format: str = "BGR"
        self.image_size: int = 448      # 448 or 224
        self.grid_size: int = 14
        self.stride: int = -1           # Will be set automatically
        self.grid_depth: int = -1       # Will be set automatically
        self.chip_depth: int = -1       # Will be set automatically
        self.scale: float = -1          # Will be set automatically

        # dataset
        self.data_dir_path = "data/{ds_name}"
        self.image_dir_path = "data/{ds_name}/JPEGImages"
        self.label_dir_path = "data/{ds_name}/Annotations"
        self.train_txt_path = "data/meta/{ds_name}_train.txt"
        self.valid_txt_path = "data/meta/{ds_name}_valid.txt"

        # other
        self.log_freq: int = 10  # Logging frequencies
        self.checkpoint_path = "checkpoint/step{n}/best.pth"
        self.best_save_path = "checkpoint/step{n}/{prefix}best.pth"
        self.last_save_path = "checkpoint/step{n}/last.pth"

    def __init_subclass__(cls, **kwargs):
        base = cls.__base__
        _init_ = getattr(base, "_init_", None)
        if _init_ is not None:
            base.__init__ = _init_
            delattr(base, "_init_")

    def __post_init__(self):
        """Function invoked after init."""
        self.set_step()
        self.hook_abspath_setup()
        self.hook_xml2txt_setup()
        self.hook_ckpt_setup()
        self.hook_model_setup()
        self.set_edit_gain()

    def hook_model_setup(self):
        """
        Setup model output scale.
        """
        self.stride = self.image_size // self.grid_size
        self.grid_depth = 10 + len(self.VOC_CLASSES)
        self.chip_depth = self.get_chip_out(self.grid_depth)
        self.scale = self.get_scale()

    def hook_xml2txt_setup(self):
        """
        Setup xml2txt settings. This paths will be used in `xml2txt.py`.
        """
        path_info = {"ds_name": self.DS_NAME}
        self.data_dir_path = self.data_dir_path.format(**path_info)
        self.image_dir_path = self.image_dir_path.format(**path_info)
        self.label_dir_path = self.label_dir_path.format(**path_info)
        self.train_txt_path = self.train_txt_path.format(**path_info)
        self.valid_txt_path = self.valid_txt_path.format(**path_info)

    def hook_ckpt_setup(self):
        """
        Setup model save path settings.
        """
        for n in STEP_SETTINGS:
            os.makedirs(os.path.dirname(self.best_save_path).format(n=n), exist_ok=True)

        path_info = {"n": self.step, "prefix": "cal-" if self.cal else ""}
        self.checkpoint_path = self.checkpoint_path.format(**path_info)
        self.best_save_path = self.best_save_path.format(**path_info)
        self.last_save_path = self.last_save_path.format(**path_info)

    def hook_abspath_setup(self):
        """
        Setup all paths to abspath.
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, str) and name.endswith("_path"):
                setattr(self, name, os.path.join(self.MDK_ROOT, attr))

    def set_step(self):
        # Set epoch to 1 if cal
        if self.cal:
            self.step = 2
            self.num_epochs = 1
            print("\n[INFO] Activate `Calibration` step.", end="")

        step = self.step
        assert (
            step in STEP_SETTINGS
        ), f"Wrong step! Must be one of: {list(STEP_SETTINGS.keys())}"

        print(f"\nSet step: `{step}`")
        q31, qr, qw = STEP_SETTINGS[step]
        self.quant_w = bool(qw)
        self.quant_r = bool(qr)
        self.quant_31 = bool(q31)

    def set_edit_gain(self):
        self.edit_gain = (self.step == 4 and "step3" in self.checkpoint_path)
        if self.edit_gain:
            print("[INFO] Set edit-gain to `True`")

    @staticmethod
    def get_chip_out(model_out_dim):
        chip_out = 0
        for n in range(20):
            thresh = model_out_dim / math.pow(2, n)
            if thresh < 1:
                chip_out = int(math.pow(2, n))
                break
        return chip_out

    def get_scale(self):
        if self.step == 4:
            f = open(self.cap_txt_path)
            lines = f.readlines()
            scale = lines[-1]
            f.close()
            return float(scale)
        else:
            return 31.0

    def get_ckpt_path(self, suffix: str=""):
        """Return Saved Path."""
        step_dir = os.path.join(self.MDK_ROOT, "checkpoint", f"step{self.step}")
        suffix = '' if not suffix else f"-{suffix}"
        best_save_path = os.path.join(step_dir, f"best{suffix}.pth")
        last_save_path = os.path.join(step_dir, f"last{suffix}.pth")
        return best_save_path, last_save_path

    def merge_opts_with_known_conf(self, args):
        """Merge with args.opts to config, override old attributes."""
        if not hasattr(args, "opts"):
            return
        if not args.opts:
            return

        OPEN_BRACKETS = "[("
        CLOSE_BRACKETS = "])"
        SEP_CHAR = ','
        SKIP_CHAR = ' \'\"'

        def parse_tree(astring: str, idx=0):
            children = None
            value = ""
            while idx < len(astring):
                c = astring[idx]
                if c in OPEN_BRACKETS:
                    if children is None:
                        children = []
                        idx += 1
                        continue
                    value, idx = parse_tree(astring, idx)
                elif c in SEP_CHAR:
                    if children is not None:
                        children.append(value)
                        value = ""
                        idx += 1
                    else:
                        raise ValueError
                elif c in CLOSE_BRACKETS:
                    if value and children is not None:
                        children.append(value)
                    return children, idx + 1
                elif c in SKIP_CHAR:
                    idx += 1
                else:
                    value += c
                    idx += 1
            return value, idx

        def convert_type(attr, new_attr):
            if not isinstance(new_attr, list):
                if isinstance(attr, str):
                    return str(new_attr)
                if isinstance(attr, bool):
                    if new_attr in ["True", "False"]:
                        return {"True": True, "False": False}[new_attr]
                    return bool(float(new_attr))
                if '.' in new_attr or 'e' in new_attr:
                    new_attr = float(new_attr)
                return type(attr)(new_attr)

            counts = len(new_attr) - len(attr)
            if counts != 0:
                # Only list with same type are allowed
                first_type = None
                for value in attr:
                    if first_type is None:
                        first_type = type(value)
                    else:
                        if first_type != type(value):
                            raise TypeError
                attr += [first_type() for _ in range(counts)]

            attr = type(attr)(
                convert_type(v0, v1)
                for v0, v1 in zip(attr, new_attr)
            )
            return attr

        # Iterate opts and override configs
        iter_opts = iter(args.opts)
        while True:
            try:
                attr_k = next(iter_opts)
            except StopIteration:
                break

            # opts passing by `key=value`
            if '=' in attr_k:
                attr_k, attr_v = attr_k.split('=')
            # opts passing by `key value`
            else:
                try:
                    attr_v = next(iter_opts)
                except StopIteration:
                    msg = f"{attr_k} got empty value."
                    raise ValueError(msg)

            attr_k = attr_k.strip().replace('-', '_')
            attr_v = attr_v.strip().replace(' ', '')

            # No need to set step here
            if attr_k == "step":
                print("[WARNING] Use `--step N` to set training step.")
                print(f"[WARNING] Current step: `{self.step}`")
                continue

            # Parse attr_v string
            try:
                attr_v, _ = parse_tree(attr_v)
            except Exception:
                msg = f"Invalid config: {attr_k} {attr_v}"
                raise ValueError(msg)

            if hasattr(self, attr_k):
                curr_attr_v = getattr(self, attr_k)

                # Convert to original type
                try:
                    attr_v = convert_type(curr_attr_v, attr_v)
                except Exception:
                    msg = f"Wrong type of argument {attr_k}, expect {curr_attr_v.__class__}"
                    raise TypeError(msg)
                else:
                    # override configs
                    setattr(self, attr_k, attr_v)
            else:
                msg = f"Invalid config: {attr_k}"
                raise ValueError(msg)

        # BUG fix: correct edit_gain if new checkpoint_path in opts
        with silent():
            self.set_edit_gain()

        # BUG fix: correct num_epochs if cal is True
        if self.cal:
            self.num_epochs = 1

    def rescale_to_distributed(self):
        """Set configs for distributed training."""
        self.world_size = comm.get_world_size()
        self.local_rank = comm.get_local_rank()
        self.global_rank = comm.get_rank()
        self.device = f"cuda:{self.local_rank}"
        
        if self.world_size > 1:
            self.batch_size = self.batch_size // self.world_size
            self.num_workers = self.num_workers // self.gpus
    
        if self.world_size > 1 and self.local_rank == 0:
            print("[INFO] Rescaled configs to distributed training")
            print("\tbatch-size:", self.batch_size)
            print("\tnum-workers:", self.num_workers)

    def clone(self):
        """Returns a copy of itself."""
        import copy
        return copy.deepcopy(self)

    def freeze(self):
        """Turns to Read-only mode."""
        cls = self.__class__

        def readonly(self, key, value):
            raise AttributeError("Read-only!")

        setattr(cls, "_readwrite_", cls.__setattr__)
        cls.__setattr__ = readonly
        return self

    def unfreeze(self):
        """Cancel Read-only mode."""
        cls = self.__class__
        readwrite = getattr(cls, "_readwrite_", None)
        if readwrite:
            cls.__setattr__ = readwrite
            delattr(cls, "_readwrite_")
        return self
