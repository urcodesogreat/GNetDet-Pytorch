import os
import sys
import yaml
import keyword
import argparse
import reprlib

from typing import List, Dict, Any, MutableMapping, MutableSequence, Callable, Optional

__all__ = ["Config", "get_config"]


VALID_CHIPS     : List[str] = ["5801", "SIT501"]
VALID_DET_TYPE  : List[str] = ["camara", "image", "video"]
VALID_IMG_EXT   : List[str] = [".jpg", ".jpeg", ".png"]
VALID_VID_EXT   : List[str] = [".mp4", ".avi"]


def _assert_with_msg_or_exit(assertion: bool, msg: str):
    if not assertion:
        print(msg)
        sys.exit(-1)


def _check_path_is_valid_type_of(valid_exts: List[str], path: str):
    _, ext = os.path.splitext(path)
    return ext in valid_exts


def _get_files(directory: str, pred: Optional[Callable[[str], bool]] = None, recursive: Optional[bool] = True):
    paths = []
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if pred is not None and callable(pred) and not pred(file):
                    continue
                paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if pred is not None and callable(pred) and not pred(file):
                continue
            paths.append(os.path.join(directory, file))
    return paths


def _check_keys_are_all_valid(d: Dict):
    for k, v in d.items():
        if keyword.iskeyword(k):
            print(f"[ERROR] Python preserved key: {k}!")
            sys.exit(-1)
        if isinstance(k, str) and not k.isidentifier():
            print(f"[ERROR] Invalid python identifier: {k}!")
            sys.exit(-1)
        if isinstance(v, Dict):
            _check_keys_are_all_valid(v)


def _parse_yaml(file: str):
    """Load configuration from yaml file."""
    if not os.path.exists(file):
        raise FileNotFoundError(f"Config yaml file does not exist: {file}")
    assert file.endswith(".yaml"), f"Not a yaml file: {file}"
    try:
        with open(file, 'r') as f:
            _c = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(-1)
    return Config(_c)


class Config(object):
    """Config class to parse yaml files."""
    _instance = False

    def __new__(cls, arg: Any):
        if isinstance(arg, MutableMapping):
            return super().__new__(cls)
        elif isinstance(arg, MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping: MutableMapping[str, Any]):
        _check_keys_are_all_valid(mapping)
        self._c = mapping
        self._instance = True

    def __getattr__(self, key: str):
        if not self._instance:
            return super().__getattribute__(key)
        if hasattr(self._c, key):
            return getattr(self._c, key)
        else:
            _assert_with_msg_or_exit(key in self._c, f"[ERROR] `{key}` not in cfg, current keys: {list(self._c.keys())}")
            return self.__class__(self._c[key])

    def __setattr__(self, key: str, value: Any):
        if not self._instance:
            return super().__setattr__(key, value)
        if hasattr(self._c, key):
            print(f"[ERROR] Invalid key: {key}")
            sys.exit(-1)
        else:
            self._c[key] = value

    def __repr__(self, depth: Optional[int] = 0):
        s = "====== Configurations ======\n" if not depth else ""
        for k, v in self._c.items():
            s += "  " * depth
            if isinstance(v, MutableMapping):
                s += f"{k}:\n{self.__class__(v).__repr__(depth + 1)}"
            else:
                s += f"{k}:  {reprlib.repr(v)}\n"
        s += "===========================\n" if not depth else ""
        return s


def get_config():
    conf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GNetDet.yaml")
    _C = _parse_yaml(conf_file)
    _assert_with_msg_or_exit(
        _C.CHIP in VALID_CHIPS, f"[ERROR] Wrong Chip: `{_C.CHIP}`! Only these chips are supported: {VALID_CHIPS}"
    )

    parser = argparse.ArgumentParser(description=f"Run Object Detection on {_C.CHIP} using {_C.MODEL.NAME}.")
    parser.add_argument("type",
                        type=str,
                        help="Type of detection (eg. `camara`, `image` or `video`).")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="./out.model",
                        help="Model path of *.model. Default is `./out.model`")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        default="",  # Leave empty here to force that users must specify this arg explicitly.
                        help="Path of input. REQUIRED if `type == image` or `type == video`. The input path could "
                             "be a specific image/video file path that are going to be detected, or a valid directory "
                             "path containing image(s)/video(s) to be detected.")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default="",
                        help="Optional, path of output. If `type == image` or `type == video`. the output path could "
                             "be a specific image/video file path to save detection results, or a valid directory "
                             "path in which all detected results with the same name as input files are to save. "
                             "If `output` is not given, then the detection results will just show up immediately.")
    parser.add_argument("--input-size",
                        type=int,
                        default=_C.MODEL.INPUT.SIZE,
                        help="Size of input image, must be either 448 or 224. If given, this value will override "
                             "MODEL.INPUT.SIZE in yaml file.")
    parser.add_argument("--input-format",
                        type=int,
                        default=_C.MODEL.INPUT.FORMAT,
                        help="Input image format, must be either 0 (for `BGR`) or 1 (for `YUV`). If given, this value "
                             "will override MODEL.INPUT.FORMAT in yaml file.")
    parser.add_argument("--conf-thresh",
                        type=float,
                        default=_C.OPTS.CONF_THRESH,
                        help="Threshold of confidence. A valid detection is that the output confidence is larger "
                             "than `conf_thresh`. If given, this value will override OPTS.CONF_THRESH in yaml file.")
    parser.add_argument("--prob-thresh",
                        type=float,
                        default=_C.OPTS.PROB_THRESH,
                        help="Threshold of probability. The probability (final confidence) of a output object is "
                             "its confidence times the maximum confidence among all objects in a single image. "
                             "Thus a valid detection is that the output probability is larger then `prob_thresh`. "
                             "If given, this value will override OPTS.PROB_THRESH in yaml file.")
    parser.add_argument("--nms-thresh",
                        type=float,
                        default=_C.OPTS.NMS_THRESH,
                        help="IOU threshold in NMS. A valid detection is that the bounding box parameterized by "
                             "(xmin, xmax, ymin, ymax) coordinates has less iou value than nms_thresh. If given, "
                             "this value will override OPTS.NMS_THRESH in yaml file.")
    parser.add_argument("-f",
                        "--fancy",
                        action="store_true",
                        default=False,
                        help="If present, drawing fancier bounding boxes.")

    args = parser.parse_args()

    # Check detection type
    _assert_with_msg_or_exit(
        args.type in VALID_DET_TYPE, f"[ERROR] Invalid type: {args.type}! Must be one of {VALID_DET_TYPE}."
    )

    # Check model
    _assert_with_msg_or_exit(
        os.path.exists(args.model), f"[ERROR] Model does not exist: {args.model}"
    )

    # Check paths
    if args.type != "camara":

        # Check input path
        _assert_with_msg_or_exit(
            os.path.exists(args.input), f"[ERROR] Input source does not exist: {args.input}"
        )

        # Input path must be file or directory
        if os.path.isfile(args.input):
            is_image = _check_path_is_valid_type_of(VALID_IMG_EXT, args.input)
            is_video = _check_path_is_valid_type_of(VALID_VID_EXT, args.input)
            _assert_with_msg_or_exit(
                (is_image and args.type == "image") or (is_video and args.type == "video"),
                f"[ERROR] Current type: `{args.type}` mismatch with input file: `{args.input}`"
            )
            input_paths = [args.input]
        else:
            _assert_with_msg_or_exit(
                os.path.isdir(args.input), f"[ERROR] Input directory path does not exists: {args.input}"
            )

            if args.type == "image":
                input_paths = _get_files(args.input, lambda p: _check_path_is_valid_type_of(VALID_IMG_EXT, p), recursive=False)
            else:
                input_paths = _get_files(args.input, lambda p: _check_path_is_valid_type_of(VALID_VID_EXT, p), recursive=False)
            _assert_with_msg_or_exit(
                bool(input_paths), f"[ERROR] No {args.type} under directory: {args.input}"
            )

        _assert_with_msg_or_exit(
            all(map(lambda p: ',' not in p, input_paths)),
            f"Path contain `,` which is not allowed in input path."
        )
        args.input = ','.join(input_paths)

        # Check output path
        if args.output:
            # output is file path
            if os.path.splitext(args.output)[1]:
                is_image = _check_path_is_valid_type_of(VALID_IMG_EXT, args.output)
                is_video = _check_path_is_valid_type_of(VALID_VID_EXT, args.output)
                _assert_with_msg_or_exit(
                    (is_image and args.type == "image") or (is_video and args.type == "video"),
                    f"[ERROR] Current type: `{args.type}` mismatch with output file: `{args.output}`"
                )
                output_paths = [args.output]

            # output is directory path
            else:
                os.makedirs(args.output, exist_ok=True)
                output_paths = []
                for path in input_paths:
                    output_paths.append(os.path.join(args.output, f"out_{os.path.basename(path)}"))

            _assert_with_msg_or_exit(
                all(map(lambda p: ',' not in p, output_paths)),
                f"[ERROR] Path contain `,` which is not allowed in output path."
            )
            args.output = ','.join(output_paths)

    # class_names match with num_classes
    _assert_with_msg_or_exit(
        len(_C.DATA.CLASS_NAMES) == _C.DATA.NUM_CLASSES,
        f"[ERROR] len(class_names) != num_classes, Check yaml file."
    )

    # Search LIBGITSDK
    libgtisdk = ""
    try:
        for path in [".", *filter(lambda p: bool(p), os.environ.get("LD_LIBRARY_PATH", "").split(':'))]:
            lib = _get_files(path, lambda p: os.path.basename(p) == "libGTILibrary.so")
            if lib:
                libgtisdk = os.path.abspath(lib[0])
                break
    except Exception as e:
        print(f"[WARNING] Some errors occur when loading files from : {path}")
        print(f"[WARNING] {e}")
    finally:
        _assert_with_msg_or_exit(bool(libgtisdk), f"[ERROR] `libGTILibrary.so` does not found!")

    # Set new configurations
    _C.LIBGTISDK = libgtisdk
    _C.TYPE = args.type
    _C.INPUT_PATH = args.input
    _C.OUTPUT_PATH = args.output
    _C.FANCY = args.fancy

    # Override Config with args
    _C.MODEL.PATH = args.model
    _C.MODEL.INPUT.SIZE = args.input_size
    _C.MODEL.INPUT.FORMAT = args.input_format
    _C.OPTS.CONF_THRESH = args.conf_thresh
    _C.OPTS.PROB_THRESH = args.prob_thresh
    _C.OPTS.NMS_THRESH = args.nms_thresh
    _C.OPTS.COLOR = [[0, 0, 0], [0, 225, 225], [225, 0, 0], [0, 225, 0], [0, 0, 225],
                     [225, 0, 225], [225, 225, 225], [64, 0, 0], [192, 0, 0],
                     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                     [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    return _C
