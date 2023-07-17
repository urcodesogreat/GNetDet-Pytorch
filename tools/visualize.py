import os
import cv2
import sys
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnetmdk.utils.visualize import visualize_detection
from gnetmdk.dataset.stats import calc_object_class_histogram
from gnetmdk.utils.xml import xml_to_dict, write_xml
from DSCollection.utils.common import list_files
from DSCollection.extract.parser import create_label_txt_from_voc
from configs import get_config


def get_bboxes_from_xml(xml_path: str, map_cls2id: dict):
    """Extract bounding boxes from a xml file."""
    # Get labels from xml
    bboxes = []
    annotation = xml_to_dict(xml_path)["annotation"]
    objects = annotation.get("object", [])
    for obj in objects:
        xmin = float(obj["bndbox"]["xmin"])
        ymin = float(obj["bndbox"]["ymin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymax = float(obj["bndbox"]["ymax"])
        try:
            clsid = float(map_cls2id[obj["name"]])
        except KeyError as e:
            msg = str(e)
            msg += f", map_cls2id keys: `{list(map_cls2id.keys())}`, xml-path: {xml_path}"
            raise KeyError(msg)

        bboxes.append([xmin, ymin, xmax, ymax, clsid])
    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes


def visualize_image_with_xml(image_path: str, xml_path: str, map_cls2id: dict, no_label=False):
    """Visualize image and all bboxes."""
    assert os.path.exists(image_path), f"Wrong path: {image_path}"
    assert os.path.exists(xml_path), f"Wrong path: {xml_path}"
    bboxes = get_bboxes_from_xml(xml_path, map_cls2id)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    map_id2cls = None if no_label else {v: k for k, v in map_cls2id.items()}
    visualize_detection(img, bboxes, map_id_to_cls=map_id2cls)


def visualize_voc(directory: str, map_cls2id: dict, vis_num: int = 10, no_label=False):
    """Visualize VOC-like dataset."""
    assert os.path.exists(directory), f"Wrong path: {directory}"
    anns_dir = os.path.join(directory, "Annotations")
    imgs_dir = os.path.join(directory, "JPEGImages")

    xml_fnames = list(list_files(anns_dir, abspath=False, recursive=False, filter_pred=lambda p: p.endswith(".xml")))
    xml_fnames = random.sample(xml_fnames, vis_num)
    for xml_fname in xml_fnames:
        xml_file = os.path.join(anns_dir, xml_fname)
        img_file = os.path.join(imgs_dir, xml_fname.replace(".xml", ".jpg"))
        visualize_image_with_xml(img_file, xml_file, map_cls2id, no_label)


def visualize_meta(meta_txt: str, img_dir: str = None, vis_num: int = 10, map_id2cls: dict = None):
    """
    Visualize dataset from meta txt which is created by `xml_2_txt.py`.
    """
    assert os.path.exists(meta_txt), f"Wrong path: {meta_txt}"
    lines = open(meta_txt, 'r').readlines()
    random.shuffle(lines)
    lines = random.sample(lines, vis_num)

    if img_dir is None:
        dataset_name = os.path.basename(meta_txt).split('_')[0]
        img_dir = os.path.join(os.path.dirname(os.path.dirname(meta_txt)), dataset_name, "JPEGImages")

    for line in lines:
        splited = line.strip().split()
        img_file = splited[0]
        img_file = os.path.join(img_dir, img_file)
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        assert img is not None, f"Wrong path: {img_file}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_bboxes = (len(splited) - 1) // 5
        bboxes = []
        for i in range(num_bboxes):
            xmin = float(splited[1 + 5 * i])
            ymin = float(splited[2 + 5 * i])
            xmax = float(splited[3 + 5 * i])
            ymax = float(splited[4 + 5 * i])
            clsid = float(splited[5 + 5 * i])
            bboxes.append([xmin, ymin, xmax, ymax, clsid])
        bboxes = np.array(bboxes, dtype=np.float32)

        visualize_detection(img, bboxes, map_id_to_cls=map_id2cls)


def visualize_voc_class_histogram(directory: str, map_cls2id: dict = None, class_names: list = None):
    """Plot histogram of voc dataset."""
    assert os.path.exists(directory), f"Wrong path: {directory}"
    anns_dir = os.path.join(directory, "Annotations")
    xml_files = list(list_files(anns_dir, filter_pred=lambda p: p.endswith(".xml")))
    ann_dicts = [xml_to_dict(file) for file in xml_files]

    if not class_names:
        label_txt = os.path.join(directory, "label.txt")
        if not os.path.exists(label_txt):
            n = None if not map_cls2id else len(map_id2cls)
            create_label_txt_from_voc(directory, n)
        class_names = open(label_txt, 'r').readlines()
        class_names = [class_name.strip() for class_name in class_names]

    if not map_cls2id:
        map_cls2id = {name: idx for idx, name in enumerate(class_names)}

    counts = calc_object_class_histogram(ann_dicts, class_names, map_cls2id)
    cls_ids = range(len(class_names) + 1)

    plt.bar(cls_ids, counts)
    plt.xticks(cls_ids, [*class_names, "other"], rotation=25)
    plt.title(os.path.basename(directory))
    plt.show()


# def visualize_model_preds(model: nn.Module, image: np.ndarray):
#     outs = model(image)


if __name__ == '__main__':
    # map_cls2id = {"helmet": 0, "non-helmet": 1, "reflective": 2, "non-reflective": 3}
    # map_cls2id = {"hat": 0, "person": 1, "reflective_clothes": 2, "other_clothes": 3}
    map_cls2id = {"fire": 0}
    map_cls2id = {"cigarette": 0, "person": 1, "head": 2, "phone": 3}
    map_cls2id = {"person": 0, "smoking-calling": 1}
    map_cls2id = {"mask": 0, "non-mask": 1, "non-uniform": 2, "uniform": 3}
    map_cls2id = {"face": 0, "face_mask": 1}
    map_cls2id = {"head": 0}

    root = r"/data2/Datasets/Face/Mask/p0"
    root = r"/data2/Datasets/Face/Mask/p1"
    root = r"/data2/Datasets/Face/Mask/p2"
    root = r"/data2/Datasets/Face/Mask/p3"
    root = r"/data2/Datasets/Face/Mask/face_mask"
    root = r"/media/sparkai/DATA2/Datasets/Head/VOCdevkit/VOC2007"

    visualize_voc(root, map_cls2id, no_label=True, vis_num=5)

    # meta_txt = r"/home/sparkai/PycharmProjects/GNetDet_MDK_Pytorch_2021_09_02/data/meta/SmokeV1_train.txt"
    # visualize_meta(meta_txt)
    # xml = r"/home/sparkai/PycharmProjects/GNetDet_MDK_Pytorch_2021_09_02/data/ReflectiveCloth/Annotations/ReflectiveCloth-train_0040.xml"
    # img = r"/home/sparkai/PycharmProjects/GNetDet_MDK_Pytorch_2021_09_02/data/ReflectiveCloth/JPEGImages/ReflectiveCloth-train_0040.jpg"
    # visualize_image_with_xml(img, xml, map_cls2id)

    visualize_voc_class_histogram(root)
