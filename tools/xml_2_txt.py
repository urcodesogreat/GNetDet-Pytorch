#!/usr/bin/env python3
import os
import sys
import random
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import get_config



# MAP_CLASS_NAMES = {
#     "bus" : "Vehicle",
#     "car" : "Vehicle",
#     "truck" : "Vehicle",
#     "person" : "Person",
# }
MAP_CLASS_NAMES = {}


def parse_rec(filename,  filter_small=False):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)

    size = tree.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    fx = cfg.image_size / width
    fy = cfg.image_size / height

    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}

        # Skip difficult and crowd
        difficult = obj.find('difficult')
        iscrowd = obj.find("iscrowd")
        if difficult is not None and difficult.text == '1':
            continue
        if iscrowd is not None and iscrowd.text == '1':
            continue

        # If MAP_CLASS_NAMES has key
        if MAP_CLASS_NAMES:
            name = MAP_CLASS_NAMES.get(obj.find('name').text, None)
        # or use original VOC_CLASSES names
        else:
            name = obj.find('name').text
        assert name in cfg.VOC_CLASSES, f"Wrong class name: {name}"
        obj_struct['name'] = name

        # Extract bbox
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))]

        # Skip small objects
        resized_width = fx * (obj_struct['bbox'][2] - obj_struct['bbox'][0])
        resized_height = fy * (obj_struct['bbox'][3] - obj_struct['bbox'][1])
        if filter_small and resized_width < 14. and resized_height < 14.:
            continue

        objects.append(obj_struct)

    return objects


def parse_xml_to_txt(load_factor: float = 1.0, split_by_name: bool = False):
    xml_dir = cfg.label_dir_path
    img_dir = cfg.image_dir_path
    xml_files = [fname.strip() for fname in os.listdir(xml_dir)]
    random.shuffle(xml_files)
    xml_files = xml_files[:int(load_factor*len(xml_files))]
    img_files = [fname.replace(".xml", ".jpg") for fname in xml_files]

    # Check image existence
    all_img_files = set([fname.strip() for fname in os.listdir(img_dir)])
    for img_file in img_files:
        assert (
            img_file in all_img_files
        ), f"Image not exist: {img_file}. Dataset corrupted!"
    del all_img_files

    # File handler for parsed data meta txt
    f_train_txt = open(cfg.train_txt_path, 'w')
    f_test_txt = open(cfg.valid_txt_path, 'w')

    count = 0
    for xml_file, img_file in tqdm(zip(xml_files, img_files), total=len(xml_files), desc=f"Write data TXT"):
        results = parse_rec(os.path.join(xml_dir, xml_file))

        if not results:
            # print("SKIP empty results:", xml_file)
            continue

        if split_by_name:
            # Use fname to split train-test
            if fnmatch(xml_file, "*train*"):
                f = f_train_txt
            else:
                f = f_test_txt
        else:
            # Use num images to split
            if count < int(0.85 * len(xml_files)):
                f = f_train_txt
            else:
                f = f_test_txt

        f.write(img_file)
        for result in results:
            if len(cfg.VOC_CLASSES) == 1:
                class_name = cfg.VOC_CLASSES[0]
            else:
                class_name = result["name"]
            cls_idx = cfg.VOC_CLASSES.index(class_name)
            bbox = result['bbox']
            f.write(' ' + str(bbox[0]) + ' ' + str(bbox[1]) +
                    ' ' + str(bbox[2]) + ' ' + str(bbox[3]) +
                    ' ' + str(cls_idx))

        f.write("\n")
        count += 1

    f_train_txt.flush()
    f_train_txt.close()
    f_test_txt.flush()
    f_test_txt.close()


if __name__ == '__main__':
    factor = 1.0 if len(sys.argv) == 1 else float(sys.argv[1])
    print("Load factor: ", factor)

    cfg = get_config()
    parse_xml_to_txt(factor)
