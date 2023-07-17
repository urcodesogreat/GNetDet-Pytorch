import numpy as np


def calc_object_class_histogram(ann_dicts, class_names, map_cls2id):
    """
    Args:
        ann_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    # classes + blank
    num_classes = len(class_names) + 1
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for ann_dict in ann_dicts:
        annos = ann_dict["annotation"]
        try:
            classes = np.asarray(
                [map_cls2id[x["name"]] for x in annos["object"]], dtype=np.int
            )
        except KeyError:
            # ann_dict has no "object"
            classes = np.array([num_classes - 1], dtype=np.int)

        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]
    return histogram
