import os

from detectron2.data.datasets import register_coco_instances


_PREDEFINED_MY_DATASETS = {}
_PREDEFINED_MY_DATASETS["custom_coco"] = {
    "custom_coco_train": ("coco2017/train2017", "coco2017/annotations/instances_train2017.json"),
    "custom_coco_validation": ("coco2017/val2017", "coco2017/annotations/instances_val2017.json"),
}

_PREDEFINED_MY_DATASETS["scene_text"] = {
    "icdar2015_train": ("icdar2015_mmocr/imgs", "icdar2015_mmocr/instances_training.json"),
    "icdar2015_test": ("icdar2015_mmocr/imgs", "icdar2015_mmocr/instances_test.json"),
}

_PREDEFINED_MY_DATASETS["chinese_invoice"] = {
    "chinese_invoice_1118_train_all": (
        "chinese_invoice_20221118_all/imgs", "chinese_invoice_20221118_all/instances_training.json"),
    "chinese_invoice_1118_validation": (
        "chinese_invoice_20221118_all/imgs", "chinese_invoice_20221118_all/instances_validation.json"),
}

_PREDEFINED_MY_DATASETS["multilingual"] = {
    "training_530_v2_total": ("training_530_v2/total/imgs/", "training_530_v2/total/total.json"),
    "validation_530_v2_total": ("validation_530_v2/total/imgs/", "validation_530_v2/total/total.json"),
}

def register_custom_coco_format_dataset(root='/home/ma-user/work/datasets'):
    for dataset_name, splits_per_dataset in _PREDEFINED_MY_DATASETS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )