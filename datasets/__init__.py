import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, cls_order, phase_idx, incremental, incremental_val, val_each_phase, balanced_ft=False):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args, cls_order, phase_idx, incremental, incremental_val, val_each_phase, balanced_ft)
    raise ValueError(f'dataset {args.dataset_file} not supported')
