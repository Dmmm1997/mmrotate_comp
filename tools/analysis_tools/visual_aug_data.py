# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS, TRANSFORMS
from mmdet.structures.bbox import BaseBoxes
from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar
from mmengine.runner.runner import Runner

from mmrotate.utils import register_all_modules
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default="./tmp_visual_dataset/",
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=True, action='store_true')
    parser.add_argument('--nums', type=int, default=10, help='the number of images to save'),
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    dataloader = Runner.build_dataloader(
                cfg.train_dataloader, seed=100, diff_rank_seed=False)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))
    ind = 0
    for batch in dataloader:
        for inputs,data_samples in zip(batch["inputs"],batch["data_samples"]):
            ind+=1
            if ind>args.nums:
                sys.exit(0)
            img = inputs.permute(1, 2, 0).numpy()
            data_sample = data_samples.numpy()
            gt_instances = data_sample.gt_instances
            img_path = osp.basename(data_samples.img_path)

            out_file = osp.join(
                args.output_dir,
                osp.basename(img_path)) if args.output_dir is not None else None

            img = img[..., [2, 1, 0]]  # bgr to rgb
            gt_bboxes = gt_instances.get('bboxes', None)
            if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
                gt_instances.bboxes = gt_bboxes.tensor
            gt_masks = gt_instances.get('masks', None)
            if gt_masks is not None:
                masks = mask2ndarray(gt_masks)
                gt_instances.masks = masks.astype(bool)
            data_sample.gt_instances = gt_instances

            visualizer.add_datasample(
                osp.basename(img_path),
                img,
                data_sample,
                show=not args.not_show,
                wait_time=args.show_interval,
                out_file=out_file)

            progress_bar.update()


if __name__ == '__main__':
    main()