# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import csv
import math
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox

def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))
def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))
def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys

def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

@METRICS.register_module()
class DOTAMetric(BaseMetric):
    """DOTA evaluation metric.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the DOTA
    large images, which can be found at: ``tools/data/dota/split``.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Only support
            'mAP' now. If is list, the first setting in the list will
             be used to evaluate metric.
        predict_box_type (str): Box type of model results. If the QuadriBoxes
            is used, you need to specify 'qbox'. Defaults to 'rbox'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format. Defaults to False.
        outfile_prefix (str, optional): The prefix of json/zip files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        merge_patches (bool): Generate the full image's results by merging
            patches' results.
        iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
            patches. Defaults to 0.1.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'. Defaults to '11points'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'dota'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        # voc evaluation metrics
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.predict_box_type = predict_box_type

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr

        self.use_07_metric = True if eval_mode == '11points' else False

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.

        You can submit it at:
        https://captain-whu.github.io/DOTA/evaluation.html

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5],
                                                  cls_dets[:,
                                                           -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    else:
                        raise NotImplementedError
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'The outfile_prefix should be a non-exist path, '
                             f'but {outfile_prefix} is existing. '
                             f'Please delete it firstly.')
        os.makedirs(outfile_prefix)

        files = [
            osp.join(outfile_prefix, 'Task1_' + cls + '.txt')
            for cls in self.dataset_meta['classes']
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                else:
                    raise NotImplementedError
                for qbox, score in zip(qboxes, scores):
                    txt_element = [img_id, str(round(float(score), 2))
                                   ] + [f'{p:.2f}' for p in qbox]
                    f.writelines(' '.join(txt_element) + '\n')
        for f in file_objs:
            f.close()



            #################################################################################
        CLASSES = ('Nimitz Aircraft Carrier', 'Barracks Ship', 'Container Ship', 'Fishing Vessel',
                   'Henry J. Kaiser-class replenishment oiler', 'Other Warship', 'Yacht',
                   'Freedom-class littoral combat ship', 'Arleigh Burke-class Destroyer',
                   'Lewis and Clark-class dry cargo ship', 'Towing vessel', 'unknown', 'Powhatan-class tugboat',
                   'Barge', '055-destroyer', '052D-destroyer', 'USNS Bob Hope', 'USNS Montford Point', 'Bunker',
                   'Ticonderoga-class cruiser', 'Oliver Hazard Perry-class frigate',
                   'Sacramento-class fast combat support ship', 'Submarine', 'Emory S. Land-class submarine tender',
                   'Hatakaze-class destroyer', 'Murasame-class destroyer', 'Whidbey Island-class dock landing ship',
                   'Hiuchi-class auxiliary multi-purpose support ship', 'USNS Spearhead',
                   'Hyuga-class helicopter destroyer', 'Akizuki-class destroyer', 'Bulk carrier',
                   'Kongo-class destroyer', 'Northampton-class tug', 'Sand Carrier', 'Iowa-class battle ship',
                   'Independence-class littoral combat ship', 'Tarawa-class amphibious assault ship',
                   'Cyclone-class patrol ship', 'Wasp-class amphibious assault ship', '074-landing ship',
                   '056-corvette', '721-transport boat', '037II-missile boat', 'Traffic boat', '037-submarine chaser',
                   'unknown auxiliary ship', '072III-landing ship', '636-hydrographic survey ship', '272-icebreaker',
                   '529-Minesweeper', '053H2G-frigate', '909A-experimental ship', '909-experimental ship',
                   '037-hospital ship', 'Tuzhong Class Salvage Tug', '022-missile boat', '051-destroyer',
                   '054A-frigate', '082II-Minesweeper', '053H1G-frigate', 'Tank ship', 'Hatsuyuki-class destroyer',
                   'Sugashima-class minesweepers', 'YG-203 class yard gasoline oiler',
                   'Hayabusa-class guided-missile patrol boats', 'JS Chihaya', 'Kurobe-class training support ship',
                   'Abukuma-class destroyer escort', 'Uwajima-class minesweepers', 'Osumi-class landing ship',
                   'Hibiki-class ocean surveillance ships', 'JMSDF LCU-2001 class utility landing crafts',
                   'Asagiri-class Destroyer', 'Uraga-class Minesweeper Tender', 'Tenryu-class training support ship',
                   'YW-17 Class Yard Water', 'Izumo-class helicopter destroyer', 'Towada-class replenishment oilers',
                   'Takanami-class destroyer', 'YO-25 class yard oiler', '891A-training ship', '053H3-frigate',
                   '922A-Salvage lifeboat', '680-training ship', '679-training ship', '072A-landing ship',
                   '072II-landing ship', 'Mashu-class replenishment oilers', '903A-replenishment ship', '815A-spy ship',
                   '901-fast combat support ship', 'Xu Xiake barracks ship',
                   'San Antonio-class amphibious transport dock', '908-replenishment ship', '052B-destroyer',
                   '904-general stores issue ship', '051B-destroyer', '925-Ocean salvage lifeboat',
                   '904B-general stores issue ship', '625C-Oceanographic Survey Ship', '071-amphibious transport dock',
                   '052C-destroyer', '635-hydrographic Survey Ship', '926-submarine support ship', '917-lifeboat',
                   'Mercy-class hospital ship', 'Lewis B. Puller-class expeditionary mobile base ship',
                   'Avenger-class mine countermeasures ship', 'Zumwalt-class destroyer', '920-hospital ship',
                   '052-destroyer', '054-frigate', '051C-destroyer', '903-replenishment ship', '073-landing ship',
                   '074A-landing ship', 'North Transfer 990', '001-aircraft carrier', '905-replenishment ship',
                   'Hatsushima-class minesweeper', 'Forrestal-class Aircraft Carrier',
                   'Kitty Hawk class aircraft carrier', 'Blue Ridge class command ship', '081-Minesweeper',
                   '648-submarine repair ship', '639A-Hydroacoustic measuring ship', 'JS Kurihama', 'JS Suma',
                   'Futami-class hydro-graphic survey ships', 'Yaeyama-class minesweeper', '815-spy ship',
                   'Sovremenny-class destroyer')

        with open('./results4.csv','w',newline="") as frr:
            writer = csv.writer(frr)
            txt_element = ["ImageID", "LabelName", 'X1', 'Y1', 'X2','Y2', 'X3', 'Y3','X4', 'Y4','Conf']
            writer.writerow(txt_element)
            #frr.writelines(' '.join(txt_element) + '\n')
            for img_id, dets_per_cls in zip(id_list, dets_list):
                print(img_id)
                for cls, dets in zip(CLASSES, dets_per_cls):
                    if dets.size == 0:
                        continue
                    bboxes = obb2poly_np_le90(dets)
                    for bbox in bboxes:
                        txt_element = [str(img_id)+'.bmp',cls
                                       ] + [f'{p:.2f}' for p in bbox[:-1]]+[str(bbox[-1])]
                        #frr.writelines(' '.join(txt_element) + '\n')
                        writer.writerow(txt_element)
            #########################################################################################



        target_name = osp.split(outfile_prefix)[-1]
        zip_path = osp.join(outfile_prefix, target_name + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return zip_path

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i].tolist()
                data['score'] = float(scores[i])
                data['category_id'] = int(label)
                bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        return result_files

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape(
                        (-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results
