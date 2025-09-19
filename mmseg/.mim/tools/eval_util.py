# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmengine.registry import METRICS
import torch.distributed as dist

@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'gpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        i = 0
        for data_sample in data_samples:
            pred_label = data_sample.pred_sem_seg.data.squeeze()
            pred_label = pred_label - 1 
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample.gt_sem_seg.data.squeeze().to(
                    pred_label)
                label = label - 1
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
            print(i)
            i = i + 1
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                
                # output_mask = pred_label.cpu().numpy()

                pred_list = [None] * dist.get_world_size()
                dist.gather_object(pred_label.cpu().numpy(), pred_list if dist.get_rank() == 0 else None, dst=0)
                if dist.get_rank() == 0:
                    output_mask = np.concatenate([x for x in pred_list], 0)

                    # The index range of official ADE20k dataset is from 0 to 150.
                    # But the index range of output is from 0 to 149.
                    # That is because we set reduce_zero_label=True.
                    if data_sample.get('reduce_zero_label', False):
                        output_mask = output_mask + 1
                    output = Image.fromarray(output_mask.astype(np.uint8))
                    output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics, ret_metrics_class

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                # ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                # ret_metrics['Precision'] = precision
                # ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, PixelData

class SegDataSample(BaseDataElement):
    """A data structure interface of MMSegmentation. They are used as
    interfaces between different components.

    The attributes in ``SegDataSample`` are divided into several parts:

        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
        - ``seg_logits``(PixelData): Predicted logits of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import PixelData
         >>> from mmseg.structures import SegDataSample

         >>> data_sample = SegDataSample()
         >>> img_meta = dict(img_shape=(4, 4, 3),
         ...                 pad_shape=(4, 4, 3))
         >>> gt_segmentations = PixelData(metainfo=img_meta)
         >>> gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
         >>> data_sample.gt_sem_seg = gt_segmentations
         >>> assert 'img_shape' in data_sample.gt_sem_seg.metainfo_keys()
         >>> data_sample.gt_sem_seg.shape
         (4, 4)
         >>> print(data_sample)
        <SegDataSample(

            META INFORMATION

            DATA FIELDS
            gt_sem_seg: <PixelData(

                    META INFORMATION
                    img_shape: (4, 4, 3)
                    pad_shape: (4, 4, 3)

                    DATA FIELDS
                    data: tensor([[[1, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 0, 1]]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = SegDataSample()
        >>> gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> data_sample.gt_sem_seg = gt_sem_seg
        >>> assert 'gt_sem_seg' in data_sample
        >>> assert 'sem_seg' in data_sample.gt_sem_seg
    """

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> PixelData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_seg_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits


# import os
# import cv2
# import numpy as np
# import torch
# from mmengine.structures import PixelData
# # from mmseg.evaluation import IoUMetric
# # from mmseg.structures import SegDataSample

# # 读取标签图像和预测图像
# def recursive_glob(rootdir='.', suffix=''): # rootdir 是根目录的路径，suffix 是要搜索的文件后缀。
#     """Performs recursive glob with given suffix and rootdir
#         :param rootdir is the root directory
#         :param suffix is the suffix to be searched
#     """
#     return [os.path.join(looproot, filename)    # 这是一个列表推导式，它将匹配后缀的文件的完整路径组成的列表返回。列表推导式会遍历文件系统中的文件，然后检查文件名是否以指定的后缀结尾。
#             for looproot, _, filenames in os.walk(rootdir)# 这是一个嵌套的循环，使用 os.walk 函数来遍历指定根目录 rootdir 及其子目录中的文件。os.walk 返回一个生成器，生成三元组 (当前目录, 子目录列表, 文件列表)。在这里，我们只关心当前目录和文件列表，因此使用 _ 来表示不关心的子目录列表。
#             for filename in filenames if filename.endswith(suffix)]# 这是列表推导式的内部循环，遍历当前目录中的文件列表 filenames，然后检查每个文件名是否以指定的后缀 suffix 结尾。如果是，就将满足条件的文件名添加到最终的返回列表中。

# METAINFO = dict(
#         classes=('industrial area','paddy field','irrigated field','dry cropland','garden land',
#                 'arbor forest','shrub forest','park land','natural meadow','artificial meadow',
#                 'river','urban residential','lake','pond','fish pond','snow','bareland',
#                 'rural residential','stadium','square','road','overpass','railway station','airport'),

#         palette=[
#                     [200,   0,   0], # industrial area
#                     [0, 200,   0], # paddy field
#                     [150, 250,   0], # irrigated field
#                     [150, 200, 150], # dry cropland
#                     [200,   0, 200], # garden land
#                     [150,   0, 250], # arbor forest
#                     [150, 150, 250], # shrub forest
#                     [200, 150, 200], # park land
#                     [250, 200,   0], # natural meadow
#                     [200, 200,   0], # artificial meadow
#                     [0,   0, 200], # river
#                     [250,   0, 150], # urban residential
#                     [0, 150, 200], # lake
#                     [0, 200, 250], # pond
#                     [150, 200, 250], # fish pond
#                     [250, 250, 250], # snow
#                     [200, 200, 200], # bareland
#                     [200, 150, 150], # rural residential
#                     [250, 200, 150], # stadium
#                     [150, 150,   0], # square
#                     [250, 150, 150], # road
#                     [250, 150,   0], # overpass
#                     [250, 200, 250], # railway station
#                     [200, 150,   0] # airport
#                     # [0,   0,   0] # unlabeled
#                  ]
#         )

# conb_path = 'conbig_test'
# label_path = 'label_ori'
# output_path = 'result.csv'

# namelist = recursive_glob(rootdir=conb_path, suffix='.png')

# test_datas = []

# for ll in range(len(namelist)):
#     pred_name = namelist[ll]
#     label_name = pred_name.replace(conb_path, label_path).replace('.png', '_24label.png')
    
#     label_image = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
#     prediction_image = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)

#     test_data = SegDataSample()
#     pred = PixelData()
#     gt = PixelData()
#     pred.data = torch.from_numpy(prediction_image)
#     gt.data = torch.from_numpy(label_image)
#     test_data.pred_sem_seg = pred
#     test_data.gt_sem_seg = gt

#     test_datas.append(test_data)

# # test_met = IoUMetric(iou_metrics = ['mIoU'])
# test_met = IoUMetric(iou_metrics = ['mIoU'])
# test_met._dataset_meta = METAINFO
# test_met.process(None, test_datas)
# final_met, ret_metrics_class = test_met.compute_metrics(test_met.results)
# # print('mIoU:', final_met['mIoU'], ',' , 'mFscore:', final_met['mFscore'])

# import pandas as pd
# df = pd.DataFrame(ret_metrics_class)
# df.to_csv(output_path, index=False, encoding="utf-8-sig")

# f = open(output_path, "a", encoding="utf-8", newline="")
# import csv
# csv_writer = csv.writer(f)
# csv_writer.writerow(['Mean', final_met['mIoU']])
# f.close()