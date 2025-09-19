import os
import numpy as np
from PIL import Image
import pandas as pd
import csv
import cv2
import argparse
import torch
from mmengine.structures import PixelData
# from mmseg.evaluation import IoUMetric
# from mmseg.structures import SegDataSample
from eval_util import IoUMetric, SegDataSample

def recursive_glob(rootdir='.', suffix=''): # rootdir 是根目录的路径，suffix 是要搜索的文件后缀。
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)    # 这是一个列表推导式，它将匹配后缀的文件的完整路径组成的列表返回。列表推导式会遍历文件系统中的文件，然后检查文件名是否以指定的后缀结尾。
            for looproot, _, filenames in os.walk(rootdir)# 这是一个嵌套的循环，使用 os.walk 函数来遍历指定根目录 rootdir 及其子目录中的文件。os.walk 返回一个生成器，生成三元组 (当前目录, 子目录列表, 文件列表)。在这里，我们只关心当前目录和文件列表，因此使用 _ 来表示不关心的子目录列表。
            for filename in filenames if filename.endswith(suffix)]# 这是列表推导式的内部循环，遍历当前目录中的文件列表 filenames，然后检查每个文件名是否以指定的后缀 suffix 结尾。如果是，就将满足条件的文件名添加到最终的返回列表中。

METAINFO = dict(
        classes=(
                'agricultural land', 'forest', 'water', 'meadow', 'road', 'dense residential area',
                'sparse residential area', 'public area', 'construction area'
                ),

        palette=[
                    [250, 250,  75], 
                    [  0, 150,   0], 
                    [  0, 150, 200], 
                    [200, 200,   0], 
                    [250, 150, 150], 
                    [200,   0,   0],
                    [250,   0, 100], 
                    [250, 200,   0], 
                    [200, 200, 200],
                 ]
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str, default='../../data/huawei/huawei_project_dataset/ori/mask/test')
    parser.add_argument("--pred_path", type=str, default='show_dirs/test0417/geofed_testbig_1024')
    parser.add_argument("--suffix", type=str, default='.png')
    parser.add_argument("--sizeImg", type=int, default=12500)
    args = parser.parse_args()
   
    return args

def main():
    args = parse_args()

    conb_path = args.pred_path
    output_path = args.pred_path + '.csv'
    
    namelist = recursive_glob(rootdir=args.label_path, suffix=args.suffix)
    namelist = [os.path.basename(file) for file in namelist]
    namelist = [os.path.splitext(file)[0] for file in namelist]
    
    H = 12500
    W = 12500

    # eval
    namelist = recursive_glob(rootdir=args.label_path, suffix='.png')
    test_datas = []

    for ll in range(len(namelist)):
        # pred_name = namelist[ll][3:].replace(args.label_path, conb_path)
        pred_name = os.path.join(conb_path, os.path.basename(namelist[ll]))
        label_name = namelist[ll]
        
        label_image = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        # prediction_image = cv2.imread(pred_name)
        prediction_image = np.array(Image.open(pred_name))
        
        test_data = SegDataSample()
        pred = PixelData()
        gt = PixelData()
        pred.data = torch.from_numpy(prediction_image)
        gt.data = torch.from_numpy(label_image)
        test_data.pred_sem_seg = pred
        test_data.gt_sem_seg = gt

        test_datas.append(test_data)

    # test_met = IoUMetric(iou_metrics = ['mIoU'])
    test_met = IoUMetric(iou_metrics = ['mIoU'])
    test_met._dataset_meta = METAINFO
    test_met.process(None, test_datas)
    final_met, ret_metrics_class = test_met.compute_metrics(test_met.results)
    # print('mIoU:', final_met['mIoU'], ',' , 'mFscore:', final_met['mFscore'])
    print('mIoU:', final_met['mIoU'])

    df = pd.DataFrame(ret_metrics_class)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    f = open(output_path, "a", encoding="utf-8", newline="")

    csv_writer = csv.writer(f)
    csv_writer.writerow(['Mean', final_met['mIoU']])
    f.close()
    
if __name__ == '__main__':
    main()