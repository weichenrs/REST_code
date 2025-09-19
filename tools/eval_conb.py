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
# METAINFO = dict(
#         classes=('river', 'lake',),

#         palette=[
#                     [0,   0, 200], # river
#                     [0, 150, 200], # lake
#                 ]
#         )
    
METAINFO = dict(
        classes=('industrial area','paddy field','irrigated field','dry cropland','garden land',
                'arbor forest','shrub forest','park land','natural meadow','artificial meadow',
                'river','urban residential','lake','pond','fish pond','snow','bareland',
                'rural residential','stadium','square','road','overpass','railway station','airport'),

        palette=[
                    [200,   0,   0], # industrial area
                    [0, 200,   0], # paddy field
                    [150, 250,   0], # irrigated field
                    [150, 200, 150], # dry cropland
                    [200,   0, 200], # garden land
                    [150,   0, 250], # arbor forest
                    [150, 150, 250], # shrub forest
                    [200, 150, 200], # park land
                    [250, 200,   0], # natural meadow
                    [200, 200,   0], # artificial meadow
                    [0,   0, 200], # river
                    [250,   0, 150], # urban residential
                    [0, 150, 200], # lake
                    [0, 200, 250], # pond
                    [150, 200, 250], # fish pond
                    [250, 250, 250], # snow
                    [200, 200, 200], # bareland
                    [200, 150, 150], # rural residential
                    [250, 200, 150], # stadium
                    [150, 150,   0], # square
                    [250, 150, 150], # road
                    [250, 150,   0], # overpass
                    [250, 200, 250], # railway station
                    [200, 150,   0] # airport
                    # [0,   0,   0] # unlabeled
                 ]
        )

# X_512 = [0,  477,  954, 1431, 1908, 2385, 2862, 3339, 3816, 4293, 4770, 5247, 5724, 6201, 6678, 6688] 
# Y_512 = [0,  483,  966, 1449, 1932, 2415, 2898, 3381, 3864, 4347, 4830, 5313, 5796, 6279, 6288]

# X_1024=[0, 882, 1764, 2646, 3528, 4410, 5292, 6174, 6176] #7200 W
# Y_1024=[0, 962, 1924, 2886, 3848, 4810, 5772, 5776] #6800 H

# X_2048=[0, 1973, 3946, 5152] #7200
# Y_2048=[0, 1840, 3680, 4752] #6800

# X_3072=[0, 2832, 4128] #7200
# Y_3072=[0, 2888, 3728] #6800

# X_4096=[0, 3104] 
# Y_4096=[0, 2704]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str, default='../data/Five-Billion-Pixels/ori/Annotation__index/test')
    parser.add_argument("--pred_path", type=str, default='mmsegmentation/show_dirs/test1127/vit_vit-b16_mln_upernet_2xb2-40k_fbp-4096x4096_ds.py/best_mIoU_iter_40000.pth')
    parser.add_argument("--suffix", type=str, default='.png')
    parser.add_argument("--sizeImg", type=int, default=4096)
    args = parser.parse_args()
   
    return args

def main():
    args = parse_args()

    conb_path = args.pred_path + '_conb'
    output_path = args.pred_path + '.csv'
    
    namelist = recursive_glob(rootdir=args.label_path, suffix=args.suffix)
    namelist = [os.path.basename(file) for file in namelist]
    namelist = [os.path.splitext(file)[0] for file in namelist]
    namelist = [name.split('_24label')[0] for name in namelist]
    
    H = 6800
    W = 7200
    nnr = np.uint8(np.floor(H/args.sizeImg))
    nnc = np.uint8(np.floor(W/args.sizeImg))
    overlapH = np.uint8(np.ceil(args.sizeImg - (H-args.sizeImg)/nnr))
    overlapW = np.uint8(np.ceil(args.sizeImg - (W-args.sizeImg)/nnc))
    disH = args.sizeImg - overlapH
    disW = args.sizeImg - overlapW
    r = np.uint8(np.ceil(H/disH))
    c = np.uint8(np.ceil(W/disW))
    if (c-1)*disW+overlapW >= W:
        c = c - 1
    if (r-1)*disH+overlapH >= H:
        r = r - 1

    for ll in range (len(namelist)):
        name = namelist[ll]
        big = np.zeros((6800,7200),np.uint8)
        
        for i in range(r):
            for j in range(c):
                tmpname = name + '_' + str(i+1) + '_' + str(j+1) + args.suffix#构建文件名如name_scale8_1024_1024
                newname = os.path.join(args.pred_path, tmpname) # 使用 os.path.join 构建一个新文件的完整路径，包含目录结构和文件名，最终形式如 'evalgidnew/2048/tmpname_color.png'
                output = np.array(Image.open(newname))  # 使用 PIL 库的 Image.open 打开图像文件 newname，然后将其转换为 NumPy 数组，结果存储在 output 变量中
                
                if i==r-1:
                    if j==c-1:
                        big[H-args.sizeImg:H, W-args.sizeImg:W] = output
                    else:
                        big[H-args.sizeImg:H, j*disW:(j+1)*disW+overlapW] = output
                else:
                    if j==c-1:
                        big[i*disH:(i+1)*disH+overlapH, W-args.sizeImg:W] = output
                    else:
                        big[i*disH:(i+1)*disH+overlapH, j*disW:(j+1)*disW+overlapW] = output
                        
        big = Image.fromarray(np.uint8(big))
        # 定义保存路径
        save_path = os.path.join(conb_path, name + args.suffix)

        # 检查目录是否存在，如果不存在则创建它
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存 big 图像
        big.save(save_path)
        print(ll)
    
    # eval
    namelist = recursive_glob(rootdir=conb_path, suffix='.png')
    test_datas = []

    for ll in range(len(namelist)):
        pred_name = namelist[ll]
        label_name = pred_name.replace(conb_path, args.label_path).replace('.png', '_24label.png')
        
        label_image = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        prediction_image = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)

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