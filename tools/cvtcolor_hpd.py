import matplotlib.pyplot as plt
import numpy as np
import torch
# import rasterio
import os
from PIL import Image
import argparse

def recursive_glob(rootdir='.', suffix=''): # rootdir 是根目录的路径，suffix 是要搜索的文件后缀。
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)    # 这是一个列表推导式，它将匹配后缀的文件的完整路径组成的列表返回。列表推导式会遍历文件系统中的文件，然后检查文件名是否以指定的后缀结尾。
            for looproot, _, filenames in os.walk(rootdir)# 这是一个嵌套的循环，使用 os.walk 函数来遍历指定根目录 rootdir 及其子目录中的文件。os.walk 返回一个生成器，生成三元组 (当前目录, 子目录列表, 文件列表)。在这里，我们只关心当前目录和文件列表，因此使用 _ 来表示不关心的子目录列表。
            for filename in filenames if filename.endswith(suffix)]# 这是列表推导式的内部循环，遍历当前目录中的文件列表 filenames，然后检查每个文件名是否以指定的后缀 suffix 结尾。如果是，就将满足条件的文件名添加到最终的返回列表中。

def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 9 + 1
    label_colours = get_hpd_labels()
    
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def get_hpd_labels():
    return np.array([
            [0,0,0],
            [250, 250,  75], 
            [  0, 150,   0], 
            [  0, 150, 200], 
            [200, 200,   0], 
            [250, 150, 150], 
            [200,   0,   0],
            [250,   0, 100], 
            [250, 200,   0], 
            [200, 200, 200],
                ])  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default=None)
    args = parser.parse_args()
   
    return args

def main():
    args = parse_args()

    filepath = args.pred_path + '_conb'
    newpath = args.pred_path + '_conb_col'
    os.makedirs(newpath, exist_ok=True)
    pathDir = recursive_glob(filepath, '.png')
    i = 0
    for filename in pathDir:
        img = Image.open(filename)
        img = np.array(img)
        col = decode_segmap(img).astype('uint8')
        newname = filename.replace(filepath, newpath)
        col = Image.fromarray(col)
        col.save(newname)
    
        i = i+1
        print(i)
        
if __name__ == '__main__':
    main()
