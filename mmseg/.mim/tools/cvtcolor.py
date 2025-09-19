import matplotlib.pyplot as plt
import numpy as np
import torch
# import rasterio
import os
from PIL import Image

def recursive_glob(rootdir='.', suffix=''): # rootdir 是根目录的路径，suffix 是要搜索的文件后缀。
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)    # 这是一个列表推导式，它将匹配后缀的文件的完整路径组成的列表返回。列表推导式会遍历文件系统中的文件，然后检查文件名是否以指定的后缀结尾。
            for looproot, _, filenames in os.walk(rootdir)# 这是一个嵌套的循环，使用 os.walk 函数来遍历指定根目录 rootdir 及其子目录中的文件。os.walk 返回一个生成器，生成三元组 (当前目录, 子目录列表, 文件列表)。在这里，我们只关心当前目录和文件列表，因此使用 _ 来表示不关心的子目录列表。
            for filename in filenames if filename.endswith(suffix)]# 这是列表推导式的内部循环，遍历当前目录中的文件列表 filenames，然后检查每个文件名是否以指定的后缀 suffix 结尾。如果是，就将满足条件的文件名添加到最终的返回列表中。


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'kd':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'dfc2022':
        n_classes = 16
        label_colours = get_dfc2022_labels()
    elif dataset == 'potsdam':
        n_classes = 6
        label_colours = get_potsdam_labels()
    elif dataset == 'deepglobe':
        n_classes = 6
        label_colours = get_deepglobe_labels()
    elif dataset == 'fbp':
        n_classes = 24 + 1
        label_colours = get_fbp_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    # mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_hpd_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    # label_mask = label_mask.astype(int)
    return label_mask

def get_deepglobe_labels():
        return np.array([
            [0,255,255],  #urban land
            [255,255,0],  #agriculture land
            [255,0,255],    #rangeland/grass
            [0,255,0],   #forest land
            [0,0,255],   #water
            [255,255,255],  #barren land
                    ])  

def get_potsdam_labels():
        return np.array([
            [255,0,0],    #clutter
            [255,255,255],  #imprevious
            [255,255,0],  #car
            [0,255,0],   #tree
            [0,255,255],  #low vegetation
            [0,0,255] ])  #building

def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def get_dfc2022_labels():
    return np.asarray([ [35, 31, 32],
                        [219, 95, 87],
                        [219, 151, 87],
                        [219, 208, 87], 
                        [173, 219, 87], 
                        [117, 219, 87],
                        [123, 196, 123],
                        [88, 177, 88],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 128, 0],
                        [88, 176, 167],
                        [153, 93, 19],
                        [87, 155, 219],
                        [0, 98, 255],
                        [35, 31, 32] ])

def get_fbp_labels():
    return np.asarray([     
                        [0,0,0],                
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
                    [200, 150,   0], # airport

                        ])

def get_hpd_labels():
    return np.asarray([     
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
    
# filepath = r'../deepglobe/ori/vis'
# newpath = r'../deepglobe/ori/lab'
# pathDir = recursive_glob(filepath, '.png')
# i = 0
# for filename in pathDir:
#     img = rasterio.open(filename).read().transpose(1,2,0)
#     # col = decode_segmap(img, 'dfc2022', plot=False).transpose(2,0,1).astype('uint8')
#     col = encode_segmap(img) + 1
#     col = np.expand_dims(col, 0)
#     newname = filename.replace(filepath, newpath).replace('_mask.png', '_id.tif')
#     # regionname = newname.split('/')[-2]
#     # name = name[0].split('\\')[-1]
#     # os.makedirs(filepath+'/'+regionname, exist_ok=True)
#     # os.makedirs(newpath+'/'+regionname, exist_ok=True)
#     # os.makedirs
#     with rasterio.open(newname, mode='w', driver='GTiff', height=col.shape[1], width=col.shape[2],
#                     count=col.shape[0], dtype=col.dtype) as dst:
#         dst.write(col)
#     i = i+1
#     print(i)

# filepath = r'label_ori'
# newpath = r'label_ori_col'

# filepath = r'iter_40000_conb'
# newpath = r'iter_40000_conb_col_new'

# filepath = r'mmsegmentation/show_dirs/test1127/vit_vit-b16_mln_upernet_2xb2-40k_fbp-4096x4096_ds.py/best_mIoU_iter_40000.pth_conb'
# newpath = r'mmsegmentation/show_dirs/test1127/vit_vit-b16_mln_upernet_2xb2-40k_fbp-4096x4096_ds.py/best_mIoU_iter_40000.pth_conb_col'

filepath = r'show_dirs/test0417/geofed_testbig_1024'
newpath = r'show_dirs/test0417/geofed_testbig_1024_id'
os.makedirs(newpath, exist_ok=True)
pathDir = recursive_glob(filepath, '.png')
i = 0
for filename in pathDir:
    if 'mask' in filename:
        continue
    # img = rasterio.open(filename).read(1)
    img = Image.open(filename)
    img = np.array(img)
    # col = decode_segmap(img, 'fbp', plot=False).transpose(2,0,1).astype('uint8')
    # newname = filename.replace(filepath, newpath).replace('.png', '.tif')
    # col = decode_segmap(img, 'fbp', plot=False).astype('uint8')
    
    col = encode_segmap(img)
    
    newname = filename.replace(filepath, newpath)
    # regionname = newname.split('/')[-2]
    # name = name[0].split('\\')[-1]
    # os.makedirs(filepath+'/'+regionname, exist_ok=True)
    # os.makedirs(newpath+'/'+regionname, exist_ok=True)
    col = Image.fromarray(col)
    col.save(newname)
    
    # with rasterio.open(newname, mode='w', driver='GTiff', height=col.shape[1], width=col.shape[2],
    #                 count=col.shape[0], dtype=col.dtype) as dst:
    #     dst.write(col)

    i = i+1
    print(i)