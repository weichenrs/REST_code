# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import argparse
import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
# from erf.resnet_for_erf import resnet101, resnet152
# from erf.replknet_for_erf import RepLKNetForERF
from swin_for_erf import SwinForERF
from torch import optim as optim

from torchvision.datasets import VisionDataset
from collections import OrderedDict

class FBP(VisionDataset):
    def __init__(self, root, transforms):
        super(FBP, self).__init__(root, transforms)
        
        self.images_dir = os.path.join(self.root, 'val', 'img')
        # self.targets_dir = os.path.join(self.root, 'val', 'annotation')
        self.images = []
        # self.targets = []
        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            # self.targets.append(os.path.join(self.targets_dir, file_name))
            
    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert('RGB')
        # target = Image.open(self.targets[index])
        if self.transforms is not None:
            # import pdb; pdb.set_trace()
            # image, target = self.transforms(image, target)
            image = self.transforms(image)

        # return image, target
        return image
    
    def __len__(self) -> int:
        return len(self.images)
    
def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='resnet101', type=str, help='model name')
    parser.add_argument('--weights', default='proj/spvit/swin_large_skysense_20240516.pth', type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--data_path', default='data/FBP_new/crop512_3band', type=str, help='dataset path')
    parser.add_argument('--save_path', default='swin_all_val.npy', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')
    args = parser.parse_args()
    return args


def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    print(out_size)
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def main(args):
    #   ================================= transform: resize to 1024x1024
    t = [
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_path)
    root = os.path.join(args.data_path)
    # dataset = datasets.ImageFolder(root, transform=transform)
    dataset = FBP(root, transforms=transform)
    
    # nori_root = os.path.join('/home/dingxiaohan/ndp/', 'imagenet.val.nori.list')
    # from nori_dataset import ImageNetNoriDataset      # Data source on our machines. You will never need it.
    # dataset = ImageNetNoriDataset(nori_root, transform=transform)

    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

    # if args.model == 'resnet101':
    #     model = resnet101(pretrained=args.weights is None)
    # elif args.model == 'resnet152':
    #     model = resnet152(pretrained=args.weights is None)
    # elif args.model == 'RepLKNet-31B':
    #     model = RepLKNetForERF(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[128,256,512,1024],
    #                 small_kernel=5, small_kernel_merged=False)
    # elif args.model == 'RepLKNet-13':
    #     model = RepLKNetForERF(large_kernel_sizes=[13] * 4, layers=[2,2,18,2], channels=[128,256,512,1024],
    #                 small_kernel=5, small_kernel_merged=False)
    # else:
    #     raise ValueError('Unsupported model. Please add it here.')
    model = SwinForERF()
    
    if args.weights is not None:
        print('load weights')
        
        weights = torch.load(args.weights, map_location='cpu')
        
        if 'state_dict' in weights:
            weights = weights['state_dict']
        elif 'model' in weights:
            weights = weights['model']
        else:
            weights = weights

        state_dict = OrderedDict()
        for k, v in weights.items():
            state_dict[k] = v

        model.load_state_dict(state_dict, strict=False)
        print('loaded')

    model.cuda()
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()
    
    for _, samples in enumerate(data_loader_val):
    # for _, (samples, _) in enumerate(data_loader_val):

        # if meter.count == args.num_images:
        if meter.count == len(data_loader_val):
            np.save(args.save_path, meter.avg)
            exit()

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)


if __name__ == '__main__':
    args = parse_args()
    main(args)