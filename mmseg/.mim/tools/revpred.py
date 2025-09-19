import os
import numpy as np
from PIL import Image

srcdir = 'show_dirs/test0319/segvit_vit-l_jax_2048x2048_80k_fbp_sp_1275_amp_ffn_ca.py/best_mIoU_iter_40000.pth_col'
revdir = srcdir + '_rev'
difdir = srcdir + '_dif'

os.makedirs(revdir, exist_ok=True)
os.makedirs(difdir, exist_ok=True)

gtdir = '../../data/Five-Billion-Pixels/fbp_2048/Annotation__index/test'
showdir = '../../data/Five-Billion-Pixels/fbp_2048/Annotation__color/test'

aa  = os.listdir(srcdir)

for imgpath in aa:
    srcpath = os.path.join(srcdir, imgpath)
    gtpath = os.path.join(gtdir, imgpath.replace('.png','_24label.png'))
    srcimg = np.array(Image.open(srcpath))
    gtimg = np.array(Image.open(gtpath))
    srcimg[:,:,0][gtimg==0] = 0
    srcimg[:,:,1][gtimg==0] = 0
    srcimg[:,:,2][gtimg==0] = 0
    
    srcimg = Image.fromarray(srcimg)
    srcimg.save(os.path.join(revdir, imgpath.replace('.png','_rev.png')))
    
    showpath = os.path.join(showdir, imgpath.replace('.png','_24label.png'))
    showimg = np.array(Image.open(showpath))
    dif = showimg - srcimg
    dif = Image.fromarray(dif)
    dif.save(os.path.join(difdir, imgpath.replace('.png','_dif.png')))