import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA


n_choose = 2
pca = PCA(n_components=n_choose)

for distnum in [0,1,2,3]:
    rootdir = 'proj/spvit/work_dirs/visattn/1130/feat/6400' + '/' + str(distnum)
    savepath = 'proj/spvit/work_dirs/visattn/1130/feat/6400_vis' + '/' + str(distnum)
    os.makedirs(savepath, exist_ok=True)

    for name in os.listdir(rootdir):
        fullname = os.path.join(savepath, name[1:]).replace('.npy', '.png')
        
        dd = np.load(os.path.join(rootdir, name))
        
        newdd = pca.fit_transform(np.transpose(dd.reshape(dd.shape[0], -1)))
        # import pdb; pdb.set_trace()
        newdd = np.transpose(newdd).reshape(n_choose, dd.shape[1], dd.shape[2])
        for pcachannel in range(n_choose):
            plt.axis('off')
            plt.imshow(newdd[pcachannel], cmap='jet', interpolation='nearest')
            plt.savefig(fullname.replace('.png', '_' + str(pcachannel) + '.png'), bbox_inches='tight', pad_inches=0)
        print(name)


# for distnum in [0,1,2,3]:
# rootdir = 'proj/spvit/work_dirs/visattn/1130/feat/6400comb'
# savepath = 'proj/spvit/work_dirs/visattn/1130/feat/6400comb_vis'
# # savepath = 'proj/spvit/work_dirs/visattn/1130/feat/6400comb_vis_norm'
# os.makedirs(savepath, exist_ok=True)

# for name in os.listdir(rootdir):
#     fullname = os.path.join(savepath, name[1:]).replace('.npy', '.png')
    
#     dd = np.load(os.path.join(rootdir, name))
    
#     # newdd = pca.fit_transform(np.transpose(dd.reshape(dd.shape[0], -1)))
#     # import pdb; pdb.set_trace()
#     # newdd = np.transpose(newdd).reshape(n_choose, dd.shape[1], dd.shape[2])
    
#     # newdd = np.uint8( (newdd - np.min(newdd))/(np.max(newdd)-np.min(newdd)) * 255 )
    
#     for pcachannel in range(dd.shape[0]):
#         plt.axis('off')
#         plt.imshow(dd[pcachannel], cmap='jet', interpolation='nearest')
#         plt.savefig(fullname.replace('.png', '_' + str(pcachannel) + '.png'), bbox_inches='tight', pad_inches=0)
#         print(pcachannel)
#     print(name)
            
        # import pdb; pdb.set_trace()
                
        # dd =  (dd - np.min(dd))/(np.max(dd)-np.min(dd)) 
        # finalfm = np.uint8( dd.mean(0) * 255 )
        
        # finalfm = dd.mean(0)
        # finalfm = np.uint8( (finalfm - np.min(finalfm))/(np.max(finalfm)-np.min(finalfm)) * 255 )
        # plt.axis('off')
        # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
        # plt.savefig(fullname, bbox_inches='tight', pad_inches=0)

