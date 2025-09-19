import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F

epsilon = 1e-4

def hsic1(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    '''
    Batched version of HSIC.
    :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
    :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
    :return: HSIC tensor, Size = (B)
    '''
    assert K.size() == L.size()
    assert K.dim() == 3
    K = K.clone()
    L = L.clone()
    n = K.size(1)

    # K, L --> K~, L~ by setting diagonals to zero
    K.diagonal(dim1=-1, dim2=-2).fill_(0)
    L.diagonal(dim1=-1, dim2=-2).fill_(0)

    KL = torch.bmm(K, L)
    trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
    middle_term = K.sum((-1, -2), keepdim=True) * L.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)
    right_term = KL.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)
    main_term = trace_KL + middle_term - right_term
    hsic = main_term / (n ** 2 - 3 * n)
    return hsic.squeeze(-1).squeeze(-1)


hsic_matrix = torch.zeros(16)
self_hsic_x = torch.zeros(1, 4)
self_hsic_y = torch.zeros(4, 1)
pathK = 'proj/spvit/work_dirs/visattn/0922/1024_1card' #5
# pathK = 'proj/spvit/work_dirs/visattn/0922/1024_4card_512' #4
# pathK = 'proj/spvit/work_dirs/visattn/0922/512_1card_feat_comb' #0
# pathK = 'proj/spvit/work_dirs/visattn/0922/1024_4card_512_512mod' #0
# pathK = 'proj/spvit/work_dirs/visattn/0922/512_1card_nospim_comb' #0


# pathL = 'proj/spvit/work_dirs/visattn/0922/1024_4card_512_512mod' #0 9747
# pathL = 'proj/spvit/work_dirs/visattn/0922/1024_4card_512' #4 9893
# pathL = 'proj/spvit/work_dirs/visattn/0922/512_1card_feat_comb' #0 9876
# pathL = 'proj/spvit/work_dirs/visattn/0922/1024_1card_nospim' #0 9212
pathL = 'proj/spvit/work_dirs/visattn/0922/512_1card_nospim_comb' #0 9748

# for i in range(52):
for i in range(53):
# for i in range(32):
    print(i)
    K = torch.tensor(np.load(pathK+'/'+str(i+5)+'.npy'))
    # K = torch.tensor(np.load(pathK+'/'+str(i+4)+'.npy'))
    # K = torch.tensor(np.load(pathK+'/'+str(i)+'.npy'))
    
    # L = torch.tensor(np.load(pathL+'/'+str(i+4)+'.npy'))
    L = torch.tensor(np.load(pathL+'/'+str(i)+'.npy'))
    # import pdb; pdb.set_trace()
    KK = torch.cat([K, K, K, K])
    LL = torch.cat([L[0:1], L[0:1], L[0:1], L[0:1],    L[1:2], L[1:2], L[1:2], L[1:2],    L[2:3], L[2:3], L[2:3], L[2:3],    L[3:4], L[3:4], L[3:4], L[3:4], ])
    self_hsic_x += hsic1(K, K) * epsilon
    self_hsic_y += hsic1(L, L).reshape(4, 1) * epsilon
    hsic_matrix += hsic1(KK, LL) * epsilon
    
cka_matrix = hsic_matrix.reshape(4, 4) / torch.sqrt(self_hsic_x * self_hsic_y)
print(cka_matrix)
# print(cka_matrix.diagonal())
print(cka_matrix.diagonal().mean())