import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F

# # rootdir = 'proj/spvit/work_dirs/visattn/0918/card4'
# # savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
# # # os.makedirs(savepath, exist_ok=True)



# # rootdir = 'proj/spvit/work_dirs/visattn/0922/512_1card_feat'
# # savepath = 'proj/spvit/work_dirs/visattn/0922/512_1card_feat_comb'
# rootdir = 'proj/spvit/work_dirs/visattn/0922/512_1card_nospim'
# savepath = 'proj/spvit/work_dirs/visattn/0922/512_1card_nospim_comb'





for stage in range(4):
    rootdir = 'proj/spvit/work_dirs/visattn/1130/feat/6400/' + str(stage)
    savepath = 'proj/spvit/work_dirs/visattn/1130/feat/6400comb/' + str(stage)

    os.makedirs(savepath, exist_ok=True)

    fmlist0 = []
    fmlist1 = []
    fmlist2 = []
    for nn in range(int(len(os.listdir(rootdir))/3)):
        print(nn)
        fm0 = np.load(os.path.join(rootdir, '_rank_' + str(nn) +'_after.npy'))
        fmlist0.append(fm0)
        fm1 = np.load(os.path.join(rootdir, '_rank_' + str(nn) +'_spout.npy'))
        fmlist1.append(fm1)
        fm2 = np.load(os.path.join(rootdir, '_rank_' + str(nn) +'_before.npy'))
        fmlist2.append(fm2)
        
    # import pdb; pdb.set_trace()
    oo = 0
    for fmlist in [fmlist0, fmlist1, fmlist2]:
        alllist = []
        for i in range(4):
            templist = []
            for j in range(4):
                templist.append(fmlist[i + int(4) * j])
            temp = np.concatenate([x for x in templist], -1)
            alllist.append(temp)
        finalfm = np.concatenate([x for x in alllist], -2)
        
        if oo == 0:
            np.save(savepath + 'stage'+str(stage)+'_after.npy', finalfm)
        elif oo == 1:
            np.save(savepath + 'stage'+str(stage)+'_spout.npy', finalfm)
        elif oo == 2:
            np.save(savepath + 'stage'+str(stage)+'_before.npy', finalfm)
            
        oo = oo + 1
















    # for nn in range(int(len(os.listdir(rootdir))/16)):
    #     print(nn)
    #     fmlist0 = []
    #     fmlist1 = []
    #     fmlist2 = []
    #     fmlist3 = []
    #     for ss in range(4):
    #         # print(nn*16+ss*4)
            
    #         fm0 = np.load(os.path.join(rootdir, str(nn*16+ss*4)+'.npy'))
    #         fmlist0.append(fm0)
    #         fm1 = np.load(os.path.join(rootdir, str(nn*16+ss*4+1)+'.npy'))
    #         fmlist1.append(fm1)
    #         fm2 = np.load(os.path.join(rootdir, str(nn*16+ss*4+2)+'.npy'))
    #         fmlist2.append(fm2)
    #         fm3 = np.load(os.path.join(rootdir, str(nn*16+ss*4+3)+'.npy'))
    #         fmlist3.append(fm3)
        
    #     fmcka = []
    #     for fmlist in [fmlist0, fmlist1, fmlist2, fmlist3]:
    #         alllist = []
    #         for j in range(2):
    #             templist = []
    #             for i in range(2):
    #                 templist.append(fmlist[i + int(2) * j])
    #             temp = np.concatenate([x for x in templist], -1)
    #             alllist.append(temp)
    #         finalfm = np.concatenate([x for x in alllist], -2)
            
    #         # import pdb; pdb.set_trace()
    #         finalfm = torch.tensor(finalfm)
    #         batch_size = finalfm.size(0)
    #         finalfm = finalfm.reshape(batch_size, -1).double()
    #         finalfm = finalfm.matmul(finalfm.t())
    #         fmcka.append(finalfm)
            
    #     KK = torch.stack(fmcka, dim=0).detach().cpu().numpy()
    #     # import pdb; pdb.set_trace()
    #     np.save(savepath +'/'+ str(nn) +'.npy', KK)
            



# for stage in range(len(os.listdir(rootdir))):
#     # name = os.listdir(rootdir)[stage]
#     name = 'stage' + str(stage)
#     for scale in ['256', '512', '1024']:
#         fmlist = []
#         root = os.path.join(rootdir, name, scale)
#         for nn in range(len(os.listdir(root))):
#             fm = np.load(os.path.join(root, str(nn+1)+'.npy'))
#             fmlist.append(fm)
        
#         # import pdb; pdb.set_trace()
#         if scale != '1024':
#             alllist = []
#             for j in range(int(np.sqrt(nn+1))):
#                 templist = []
#                 for i in range(int(np.sqrt(nn+1))):
#                     templist.append(fmlist[i + int(np.sqrt(nn+1)) * j])
#                 temp = np.concatenate([x for x in templist], 2)
#                 alllist.append(temp)
#             finalfm = np.concatenate([x for x in alllist], 1)
#         else:
#             finalfm = fm
#         # import pdb; pdb.set_trace()
#         np.save(savepath + 'stage'+str(stage)+'_'+scale+'.npy', finalfm)
        


# rootdir = 'proj/spvit/work_dirs/visattn/0918/card1_512'
# savepath = 'proj/spvit/work_dirs/visattn/0918/card1_512_comb'
# os.makedirs(savepath, exist_ok=True)

# for stage in range(len(os.listdir(rootdir))):
#     # name = os.listdir(rootdir)[stage]
#     name = 'stage' + str(stage)
#     for ss in ['_after', '_before', '_spout']:
#         fmlist = []
#         root = os.path.join(rootdir, name)
#         for oo in ['1', '4', '7', '10']:
#             fm = np.load(os.path.join(root, oo + ss +'.npy'))
#             fmlist.append(fm)
        
#         # import pdb; pdb.set_trace()

#         alllist = []
#         for j in range(2):
#             templist = []
#             for i in range(2):
#                 templist.append(fmlist[i + 2 * j])
#             temp = np.concatenate([x for x in templist], 2)
#             alllist.append(temp)
#         finalfm = np.concatenate([x for x in alllist], 1)

#         # import pdb; pdb.set_trace()
#         os.makedirs(savepath + '/stage'+str(stage), exist_ok=True)
#         np.save(savepath + '/stage'+str(stage)+'/0'+ss+'.npy', finalfm)
                



# rootdir = 'proj/spvit/work_dirs/visattn/0918/card4'
# savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
# # os.makedirs(savepath, exist_ok=True)

# for stage in range(len(os.listdir(rootdir))):
#     a = np.load('proj/spvit/work_dirs/visattn/0914stage'+str(stage)+'_256.npy')
#     b = np.load('proj/spvit/work_dirs/visattn/0914stage'+str(stage)+'_512.npy')
#     c = np.load('proj/spvit/work_dirs/visattn/0914stage'+str(stage)+'_1024.npy')
    
#     finalfm = a.mean(0)
#     finalfm = np.uint8( (finalfm - np.min(finalfm))/(np.max(finalfm)-np.min(finalfm)) * 255 )
#     plt.axis('off')
#     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     plt.savefig(savepath + 'stage'+str(stage)+'_256'+'.png', bbox_inches='tight', pad_inches=0)

#     finalfm = b.mean(0)
#     finalfm = np.uint8( (finalfm - np.min(finalfm))/(np.max(finalfm)-np.min(finalfm)) * 255 )
#     plt.axis('off')
#     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     plt.savefig(savepath + 'stage'+str(stage)+'_512'+'.png', bbox_inches='tight', pad_inches=0)

#     finalfm = c.mean(0)
#     finalfm = np.uint8( (finalfm - np.min(finalfm))/(np.max(finalfm)-np.min(finalfm)) * 255 )
#     plt.axis('off')
#     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     plt.savefig(savepath + 'stage'+str(stage)+'_1024'+'.png', bbox_inches='tight', pad_inches=0)

#     a = torch.tensor(a).reshape(a.shape[0], -1)
#     b = torch.tensor(b).reshape(b.shape[0], -1)
#     c = torch.tensor(c).reshape(c.shape[0], -1)

#     simAB = F.cosine_similarity(a, b).mean()
#     simBC = F.cosine_similarity(b, c).mean()
#     simAC = F.cosine_similarity(a, c).mean()
#     print(stage, simAB, simBC, simAC)






# rootdir = 'proj/spvit/work_dirs/visattn/0918/card4'
# savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
# # os.makedirs(savepath, exist_ok=True)

# for stage in range(len(os.listdir(rootdir))):
#     # name = os.listdir(rootdir)[stage]
#     name = 'stage' + str(stage)
    
#     os.makedirs(os.path.join(savepath, name), exist_ok=True)

#     root = os.path.join(rootdir, name)
#     for suffix in ['after', 'before', 'spout']:
#         fmlist = []
#         for nn in range(int(len(os.listdir(root))/3)):
#             fm = np.load(os.path.join(root, str(nn)+'_'+suffix+'.npy'))
#             fmlist.append(fm)
        
#         # import pdb; pdb.set_trace()
#         alllist = []
#         for j in range(int(np.sqrt(4))):
#             templist = []
#             for i in range(int(np.sqrt(4))):
#                 templist.append(fmlist[i + int(np.sqrt(4)) * j])
#             temp = np.concatenate([x for x in templist], 1)
#             alllist.append(temp)
#         finalfm = np.concatenate([x for x in alllist], 2)

#         # import pdb; pdb.set_trace()
        
#         np.save(savepath +'/'+ name +'/'+ '0_'+suffix+'.npy', finalfm)





# rootdir = 'proj/spvit/work_dirs/visattn/0918/card4'
# savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
# # os.makedirs(savepath, exist_ok=True)

# for stage in range(len(os.listdir(rootdir))):
#     a = np.load('proj/spvit/work_dirs/visattn/0914stage'+str(stage)+'_1024'+'before.npy')
#     b = np.load('proj/spvit/work_dirs/visattn/0914stage'+str(stage)+'_1024'+'after.npy') 
#     finalfm = a.mean(0)
#     finalfm = np.uint8( (finalfm - np.min(finalfm))/(np.max(finalfm)-np.min(finalfm)) * 255 )
#     plt.axis('off')
#     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'1before'+'.png', bbox_inches='tight', pad_inches=0)
#     finalfm = b.mean(0)
#     finalfm = np.uint8( (finalfm - np.min(finalfm))/(np.max(finalfm)-np.min(finalfm)) * 255 )
#     plt.axis('off')
#     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'1after'+'.png', bbox_inches='tight', pad_inches=0)
#     a = torch.tensor(a).reshape(a.shape[0], -1)
#     b = torch.tensor(b).reshape(b.shape[0], -1)
#     simAB = F.cosine_similarity(a, b).mean()
#     print(stage, simAB)
    
    
    
    
    
    
    
    
    
    
    
# rootdir = 'proj/spvit/work_dirs/visattn/0918/card4'
# savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
# # os.makedirs(savepath, exist_ok=True)

    
# min=10.0
# max=-10.0
# for stage in range(4):
#     savepath = 'proj/spvit/work_dirs/visattn/0918/card1'
#     a0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_before.npy')
#     b0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_after.npy')
#     c0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_spout.npy')
    
#     min=np.min([min, np.min(a0)])
#     max=np.max([max, np.max(a0)])
#     min=np.min([min, np.min(b0)])
#     max=np.max([max, np.max(b0)])
#     min=np.min([min, np.min(c0)])
#     max=np.max([max, np.max(c0)])

#     savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
#     a1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_before.npy')
#     b1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_after.npy')
#     c1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_spout.npy')
    
#     min=np.min([min, np.min(a1)])
#     max=np.max([max, np.max(a1)])
#     min=np.min([min, np.min(b1)])
#     max=np.max([max, np.max(b1)])
#     min=np.min([min, np.min(c1)])
#     max=np.max([max, np.max(c1)])
    
# print('min', min)
# print('max', max)
    
    

# for stage in range(4):
#     savepath = 'proj/spvit/work_dirs/visattn/0918/card1'
#     a0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_before.npy')
#     b0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_after.npy')
#     c0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_spout.npy')
    
    
#     # finalfm = a0.mean(0)
#     # finalfm = np.uint8( (finalfm - min)/(max-min) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_before_card1'+'.png', bbox_inches='tight', pad_inches=0)

#     # finalfm = b0.mean(0)
#     # finalfm = np.uint8( (finalfm - min)/(max-min) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_after_card1'+'.png', bbox_inches='tight', pad_inches=0)

#     # finalfm = c0.mean(0)
#     # finalfm = np.uint8( (finalfm - min)/(max-min) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_spout_card1'+'.png', bbox_inches='tight', pad_inches=0)


#     savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
#     a1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_before.npy')
#     b1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_after.npy')
#     c1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_spout.npy')
    
#     # finalfm = a1.mean(0)
#     # finalfm = np.uint8( (finalfm - min)/(max-min) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_before_card4'+'.png', bbox_inches='tight', pad_inches=0)

#     # finalfm = b1.mean(0)
#     # finalfm = np.uint8( (finalfm - min)/(max-min) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_after_card4'+'.png', bbox_inches='tight', pad_inches=0)

#     # finalfm = c1.mean(0)
#     # finalfm = np.uint8( (finalfm - min)/(max-min) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_spout_card4'+'.png', bbox_inches='tight', pad_inches=0)
    
#     savepath = 'proj/spvit/work_dirs/visattn/0918/card1_512_comb'
#     a2 = np.load('proj/spvit/work_dirs/visattn/0918/card1_512_comb/stage'+str(stage)+'/'+'0_before.npy')
#     b2 = np.load('proj/spvit/work_dirs/visattn/0918/card1_512_comb/stage'+str(stage)+'/'+'0_after.npy')
#     c2 = np.load('proj/spvit/work_dirs/visattn/0918/card1_512_comb/stage'+str(stage)+'/'+'0_spout.npy')
    
#     # amin = np.min(a2)
#     # amax = np.max(a2)
#     # bmin = np.min(b2)
#     # bmax = np.max(b2)
#     # cmin = np.min(c2)
#     # cmax = np.max(c2)
    
#     # finalfm = a2.mean(0)
#     # finalfm = np.uint8( (finalfm - amin)/(amax-amin) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_before_card1_512'+'.png', bbox_inches='tight', pad_inches=0)

#     # finalfm = b2.mean(0)
#     # finalfm = np.uint8( (finalfm - amin)/(amax-amin) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_after_card1_512'+'.png', bbox_inches='tight', pad_inches=0)

#     # finalfm = c2.mean(0)
#     # finalfm = np.uint8( (finalfm - amin)/(amax-amin) * 255 )
#     # plt.axis('off')
#     # plt.imshow(finalfm, cmap='jet', interpolation='nearest')
#     # plt.savefig(savepath +'/'+ 'stage'+str(stage)+'_spout_card1_512'+'.png', bbox_inches='tight', pad_inches=0)
    

#     a0 = torch.tensor(a0).reshape(a0.shape[0], -1)
#     b0 = torch.tensor(b0).reshape(b0.shape[0], -1)
#     c0 = torch.tensor(c0).reshape(c0.shape[0], -1)
#     a1 = torch.tensor(a1).reshape(a1.shape[0], -1)
#     b1 = torch.tensor(b1).reshape(b1.shape[0], -1)
#     c1 = torch.tensor(c1).reshape(c1.shape[0], -1)
#     a2 = torch.tensor(a2).reshape(a1.shape[0], -1)
#     b2 = torch.tensor(b2).reshape(b1.shape[0], -1)
#     c2 = torch.tensor(c2).reshape(c1.shape[0], -1)

#     sim1 = F.cosine_similarity(a0, a1).mean()
#     # sim2 = F.cosine_similarity(a0, b1).mean()
#     # sim3 = F.cosine_similarity(a0, c1).mean()
#     # print(stage, sim1, sim2, sim3)
#     # sim1 = F.cosine_similarity(b0, a1).mean()
#     sim2 = F.cosine_similarity(b0, b1).mean()
#     # sim3 = F.cosine_similarity(b0, c1).mean()
#     # print(stage, sim1, sim2, sim3)
#     # sim1 = F.cosine_similarity(c0, a1).mean()
#     # sim2 = F.cosine_similarity(c0, b1).mean()
#     sim3 = F.cosine_similarity(c0, c1).mean()
#     print(stage, sim1, sim2, sim3)
    
    
#     # sim1 = F.cosine_similarity(a2, a1).mean()
#     # # sim2 = F.cosine_similarity(a2, b1).mean()
#     # # sim3 = F.cosine_similarity(a2, c1).mean()
#     # # print(stage, sim1, sim2, sim3)
#     # # sim1 = F.cosine_similarity(b2, a1).mean()
#     # sim2 = F.cosine_similarity(b2, b1).mean()
#     # # sim3 = F.cosine_similarity(b2, c1).mean()
#     # # print(stage, sim1, sim2, sim3)
#     # # sim1 = F.cosine_similarity(c2, a1).mean()
#     # # sim2 = F.cosine_similarity(c2, b1).mean()
#     # sim3 = F.cosine_similarity(c2, c1).mean()
#     # print(stage, sim1, sim2, sim3)
    
    
#     sim1 = F.cosine_similarity(a2, a0).mean()
#     # sim2 = F.cosine_similarity(a2, b0).mean()
#     # sim3 = F.cosine_similarity(a2, c0).mean()
#     # print(stage, sim1, sim2, sim3)
#     # sim1 = F.cosine_similarity(b2, a0).mean()
#     sim2 = F.cosine_similarity(b2, b0).mean()
#     # sim3 = F.cosine_similarity(b2, c0).mean()
#     # print(stage, sim1, sim2, sim3)
#     # sim1 = F.cosine_similarity(c2, a0).mean()
#     # sim2 = F.cosine_similarity(c2, b0).mean()
#     sim3 = F.cosine_similarity(c2, c0).mean()
#     print(stage, sim1, sim2, sim3)
    

#     # a0 = torch.tensor(a0).reshape(a0.shape[0], -1)
#     # b0 = torch.tensor(b0).reshape(b0.shape[0], -1)
#     # c0 = torch.tensor(c0).reshape(c0.shape[0], -1)

#     # simAB = F.cosine_similarity(a0, b0).mean()
#     # simBC = F.cosine_similarity(b0, c0).mean()
#     # simAC = F.cosine_similarity(a0, c0).mean()
#     # print(stage, simAB, simBC, simAC)
    
    
    
    
    
    
    
    
    
    
    
    
# rootdir = 'proj/spvit/work_dirs/visattn/0918/card4'
# savepath = 'proj/spvit/work_dirs/visattn/0918/card4_comb'
# # os.makedirs(savepath, exist_ok=True)
    
# for stage in range(4):
    # savepath = 'proj/spvit/work_dirs/visattn/0918/card1_channel'
    # a0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_before.npy')
    # b0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_after.npy')
    # c0 = np.load('proj/spvit/work_dirs/visattn/0918/card1/stage'+str(stage)+'/'+'0_spout.npy')
    # amin = np.min(a0)
    # amax = np.max(a0)
    # bmin = np.min(b0)
    # bmax = np.max(b0)
    # cmin = np.min(c0)
    # cmax = np.max(c0)
    # for i in range(a0.shape[0]):
    #     finalfm = a0[i]
    #     finalfm = np.uint8( (finalfm - amin)/(amax-amin) * 255 )
    #     plt.axis('off')
    #     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
    #     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'/channel_'+str(i)+ '_before_card1.png', bbox_inches='tight', pad_inches=0)

    #     finalfm = b0[i]
    #     finalfm = np.uint8( (finalfm - bmin)/(bmax-bmin) * 255 )
    #     plt.axis('off')
    #     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
    #     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'/channel_'+str(i)+ '_after_card1.png', bbox_inches='tight', pad_inches=0)

    #     finalfm = c0[i]
    #     finalfm = np.uint8( (finalfm - cmin)/(cmax-cmin) * 255 )
    #     plt.axis('off')
    #     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
    #     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'/channel_'+str(i)+ '_spout_card1.png', bbox_inches='tight', pad_inches=0)


    # savepath = 'proj/spvit/work_dirs/visattn/0918/card4_channel'
    # a1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_before.npy')
    # b1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_after.npy')
    # c1 = np.load('proj/spvit/work_dirs/visattn/0918/card4_comb/stage'+str(stage)+'/'+'0_spout.npy')
    # amin = np.min(a1)
    # amax = np.max(a1)
    # bmin = np.min(b1)
    # bmax = np.max(b1)
    # cmin = np.min(c1)
    # cmax = np.max(c1)
    # for i in range(a1.shape[0]):
    #     finalfm = a1[i]
    #     finalfm = np.uint8( (finalfm - amin)/(amax-amin) * 255 )
    #     plt.axis('off')
    #     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
    #     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'/channel_'+str(i)+ '_before_card4.png', bbox_inches='tight', pad_inches=0)

    #     finalfm = b1[i]
    #     finalfm = np.uint8( (finalfm - bmin)/(bmax-bmin) * 255 )
    #     plt.axis('off')
    #     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
    #     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'/channel_'+str(i)+ '_after_card4.png', bbox_inches='tight', pad_inches=0)

    #     finalfm = c1[i]
    #     finalfm = np.uint8( (finalfm - cmin)/(cmax-cmin) * 255 )
    #     plt.axis('off')
    #     plt.imshow(finalfm, cmap='jet', interpolation='nearest')
    #     plt.savefig(savepath +'/'+ 'stage'+str(stage)+'/channel_'+str(i)+ '_spout_card4.png', bbox_inches='tight', pad_inches=0)