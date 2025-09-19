#CUDA_VISIBLE_DEVICES=2    tools/dist_test.sh      configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512_test.py            checkpoints/baseline_fbp_swin_large.pth            1     --out             show_dirs/test0331      --work-dir        show_dirs/test0331 


#CUDA_VISIBLE_DEVICES=3    tools/dist_test.sh      configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_2048.py          checkpoints/REST_water_swin_large.pth            1     --out             show_dirs/test0331_1      --work-dir        show_dirs/test0331_1

#CUDA_VISIBLE_DEVICES=2,3    tools/dist_test.sh      configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_2048.py          checkpoints/REST_water_swin_large.pth            2     --out             show_dirs/test0331_2      --work-dir        show_dirs/test0331_2

#CUDA_VISIBLE_DEVICES=0,1,2,3    tools/dist_test.sh      configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_2048.py          checkpoints/REST_water_swin_large.pth            4     --out             show_dirs/test0331_4      --work-dir        show_dirs/test0331_4

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    tools/dist_test.sh      configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_2048.py          checkpoints/REST_water_swin_large.pth            8     --out             show_dirs/test0331_8      --work-dir        show_dirs/test0331_8

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    tools/dist_test.sh      configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_12800.py          checkpoints/REST_water_swin_large.pth            8     --out             show_dirs/test0331_8_ws      --work-dir        show_dirs/test0331_8_ws