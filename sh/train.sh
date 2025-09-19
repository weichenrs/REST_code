#CUDA_VISIBLE_DEVICES=3     tools/dist_train.sh    configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512.py      1    --resume  --amp    --work-dir     work_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512.py

#CUDA_VISIBLE_DEVICES=3     tools/dist_train.sh    configs/vssm/upernet_vssm_4xb4-160k_fbp-512x512_base.py      1    --resume  --amp    --work-dir     work_dirs/fbp/upernet_vssm_4xb4-160k_fbp-512x512_base.py

#CUDA_VISIBLE_DEVICES=3     tools/dist_train.sh    configs/convnext/fbp/convnext-large_upernet_8xb2-amp-80k_fbp-512x512.py      1    --resume  --amp    --work-dir     work_dirs/fbp/convnext-large_upernet_8xb2-amp-80k_fbp-512x512.py 


#CUDA_VISIBLE_DEVICES=0,1,2,3     tools/dist_train.sh    configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-1024x1024_sp.py      4    --resume  --amp    --work-dir     work_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-1024x1024_sp.py

#CUDA_VISIBLE_DEVICES=0,1,2,3     tools/dist_train.sh      configs/convnext/fbp/convnext-large_upernet_8xb2-amp-80k_fbp-2048x2048_sp.py      4    --resume  --amp    --work-dir     work_dirs/fbp/convnext-large_upernet_8xb2-amp-80k_fbp-2048x2048_sp.py.py