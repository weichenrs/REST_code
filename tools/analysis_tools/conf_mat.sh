#!/bin/bash
#SBATCH --partition=hpxg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16


#python proj/spvit/tools/analysis_tools/confusion_matrix.py                                        proj/spvit/configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-2048x2048_sp_layer_new_data_slide6400.py                                                                       proj/spvit/show_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512_decay_new_bs8_tv.py/1107/best_mIoU_iter_72000.pth                                               proj/spvit/work_dirs/visattn/1224confmat --color-theme jet


#python proj/spvit/tools/analysis_tools/confusion_matrix.py                                        proj/spvit/configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-2048x2048_sp_layer_new_data_slide6400.py                                                                         proj/spvit/show_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-1024x1024_sp_layer_new_data_bs8.py/tv/1110_6400/best_mIoU_iter_4500.pth                                                proj/spvit/work_dirs/visattn/1223confmat --color-theme jet

#python proj/spvit/tools/analysis_tools/confusion_matrix.py                                        proj/spvit/configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-2048x2048_sp_layer_new_data_slide6400.py                                                                       proj/spvit/show_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-1024x1024_sp_layer_new_data_bs8.py/0912_6400_7295/best_mIoU_iter_6500.pth                                               proj/spvit/work_dirs/visattn/1224confmat --color-theme jet

python proj/spvit/tools/analysis_tools/confusion_matrix.py                                        proj/spvit/configs/swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-2048x2048_sp_layer_new_data_slide6400.py                                                                       proj/spvit/show_dirs/fbp/swin/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512_decay_new_bs8.py/0827/best_mIoU_iter_32000.pth                                               proj/spvit/work_dirs/visattn/1224confmat --color-theme jet