import os

# 运行第一个脚本
os.system("python train_hfrm.py")

# 运行第二个脚本（单 GPU 版本）
os.system("CUDA_VISIBLE_DEVICES=0 python train_diffusion.py --config LLIE_wavelet.yml --test_set ./LOLv1/eval15/")