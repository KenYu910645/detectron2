# # Train, doesn't support single-gpu training, this will make lab crash
# python train_net.py \
# --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml \
# --num-gpus 2 \
# MODEL.WEIGHTS model_final_bd324a.pkl

# Inference
# export CUDA_VISIBLE_DEVICES=1
# python train_net.py \
# --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml \
# --eval-only MODEL.WEIGHTS model_final_bd324a.pkl

#########################
### panoptic_depthlab ###
#########################
# # Training
# python train_net.py \
# --config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
# --num-gpus 2 \
# MODEL.WEIGHTS model_final_bd324a.pkl
# --resume

# Inference
export CUDA_VISIBLE_DEVICES=0
python train_net.py \
--config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
--eval-only MODEL.WEIGHTS output_depthlab_L1loss/model_0004999.pth

# export CUDA_VISIBLE_DEVICES=1
# python train_net.py \
# --config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab_dorn.yaml \
# --eval-only MODEL.WEIGHTS output_dorn/model_0009999.pth

# Visualize prediction result
# Benchmark network speed
# python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED True
