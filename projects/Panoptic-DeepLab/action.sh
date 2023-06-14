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
# Training
# python train_net.py \
# --config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
# --num-gpus 2 \
# MODEL.WEIGHTS model_final_bd324a.pkl

# --resume 

# Inference
export CUDA_VISIBLE_DEVICES=1
python train_net.py \
--config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
--eval-only MODEL.WEIGHTS output/model_0004999.pth

# Visualize prediction result
# Benchmark network speed
# python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED True

# # Visualize prediction result
# # /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/output/inference/predictions.json\
# /home/lab530/KenYu/detectron2/tools/visualize_json_results.py \
# --input /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/output_original_panoptic_seg/inference/predictions.json \
# --output /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/viz_depth_output/. \
# --dataset cityscapes_fine_panoptic_val \
# --conf-threshold 0.5
