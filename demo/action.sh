export CUDA_VISIBLE_DEVICES=1

#################################################
### Output Panoptic-Deeplab prediction result ###
#################################################
python demo.py \
--config-file /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml \
--input /home/lab530/KenYu/cityscapes/leftImg8bit/val/*/*.png \
--output /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/viz_depth_output \
--opts MODEL.WEIGHTS /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/model_final_bd324a.pkl



# /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/output/model_0009999.pth

#####################################################
### Output Panoptic-Depth prediction result ###
#####################################################
# python demo.py \
# --config-file /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
# --input /home/lab530/KenYu/cityscapes/leftImg8bit/val/*/*.png \
# --output /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/viz_depth_output \
# --opts MODEL.WEIGHTS /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/output/model_0009999.pth

# Output Mask-RCNN instance segmentation prediction result on cityscape validation set
# python demo.py \
# --config-file /home/lab530/KenYu/detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml \
# --input /home/lab530/KenYu/cityscapes/leftImg8bit/val/*/*.png \
# --output /home/lab530/KenYu/detectron2/viz_output_mask_RCNN_cityscape \
# --opts MODEL.WEIGHTS /home/lab530/KenYu/detectron2/pretrain_weight/model_final_af9cf5.pkl
