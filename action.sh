#################
### Mask_RCNN ###
#################
# Test and evaluate mask-RCNN on cityscape validation set
# tools/train_net.py \
# --config-file /home/lab530/KenYu/detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml \
# --eval-only MODEL.WEIGHTS /home/lab530/KenYu/detectron2/pretrain_weight/model_final_af9cf5.pkl

# Visualize groundtrue of Cityscape validation set
# tools/visualize_data.py \
# --source annotation \
# --config-file /home/lab530/KenYu/detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml \
# --output-dir /home/lab530/KenYu/detectron2/viz_cityscape_instance_seg_groundtrue/

##############################
### Panoptic Segmentation  ###
##############################

# # Visualize groundtrue of cityscape panoptic segmentation validation set 
# tools/visualize_data.py \
# --source annotation \
# --config-file /home/lab530/KenYu/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml \
# --output-dir /home/lab530/KenYu/detectron2/viz_cityscape_panoptic_seg_groundtrue/
