# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from .post_processing import get_panoptic_segmentation
from .dorn_loss import OrdinalRegressionLoss

__all__ = ["PanopticDeepLab", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch", "DEPTH_BRANCH_REGISTRY", "build_depth_branch"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""
DEPTH_BRANCH_REGISTRY = Registry("DEPTH_BRANCH_REGISTRY")
DEPTH_BRANCH_REGISTRY.__doc__ = """
Registry for depth estimation branches, which make depth estimation
"""


@META_ARCH_REGISTRY.register()
class PanopticDeepLab(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head   = build_sem_seg_head    (cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.depth_head     = build_depth_branch    (cfg, self.backbone.output_shape())
        self.register_buffer(
            "pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False
        )
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = (
            cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = (
            cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)

        features = self.backbone(images.tensor)

        # print(f"features = {features.keys()}")
        # print(f"features['res2'] = {features['res2'].shape}") # [1, 256, 256, 512], 1/4
        # print(f"features['res3'] = {features['res3'].shape}") # [1, 512, 128, 256], 1/8
        # print(f"features['res5'] = {features['res5'].shape}") # [1, 2048, 64, 128], 1/16

        losses = {}

        ####################################
        ### Semantic Segmentation Branch ###
        ####################################
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None

        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, targets, weights)
        losses.update(sem_seg_losses)

        # print(f"targets = {targets.shape}") # [1, 1024, 2048]
        # print(f"sem_seg_results = {sem_seg_results.shape}") # [1, 19, 1024, 2048] # 19 different classes

        ####################################
        ### Instance Segmentation Branch ###
        ####################################
        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [
                x["center_weights"].to(self.device) for x in batched_inputs
            ]
            center_weights = ImageList.from_tensors(
                center_weights, size_divisibility
            ).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(
                offset_targets, size_divisibility
            ).tensor
            offset_weights = [
                x["offset_weights"].to(self.device) for x in batched_inputs
            ]
            offset_weights = ImageList.from_tensors(
                offset_weights, size_divisibility
            ).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

        (center_results,
         offset_results,
         center_losses,
         offset_losses,) = self.ins_embed_head(features, center_targets, center_weights, offset_targets, offset_weights)
        # print(f"center_targets = {center_targets.shape}") # None
        # print(f"offset_targets = {offset_targets.shape}") # None
        # print(f"center_results = {center_results.shape}") # [1, 1, 1024, 2048]
        # print(f"offset_results = {offset_results.shape}") # [1, 2, 1024, 2048]

        losses.update(center_losses)
        losses.update(offset_losses)

        ################################
        ### Depth Estimation  Branch ###
        ################################
        if "depth" in batched_inputs[0]:
            targets = [x["depth"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(targets, size_divisibility).tensor
            
            weights = [torch.ones_like(x["depth"]).to(self.device) for x in batched_inputs]
            weights = ImageList.from_tensors(weights, size_divisibility).tensor
            
            # Ignore pixels with depth = 0
            weights[targets == 0] = 0

        else:
            targets = None
            weights = None

        depth_results, depth_losses = self.depth_head(features, targets, weights)
        losses.update(depth_losses)

        if self.training:
            # print(f"losses = {losses}")
            return losses

        if self.benchmark_network_speed:
            return []

        ###########################################
        ### Post-processing(Only for inference) ###
        ###########################################
        processed_results = []
        for (sem_seg_result , center_result , offset_result , depth_result,  input_per_image, image_size,) in zip(
             sem_seg_results, center_results, offset_results, depth_results, batched_inputs , images.image_sizes,):
            
            # print(f"sem_seg_result = {sem_seg_result.shape}") # [19, 1024, 2048]
            # print(f"center_result = {center_result.shape}")   # [1, 1024, 2048]
            # print(f"depth_result = {depth_result.shape}") # [1, 1024, 2048]
            # print(f"offset_result = {offset_result.shape}")   # [2, 1024, 2048]
            # print(f"input_per_image = {input_per_image}")
            # print(f"image_size = {image_size}") # (1024, 2048)
            
            height = input_per_image.get("height")
            width  = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result , image_size, height, width)
            o = sem_seg_postprocess(offset_result , image_size, height, width)
            # print(f"r = {r.shape}") # [19,1024, 2048]
            # print(f"c = {c.shape}") # [1, 1024, 2048]
            # print(f"o = {o.shape}") # [2, 1024, 2048]
            
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,)
            # print(f"panoptic_image = {panoptic_image.shape}") # [1, 1024, 2048]
            
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            
            # For depth estimation evaluation.
            processed_results[-1]["depth"] = depth_result
            
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(
                            instance.pred_masks
                        ).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        # print(f"len(processed_results) = {len(processed_results)}") # 1
        # print(processed_results[0])
        return processed_results


@SEM_SEG_HEADS_REGISTRY.register()
class PanopticDeepLabSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only
        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        # print(f"loss_type = {loss_type}") # hard_pixel_mining
        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(
                ignore_label=ignore_value, top_k_percent_pixels=loss_top_k
            )
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"]    = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)  # [7, 19, 128, 256]
        # print(f"self.common_stride = {self.common_stride}") # 4
        if self.training:
            return None, self.losses(y, targets, weights)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )  # [7, 19, 512, 1024]
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)

@INS_EMBED_BRANCHES_REGISTRY.register()
class PanopticDeepLabInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(
            input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs
        )
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(
                center,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            offset = (
                F.interpolate(
                    offset,
                    scale_factor=self.common_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        # print(f"offset shape = {offset.shape}") # [1, 32, 128, 256]
        offset = self.offset_predictor(offset)
        # print(f"offset shape = {offset.shape}") # [1, 2, 128, 256]
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses

def build_depth_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.DEPTH_HEAD.NAME
    return DEPTH_BRANCH_REGISTRY.get(name)(cfg, input_shape)

@DEPTH_BRANCH_REGISTRY.register()
class PanopticDeepLabDepthHead(DeepLabV3PlusHead):
    """
    A Depth Estimation Head
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_type: str,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            # ignore_value=ignore_value,
            **kwargs,)
        assert self.decoder_only
        # self.loss_weight = loss_weight
        self.loss_type = loss_type
        use_bias = norm == ""
        self.beta = 80.0
        self.gamma = 1.0
        self.ord_num = 90
        ############
        ### Head ###
        ############
        # Note that, we temporarily ignore the depth-wise convolution arugments.
        self.head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.head[0])
        weight_init.c2_xavier_fill(self.head[1])

        #################
        ### Predictor ###
        #################
        if loss_type == "dorn":
            self.predictor = Conv2d(head_channels, 2*self.ord_num, kernel_size=1)
        else:
            self.predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        #####################
        ### Loss Function ###
        #####################
        print(f"[panoptic_seg.py] loss_type = {loss_type}")
        if loss_type == "smoothedL1":
            self.loss = nn.L1Loss(reduction="none")
        elif loss_type == "dorn":
            self.loss = OrdinalRegressionLoss(ord_num=self.ord_num, beta=self.beta, discretization="SID")
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.DEPTH_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.DEPTH_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.DEPTH_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.DEPTH_HEAD.IN_FEATURES
            },
            project_channels             = cfg.MODEL.DEPTH_HEAD.PROJECT_CHANNELS,
            aspp_dilations               = cfg.MODEL.DEPTH_HEAD.ASPP_DILATIONS,
            aspp_dropout                 = cfg.MODEL.DEPTH_HEAD.ASPP_DROPOUT,
            decoder_channels             = decoder_channels,
            common_stride                = cfg.MODEL.DEPTH_HEAD.COMMON_STRIDE,
            norm                         = cfg.MODEL.DEPTH_HEAD.NORM,
            train_size                   = train_size,
            head_channels                = cfg.MODEL.DEPTH_HEAD.HEAD_CHANNELS,
            loss_type                    = cfg.MODEL.DEPTH_HEAD.LOSS_TYPE,
            use_depthwise_separable_conv = cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )

        return ret

        # ret = super().from_config(cfg, input_shape)
        # ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        # ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        # return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.training:
            return None, self.losses(y, targets, weights)
        else:
            # TODO, need to test
            if self.loss_type == "dorn": # DORN inference
                N, C, H, W = y.size()
                y = y.view(-1, 2, self.ord_num, H, W)
                prob = F.softmax(y, dim=1)[:, 0, :, :, :]
                label = torch.sum((prob > 0.5), dim=1)
                # print("prob shape:", prob.shape, " label shape:", label.shape)
                t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
                t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
                y = (t0 + t1) / 2 - self.gamma
                # print("depth min:", torch.min(depth), " max:", torch.max(depth),
                #       " label min:", torch.min(label), " max:", torch.max(label))
                # return {"target": [depth], "prob": [prob], "label": [label]}
                y = y.unsqueeze(1)
                print(f"[panoptic_seg.py] y = {y.shape}") # ([1, 256, 512])
            
            y = F.interpolate(y, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            return y, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        predictions = predictions.squeeze(dim=1) # Drop the second dimension
        # print(f"losses predictions = {predictions.shape}") # [2, 512, 1024]
        # print(f"losses targets = {targets.shape}") # [2, 512, 1024]
        
        if self.loss_type == "dorn":
            N, C, H, W = predictions.size()
            predictions = predictions.view(-1, 2, C//2, H, W)
            predictions = F.log_softmax(predictions, dim=1).view(N, C, H, W)
            loss = self.loss(predictions, targets)
        else:
            loss = self.loss(predictions, targets) * weights
            if weights.sum() > 0:
                loss = loss.sum() / weights.sum()
            else:
                loss = loss.sum() * 0
        
        losses = {"loss_depth": loss * self.loss_weight}
        return losses
