# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_panoptic_deeplab_config
from .dataset_mapper import PanopticDeeplabDatasetMapper
from .panoptic_seg import (
    PanopticDeepLab,
    INS_EMBED_BRANCHES_REGISTRY,
    DEPTH_BRANCH_REGISTRY,
    build_ins_embed_branch,
    build_depth_branch,
    PanopticDeepLabSemSegHead,
    PanopticDeepLabInsEmbedHead,
    PanopticDeepLabDepthHead,
)
