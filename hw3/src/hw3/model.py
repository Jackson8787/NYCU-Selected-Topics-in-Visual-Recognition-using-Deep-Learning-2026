"""Model construction for HW3."""

from __future__ import annotations

import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

from hw3.config import NUM_CLASSES


SUPPORTED_ARCHITECTURES = ("maskrcnn_r50_fpn", "maskrcnn_r50_fpn_v2")
SUPPORTED_PRETRAINING = ("imagenet", "coco", "none")


def build_model(
    num_classes: int = NUM_CLASSES,
    architecture: str = "maskrcnn_r50_fpn",
    pretrained: str = "imagenet",
    min_size: tuple[int, ...] = (512, 640, 768, 896, 1024),
    max_size: int = 1536,
    trainable_backbone_layers: int = 5,
    small_cell_anchors: bool = False,
    detections_per_img: int = 300,
) -> torch.nn.Module:
    """Build Mask R-CNN for HW3.

    COCO torchvision weights are allowed by the HW3 E3 clarification posted on
    2026-04-27. For COCO pretraining, replace the COCO class predictors while
    keeping pretrained backbone/FPN/RPN/ROI feature layers.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if pretrained not in SUPPORTED_PRETRAINING:
        raise ValueError(f"Unsupported pretrained setting: {pretrained}")

    kwargs = {}
    anchor_generator = None
    if small_cell_anchors:
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        if architecture == "maskrcnn_r50_fpn":
            kwargs["rpn_anchor_generator"] = anchor_generator
        kwargs["rpn_pre_nms_top_n_train"] = 4000
        kwargs["rpn_pre_nms_top_n_test"] = 3000
        kwargs["rpn_post_nms_top_n_train"] = 2000
        kwargs["rpn_post_nms_top_n_test"] = 1500

    builder = (
        maskrcnn_resnet50_fpn_v2
        if architecture == "maskrcnn_r50_fpn_v2"
        else maskrcnn_resnet50_fpn
    )

    if pretrained == "coco":
        weights = (
            MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            if architecture == "maskrcnn_r50_fpn_v2"
            else MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        model = builder(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
            box_detections_per_img=detections_per_img,
            **kwargs,
        )
        replace_predictors(model, num_classes)
    else:
        weights_backbone = (
            ResNet50_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        )
        model = builder(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
            box_detections_per_img=detections_per_img,
            **kwargs,
        )
    if architecture == "maskrcnn_r50_fpn_v2" and anchor_generator is not None:
        model.rpn.anchor_generator = anchor_generator
    return model


def replace_predictors(model: torch.nn.Module, num_classes: int) -> None:
    """Replace COCO predictors with HW3 class predictors."""
    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)

    mask_in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_channels,
        hidden_layer,
        num_classes,
    )


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> dict:
    """Load model weights and return the checkpoint dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint
