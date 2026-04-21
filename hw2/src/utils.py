import json
import random
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor


warnings.filterwarnings(
    "ignore",
    message=(
        ".*copying from a non-meta parameter in the checkpoint "
        "to a meta parameter.*"
    ),
    category=UserWarning,
)

ID2LABEL = {index: str(index) for index in range(10)}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def resolve_path(path, base_dir=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir) / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_processor(config):
    return DetrImageProcessor(
        size={
            "shortest_edge": config["shortest_edge"],
            "longest_edge": config["longest_edge"],
        },
    )


def build_model(config):
    kwargs = {
        "num_labels": config["num_labels"],
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "ignore_mismatched_sizes": True,
    }
    if config.get("num_queries", 100) != 100:
        kwargs["num_queries"] = config["num_queries"]

    init_checkpoint = config.get("init_checkpoint")
    if init_checkpoint:
        return DetrForObjectDetection.from_pretrained(
            init_checkpoint,
            **kwargs,
        )

    model_name = config["model_name"].lower()
    backbone_name = "resnet50"
    use_dc5 = "dc5" in model_name
    detr_config = DetrConfig(
        backbone=backbone_name,
        use_pretrained_backbone=True,
        dilation=use_dc5,
        num_labels=config["num_labels"],
        num_queries=config.get("num_queries", 100),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return DetrForObjectDetection(detr_config)


def move_labels_to_device(labels, device):
    moved = []
    for label in labels:
        moved.append({key: value.to(device) for key, value in label.items()})
    return moved


def make_optimizer(model, config):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": config["learning_rate"]},
            {
                "params": backbone_params,
                "lr": config["backbone_learning_rate"],
            },
        ],
        weight_decay=config["weight_decay"],
    )


def make_scheduler(optimizer, config, steps_per_epoch):
    scheduler_name = config.get("scheduler", "none")
    total_steps = max(steps_per_epoch * config["epochs"], 1)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.0))

    if scheduler_name == "none":
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
        min_factor = config.get("min_lr_factor", 0.05)
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_torch():
    # The DETR/timm path in this Windows environment is faster with PyTorch
    # defaults than with forced TF32/cuDNN benchmark flags.
    return


def save_checkpoint(model, processor, output_dir, config, name):
    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    save_json(config, checkpoint_dir / "baseline_config.json")
    return checkpoint_dir


def load_model_and_processor(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    processor = DetrImageProcessor.from_pretrained(checkpoint_dir)
    model = DetrForObjectDetection.from_pretrained(checkpoint_dir)
    model.to(device)
    return model, processor


def zip_prediction(pred_json_path, zip_path):
    pred_json_path = Path(pred_json_path)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.write(pred_json_path, arcname="pred.json")
