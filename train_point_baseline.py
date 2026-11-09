"""Train the pinned upstream P2PNet or public CLTR implementation on leaf-tip points."""
import argparse
from importlib import import_module, util
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import DataLoader


def load_data_module(root):
    # CLTR also has a dataset.py. Resolve the leaf dataset explicitly.
    spec = util.spec_from_file_location("leaf_dataset", root / "dataset.py")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
        metrics = import_module("counting_metrics")
        checkpoints = import_module("checkpointing")
    finally:
        sys.path.pop(0)
    return module, metrics, checkpoints


def load_upstream_utils(root):
    """Bypass obsolete torchvision <0.7 private APIs in upstream util.misc."""
    package = import_module("util")
    path = root / "util" / "misc.py"
    spec = util.spec_from_file_location("util.misc", path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    package.misc = module
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"float\(torchvision\.__version__\[:3\]\) < 0\.7", "False", source)
    source = source.replace('float(torchvision.__version__.split(".")[1]) < 7.0', "False")
    source = source.replace("torchvision._is_tracing()", "torch.jit.is_tracing()")
    exec(compile(source, str(path), "exec"), vars(module))


class PerImageMatcher(nn.Module):
    """Keep upstream costs; avoid cross-image costs and handle empty targets."""

    def __init__(self, matcher, key):
        super().__init__()
        self.matcher = matcher
        self.key = key

    @torch.no_grad()
    def forward(self, outputs, targets):
        matches = []
        for i, target in enumerate(targets):
            if not len(target[self.key]):
                matches.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            if len(target[self.key]) > outputs["pred_points"].shape[1]:
                raise ValueError("More targets than queries; increase CLTR query count")
            single = {k: outputs[k][i:i+1] for k in ("pred_points", "pred_logits")}
            matches.extend(self.matcher(single, [target]))
        return matches


def build_model(args, initializing):
    from torchvision import models as tv
    weights = torch.load(args.weights, map_location="cpu", weights_only=True) if initializing else None
    if args.model == "p2pnet":
        from models.p2pnet import build
        from models.backbone import BackboneBase_VGG
        from torchvision.models.vgg import cfgs, make_layers

        features = make_layers(cfgs["D"], batch_norm=True)
        if weights is not None:
            features.load_state_dict({k.removeprefix("features."): v for k, v in weights.items()
                                      if k.startswith("features.")}, strict=True)
        backbone = BackboneBase_VGG(SimpleNamespace(features=features), 256, "vgg16_bn", True)
        config = SimpleNamespace(row=2, line=2, set_cost_class=1, set_cost_point=.05,
                                 point_loss_coef=.0002, eos_coef=.5)
        with patch("models.p2pnet.build_backbone", return_value=backbone):
            model, criterion = build(config, training=True)
        criterion.matcher = PerImageMatcher(criterion.matcher, "point")
        # The official criterion returns loss_point, while its weight dict
        # spells the key loss_points. Wire the published regression term in.
        criterion.weight_dict["loss_point"] = criterion.weight_dict.pop("loss_points")
        return model, criterion

    from Networks.CDETR import build_model as build
    from Networks.CDETR.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
    from Networks.CDETR.position_encoding import build_position_encoding

    config = SimpleNamespace(dataset_file="crowd_data", device="cuda", backbone="resnet50",
        position_embedding="sine", hidden_dim=256, nheads=8, num_queries=args.cltr_queries,
        dropout=.1, dim_feedforward=2048, enc_layers=6, dec_layers=6, pre_norm=False,
        channel_point=3, aux_loss=True, masks=False, set_cost_class=2, set_cost_point=5,
        set_cost_giou=2, cls_loss_coef=2, point_loss_coef=5, giou_loss_coef=2, focal_alpha=.25)
    resnet = tv.resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    if weights is not None:
        resnet.load_state_dict(weights, strict=True)
    backbone = Joiner(BackboneBase(resnet, True, 2048, False), build_position_encoding(config))
    backbone.num_channels = 2048
    with patch("Networks.CDETR.conditional_detr.build_backbone", return_value=backbone):
        model, criterion, _ = build(config)
    criterion.matcher = PerImageMatcher(criterion.matcher, "points")
    return model, criterion


def targets_for(model, targets, shape, device):
    h, w = shape
    result = []
    for target in targets:
        xy = target["points"]
        labels = torch.ones(len(xy), dtype=torch.long, device=device)
        if model == "p2pnet":
            result.append({"point": xy.to(device), "labels": labels})
            continue
        # Public CLTR uses [y, x, mean neighbor distance] targets;
        # its 4-neighbor lookup includes the point itself before removal.
        spacing = np.zeros(len(xy), dtype=np.float32)
        if len(xy) > 1:
            distances, _ = cKDTree(xy.numpy()).query(xy.numpy(), k=min(4, len(xy)))
            spacing = distances[:, 1:].mean(axis=1).astype(np.float32)
        points = torch.cat((xy[:, [1, 0]], torch.from_numpy(spacing)[:, None]), dim=1)
        result.append({"points": (points / points.new_tensor([h, w, w])).to(device), "labels": labels})
    return result


def decode(model, outputs, shape, threshold):
    h, w = shape
    points = outputs["pred_points"][0, :, :2]
    if model == "cltr":
        points = points[:, [1, 0]] * points.new_tensor([w, h])
        scores = outputs["pred_logits"][0, :, 1].sigmoid()
    else:
        scores = outputs["pred_logits"][0].softmax(-1)[:, 1]
    keep = ((scores > threshold) & (points[:, 0] >= 0) & (points[:, 0] < w)
            & (points[:, 1] >= 0) & (points[:, 1] < h))
    return points[keep].cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("p2pnet", "cltr"))
    for key in ("upstream", "data-code", "data-path", "weights", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in ("batch-size", "epochs", "workers", "eval-every", "seed"):
        p.add_argument("--" + key, required=True, type=int)
    p.add_argument("--cltr-queries", type=int, default=500)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA is required")
    device = torch.device("cuda")
    data, metrics, ckpt = load_data_module(args.data_code)
    sys.path.insert(0, str(args.upstream))
    load_upstream_utils(args.upstream)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = data.RiceW3999(args.data_path, "train", patch_size=512)
    val = data.RiceW3999(args.data_path, "val", patch_size=512, augment=False)
    fingerprint = (train.fingerprint(), val.fingerprint())
    common = dict(num_workers=args.workers, pin_memory=True,
                  collate_fn=data.collate, worker_init_fn=data.seed_worker)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val, batch_size=1, **common)
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "checkpoint.pth"
    state = torch.load(latest, map_location="cpu", weights_only=False) if latest.is_file() else None
    if state is not None and state["fingerprint"] != fingerprint:
        raise ValueError("Training/validation inputs changed since checkpoint")
    model, criterion = build_model(args, initializing=state is None)
    model, criterion = model.to(device), criterion.to(device)
    groups = [{"params": [v for k, v in model.named_parameters() if k.startswith("backbone.") and v.requires_grad],
               "lr": 1e-5 if args.model == "p2pnet" else 1e-4},
              {"params": [v for k, v in model.named_parameters() if not k.startswith("backbone.") and v.requires_grad]}]
    optimizer = torch.optim.Adam(groups, lr=1e-4, weight_decay=5e-4 if args.model == "cltr" else 0)
    start, best = 0, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["epoch"] + 1, state["best"]
        ckpt.restore_rng(state["rng"])
        del state
    from tqdm import tqdm
    for epoch in range(start, args.epochs):
        model.train()
        bar = tqdm(train_loader, desc=f"{args.model} {epoch+1}/{args.epochs}")
        for images, targets in bar:
            targets = targets_for(args.model, targets, images.shape[-2:], device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(device, non_blocking=True))
            losses = criterion(outputs, targets)
            loss = sum(losses[k] * v for k, v in criterion.weight_dict.items() if k in losses)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            meter = metrics.PointMetrics((4, 8, 16))
            threshold = .5 if args.model == "p2pnet" else .35
            with torch.inference_mode():
                for images, targets in tqdm(val_loader, desc="Validation"):
                    target = targets[0]
                    h, w = target["size"]
                    # CLTR coordinates are normalized to the model input.
                    # Run unpadded CLTR images; P2PNet requires aligned input.
                    inputs = images[:, :, :h, :w] if args.model == "cltr" else images
                    prediction = decode(args.model, model(inputs.to(device)), (h, w), threshold)
                    meter.update(target["image_id"], prediction[:, ::-1], target["points"].numpy()[:, ::-1])
            report = meter.summary()
            payload = meter.export(args.output / "evaluation" / f"epoch_{epoch+1:04d}", report,
                                   {"epoch_index": epoch, "split": "val", "threshold": threshold})
            print(f"epoch={epoch+1} MAE={report['mae']:.3f} RMSE={report['rmse']:.3f} F1@8={report['f1_8px']:.4f}", flush=True)
            if report["mae"] < best:
                from batch_train import save_json
                best = report["mae"]
                ckpt.atomic_save({"model": model.state_dict(), "epoch": epoch}, args.output / "best_checkpoint.pth")
                save_json(args.output / "best_metrics.json", payload)
        ckpt.atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch, "best": best, "fingerprint": fingerprint,
                          "rng": ckpt.rng_state()}, latest)


if __name__ == "__main__":
    main()
