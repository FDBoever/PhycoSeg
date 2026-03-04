#!/usr/bin/env python3
"""
Seaweed Segmentation – U-Net (PyTorch)
--------------------------------------
training and inference pipeline for binary segmentation.

Features
- Minimal U-Net
- Dataset class expecting matching image/mask filenames
- Training and validation steps are split, on-the-fly augmentations, mixed precision
- BCEWithLogits + Dice loss combo; metrics: Dice, IoU, Precision, Recall
- (Optional) Checkpointing on best Val Dice for early stopping.
- Threshold sweep on Val to pick optimal probability threshold (streaming; no big intermediates)
- Inference with optional TTA (flips)

Usage examples
--------------
# train (no saving to disk)
python seaweed_unet.py train \
  --data_dir data \
  --out_dir runs/seaweed_v1 \
  --epochs 60 \
  --img_size 768 768 \
  --batch_size 4 \
  --lr 3e-4 \
  --val_split 0.15 \
  --no_save

# train (save best)
python seaweed_unet.py train \
  --data_dir data \
  --out_dir runs/seaweed_v1 \
  --epochs 60 \
  --img_size 768 768 \
  --batch_size 4 \
  --lr 3e-4 \
  --val_split 0.15

# Inference
python seaweed_unet.py infer \
  --ckpt runs/seaweed_v1/best.pt \
  --threshold_json runs/seaweed_v1/best_threshold.json \
  --input_dir data/images \
  --output_dir runs/seaweed_v1/preds
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

try:
    import cv2  # optional; only used if you prefer cv2 imwrite
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# U-Net model
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.up1 = Up(base_ch*16, base_ch*4)
        self.up2 = Up(base_ch*8, base_ch*2)
        self.up3 = Up(base_ch*4, base_ch)
        self.up4 = Up(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, n_classes, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Dataset
class SegDataset(Dataset):
    def __init__(self, root: str | Path, img_size: Tuple[int, int] = (512, 512),
                 augment: bool = True):
        root = Path(root)
        self.img_dir = root / 'images'
        self.mask_dir = root / 'masks'
        assert self.img_dir.exists() and self.mask_dir.exists(), "images/ and masks/ must exist"

        self.items = []
        for p in sorted(self.img_dir.iterdir()):
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                m = self.mask_dir / (p.stem + '.png')
                if not m.exists():
                    m_alt = self.mask_dir / p.name
                    if m_alt.exists():
                        m = m_alt
                if m.exists():
                    self.items.append((p, m))
        if len(self.items) == 0:
            raise RuntimeError("No paired image/mask files found.")

        self.img_size = img_size
        self.augment = augment

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.items)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _load_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert('L')

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        target_size = self.img_size
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)

        img_t = self.to_tensor(img)
        mask_np = np.array(mask)
        mask_bin = (mask_np > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)  # (1,H,W)
        return img_t, mask_t

# Loss and metrics
def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (denom + eps)).mean()

def iou_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1,2,3))
        denom = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2*inter + 1e-6) / (denom + 1e-6)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss

# Streaming threshold sweep
def sweep_best_threshold(model: nn.Module, loader: DataLoader, device: torch.device,
                         thr_min=0.2, thr_max=0.9, steps=36) -> tuple[float, float]:
    model.eval()
    thresholds = torch.linspace(thr_min, thr_max, steps, device=device)
    # running mean dice for each threshold
    dice_sums = torch.zeros(steps, device=device)
    count = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.sigmoid(model(imgs))  # (B,1,H,W)
            # evaluate per-threshold without storing across batches
            for i, thr in enumerate(thresholds):
                preds = (probs >= thr).float()
                dice_sums[i] += dice_coef(preds, masks)
            count += 1
    mean_dice = dice_sums / max(1, count)
    best_idx = int(torch.argmax(mean_dice).item())
    best_thr = float(thresholds[best_idx].item())
    best_dice = float(mean_dice[best_idx].item())
    return best_thr, best_dice

def eval_at_threshold(model: nn.Module, loader: DataLoader, device: torch.device, thr: float):
    model.eval()
    criterion = BCEDiceLoss()
    n = 0
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            loss = criterion(logits, masks)
            preds = (probs >= thr).float()

            loss_sum += float(loss.item()) * imgs.size(0)
            dice_sum += float(dice_coef(preds, masks).item()) * imgs.size(0)
            iou_sum  += float(iou_coef(preds, masks).item())  * imgs.size(0)

            tp = (preds * masks).sum().item()
            fp = (preds * (1 - masks)).sum().item()
            fn = ((1 - preds) * masks).sum().item()
            tp_sum += tp; fp_sum += fp; fn_sum += fn

            n += imgs.size(0)

    precision = tp_sum / (tp_sum + fp_sum + 1e-6)
    recall    = tp_sum / (tp_sum + fn_sum + 1e-6)
    return (loss_sum / max(1, n),
            dice_sum / max(1, n),
            iou_sum  / max(1, n),
            precision, recall)

# training and validation
def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_ds = SegDataset(args.data_dir, img_size=tuple(args.img_size), augment=True)
    val_len = max(1, int(len(full_ds) * args.val_split))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(args.seed))
    val_ds.dataset.augment = False

    # keep workers modest to avoid Mac I/O pain
    num_workers = max(0, min(args.workers, 2))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = UNet(n_channels=3, n_classes=1, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5)
    criterion = BCEDiceLoss(bce_weight=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.out_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    best_thr = 0.5
    epochs_no_improve = 0

    print(f"Device: {device}; Train: {len(train_ds)} images; Val: {len(val_ds)} images")

    for epoch in range(1, args.epochs + 1):
        #Train
        model.train()
        train_loss = 0.0
        seen = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            train_loss += loss.item() * bs
            seen += bs

        train_loss /= max(1, seen)

        #find best threshold on val (streaming, no storage)
        thr, _ = sweep_best_threshold(model, val_loader, device)

        val_loss, dice, iou, precision, recall = eval_at_threshold(model, val_loader, device, thr)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"dice {dice:.4f} | iou {iou:.4f} | thr {thr:.3f} | "
              f"prec {precision:.3f} | rec {recall:.3f}")

        # Save best
        improved = dice > best_dice + 1e-6
        if improved:
            best_dice = dice
            best_thr = thr
            epochs_no_improve = 0
            if not args.no_save:
                try:
                    ckpt = {
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'best_dice': best_dice,
                        'threshold': best_thr,
                        'img_size': args.img_size,
                        'base_ch': args.base_ch,
                    }
                    torch.save(ckpt, out_dir / 'best.pt')
                    with open(out_dir / 'best_threshold.json', 'w') as f:
                        json.dump({'threshold': best_thr}, f)
                except Exception as e:
                    print(f"[WARN] Failed to save checkpoint: {e}")
        else:
            epochs_no_improve += 1

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

    print(f"Best Val Dice: {best_dice:.4f} at thr {best_thr:.3f}")
    if args.no_save:
        print("Note: --no_save was set, so no checkpoint was written.")

# inference
def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[UNet, float, Tuple[int,int]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    base_ch = ckpt.get('base_ch', 64)
    model = UNet(n_channels=3, n_classes=1, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt['model'])
    thr = float(ckpt.get('threshold', 0.5))
    img_size = tuple(ckpt.get('img_size', (512, 512)))
    model.eval()
    return model, thr, img_size

def preprocess_image(path: Path, size: Tuple[int,int]) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return t(img)

def tta_flips_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x); p0 = torch.sigmoid(logits)
        logits_h = model(torch.flip(x, dims=[-1])); p1 = torch.flip(torch.sigmoid(logits_h), dims=[-1])
        logits_v = model(torch.flip(x, dims=[-2])); p2 = torch.flip(torch.sigmoid(logits_v), dims=[-2])
        logits_hv = model(torch.flip(x, dims=[-1,-2])); p3 = torch.flip(torch.sigmoid(logits_hv), dims=[-1,-2])
        return (p0 + p1 + p2 + p3) / 4.0

def save_mask(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_CV2:
        cv2.imwrite(str(out_path), mask)
    else:
        Image.fromarray(mask).save(out_path)

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, thr_ckpt, img_size = load_model(args.ckpt, device)
    threshold = thr_ckpt
    if args.threshold_json and Path(args.threshold_json).exists():
        with open(args.threshold_json, 'r') as f:
            threshold = float(json.load(f).get('threshold', thr_ckpt))
    if args.threshold is not None:
        threshold = float(args.threshold)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    imgs = [p for p in input_dir.iterdir()
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
    if len(imgs) == 0:
        raise RuntimeError("No images found in input_dir")

    for p in imgs:
        x = preprocess_image(p, size=tuple(img_size)).unsqueeze(0).to(device)
        if args.tta:
            probs = tta_flips_predict(model, x)
        else:
            with torch.no_grad():
                probs = torch.sigmoid(model(x))
        prob = probs.squeeze().cpu().numpy()
        mask = (prob >= threshold).astype(np.uint8) * 255
        save_mask(mask, output_dir / (p.stem + '_mask.png'))

        if args.save_overlay:
            orig = Image.open(p).convert('RGB').resize(tuple(img_size), Image.BILINEAR)
            orig_np = np.array(orig)
            overlay = orig_np.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            # cv2 expects BGR; PIL expects RGB
            save_mask(overlay[:, :, ::-1] if _HAS_CV2 else overlay, output_dir / (p.stem + '_overlay.png'))

    print(f"Saved masks to {output_dir}")

# CLI
def build_argparser():
    p = argparse.ArgumentParser(description="Seaweed U-Net segmentation")
    sub = p.add_subparsers(dest='cmd', required=True)

    # train
    pt = sub.add_parser('train')
    pt.add_argument('--data_dir', type=str, required=True)
    pt.add_argument('--out_dir', type=str, required=True)
    pt.add_argument('--img_size', type=int, nargs=2, default=[512, 512])
    pt.add_argument('--batch_size', type=int, default=4)
    pt.add_argument('--epochs', type=int, default=60)
    pt.add_argument('--val_split', type=float, default=0.15)
    pt.add_argument('--lr', type=float, default=3e-4)
    pt.add_argument('--base_ch', type=int, default=64)
    pt.add_argument('--workers', type=int, default=4)
    pt.add_argument('--seed', type=int, default=42)
    pt.add_argument('--amp', action='store_true', help='enable mixed precision')
    pt.add_argument('--early_stop', type=int, default=10, help='epochs without improvement to stop (0=off)')
    pt.add_argument('--no_save', action='store_true', help='do not write checkpoints or threshold files')

    # infer
    pi = sub.add_parser('infer')
    pi.add_argument('--ckpt', type=str, required=True)
    pi.add_argument('--threshold_json', type=str, default=None)
    pi.add_argument('--threshold', type=float, default=None, help='override threshold')
    pi.add_argument('--input_dir', type=str, required=True)
    pi.add_argument('--output_dir', type=str, required=True)
    pi.add_argument('--tta', action='store_true')
    pi.add_argument('--save_overlay', action='store_true')

    return p

def main():
    args = build_argparser().parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'infer':
        infer(args)

if __name__ == '__main__':
    main()
