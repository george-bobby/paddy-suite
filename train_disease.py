"""
train_disease.py — Module 2: Paddy Disease Classification
EfficientNet-B3 fine-tuned in 2 stages on the Kaggle paddy-disease-classification dataset.

Run:  python train_disease.py
Skip: Automatically skipped if models/disease/best.pth already exists.
"""

import os
import pickle
import json
import zipfile
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler as AmpGradScaler, autocast
import torchvision.transforms as T
from PIL import Image
import timm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

import config

warnings.filterwarnings('ignore')


# ──────────────────────── Dataset class ──────────────────────────────────────

class PaddyDiseaseDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        self.df = df
        self.img_dir  = Path(img_dir)
        self.transform  = transform
        self.is_test    = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = (self.img_dir / row['image_id'] if self.is_test
                else self.img_dir / row['label'] / row['image_id'])
        img  = Image.open(path).convert('RGB')
        return self.transform(img), (row['image_id'] if self.is_test else int(row['label_enc']))


# ──────────────────────── Transform builders ──────────────────────────────────

def train_tfm(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.1), T.RandomRotation(15),
        T.ToTensor(), T.Normalize(config.IMG_MEAN, config.IMG_STD),
    ])

def val_tfm(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(config.IMG_MEAN, config.IMG_STD)])


# ──────────────────────── Train / val epochs ──────────────────────────────────

def _train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    amp_scaler,
    scheduler,
    stage_label,
    epoch_idx,
    total_epochs,
    progress_every,
    use_amp,
):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    total_batches = len(loader)
    start_time = time.time()

    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        loss_sum += loss.item() * labels.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)

        should_log = (
            batch_idx == 1
            or batch_idx == total_batches
            or (progress_every > 0 and batch_idx % progress_every == 0)
        )
        if should_log:
            elapsed = max(time.time() - start_time, 1e-9)
            batches_per_sec = batch_idx / elapsed
            eta_sec = (total_batches - batch_idx) / max(batches_per_sec, 1e-9)
            avg_loss = loss_sum / max(total, 1)
            avg_acc = correct / max(total, 1)
            pct = 100.0 * batch_idx / max(total_batches, 1)
            lr = optimizer.param_groups[0]['lr']
            print(
                f'    [{stage_label}][Ep {epoch_idx:02d}/{total_epochs}] '
                f'Batch {batch_idx:03d}/{total_batches} ({pct:5.1f}%) | '
                f'loss {avg_loss:.4f} acc {avg_acc:.4f} lr {lr:.2e} | ETA {eta_sec:6.1f}s'
            )

    return loss_sum / total, correct / total


@torch.no_grad()
def _val_epoch(model, loader, criterion, use_amp):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        if use_amp:
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
        else:
            out = model(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * labels.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


# ──────────────────────── Download ───────────────────────────────────────────

def _download_dataset():
    train_csv = Path(config.DISEASE_CSV)
    data_dir  = Path(config.DISEASE_DATA_DIR)
    zip_file  = Path('paddy-disease-classification.zip')
    
    # If data already extracted, skip
    if train_csv.exists() and (data_dir / 'train_images').exists():
        print(f'  ✅ Disease dataset already exists.')
        return
    
    # If zip exists but not extracted, extract it
    if zip_file.exists():
        print(f'  📦 Extracting existing {zip_file}...')
        data_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(str(data_dir))
        print('  ✅ Disease dataset extracted!')
        return

    # Otherwise download from Kaggle
    print('  📥 Downloading paddy-disease-classification competition data...')
    ret = os.system('kaggle competitions download -c paddy-disease-classification -q')
    if ret != 0:
        raise RuntimeError(
            'Kaggle download failed. Make sure you have accepted the competition rules at '
            'https://www.kaggle.com/competitions/paddy-disease-classification/rules'
        )

    # Extract the downloaded zip
    if zip_file.exists():
        data_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(str(data_dir))
        print('  ✅ Disease dataset extracted!')


def _build_disease_model(num_classes):
    use_pretrained = bool(getattr(config, 'DISEASE_USE_PRETRAINED', True))
    if use_pretrained:
        try:
            print('  📦 Initializing EfficientNet-B3 with pretrained weights...')
            model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
            print('  ✅ Pretrained weights loaded.')
            return model
        except Exception as exc:
            print(f'  ⚠️ Pretrained weights unavailable ({exc}). Using random initialization.')
    else:
        print('  ℹ️ Pretrained weights disabled for this environment (set DISEASE_USE_PRETRAINED=1 to enable).')

    return timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)


# ──────────────────────── Main ───────────────────────────────────────────────

def train():
    """Full disease training pipeline. Skipped if model already saved."""
    os.makedirs(config.MODEL_DIR_DISEASE, exist_ok=True)

    if os.path.exists(config.DISEASE_MODEL_PATH):
        print('⏭️  Disease model already trained — skipping. Delete models/disease/ to retrain.')
        return

    print('\n' + '='*60)
    print('🍄  MODULE 2 — Paddy Disease Classification (EfficientNet-B3)')
    print('='*60)
    print(f'  Device: {config.DEVICE}')
    use_amp = config.DEVICE == 'cuda'
    pin_memory = bool(getattr(config, 'DISEASE_PIN_MEMORY', config.DEVICE == 'cuda'))
    progress_every = max(1, int(getattr(config, 'DISEASE_PROGRESS_EVERY', 10)))

    img1 = config.IMG1
    img2 = config.IMG2
    bs1 = config.BS1
    bs2 = config.BS2
    epochs1 = config.EPOCHS1
    epochs2 = config.EPOCHS2

    fast_mode = bool(getattr(config, 'DISEASE_FAST_MODE', False))
    if fast_mode:
        # CPU-friendly defaults; can still be overridden via env vars.
        img2 = min(img2, 224)
        bs1 = min(bs1, 32)
        bs2 = min(bs2, 16)
        epochs1 = min(epochs1, 3)
        epochs2 = min(epochs2, 1)
        print('  ⚡ Fast mode enabled: smaller batches/epochs for quicker CPU training.')

    print(
        f'  Runtime config: workers={config.DISEASE_DATALOADER_WORKERS} '
        f'pin_memory={pin_memory} amp={use_amp} progress_every={progress_every} batches'
    )
    print(f'  Stage plan: S1={img1}px x {epochs1} ep (bs={bs1}) | S2={img2}px x {epochs2} ep (bs={bs2})')

    # 1. Download
    _download_dataset()

    # 2. Load CSV
    df = pd.read_csv(config.DISEASE_CSV)
    le = LabelEncoder()
    df['label_enc']   = le.fit_transform(df['label'])
    num_classes       = df['label_enc'].nunique()

    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df['label_enc'], random_state=config.SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f'  Train: {len(train_df)} | Val: {len(val_df)} | Classes: {num_classes}')

    train_dir = Path(config.DISEASE_TRAIN_DIR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 3. Stage 1 — head only warm-up (224px, 6 epochs)
    print(f'\n  --- Stage 1: Head-Only Warm-Up | {img1}px | {epochs1} epochs ---')
    model = _build_disease_model(num_classes)
    model = model.to(config.DEVICE)

    for name, p in model.named_parameters():
        p.requires_grad = ('classifier' in name)

    tr_ds1  = PaddyDiseaseDataset(train_df, train_dir, train_tfm(img1))
    val_ds1 = PaddyDiseaseDataset(val_df,   train_dir, val_tfm(img1))
    tr_ld1  = DataLoader(
        tr_ds1,
        batch_size=bs1,
        shuffle=True,
        num_workers=config.DISEASE_DATALOADER_WORKERS,
        pin_memory=pin_memory,
    )
    val_ld1 = DataLoader(
        val_ds1,
        batch_size=bs1 * 2,
        shuffle=False,
        num_workers=config.DISEASE_DATALOADER_WORKERS,
        pin_memory=pin_memory,
    )

    opt1  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=config.LR1, weight_decay=1e-4)
    sch1  = optim.lr_scheduler.OneCycleLR(opt1, max_lr=config.LR1,
                steps_per_epoch=len(tr_ld1), epochs=epochs1, pct_start=0.3)
    scaler1 = AmpGradScaler(enabled=use_amp)

    best_acc1 = 0.0
    for ep in range(1, epochs1 + 1):
        epoch_start = time.time()
        tr_loss, tr_acc = _train_epoch(
            model,
            tr_ld1,
            opt1,
            criterion,
            scaler1,
            sch1,
            stage_label='S1',
            epoch_idx=ep,
            total_epochs=epochs1,
            progress_every=progress_every,
            use_amp=use_amp,
        )
        vl_loss, vl_acc = _val_epoch(model, val_ld1, criterion, use_amp=use_amp)
        tag = ''
        if vl_acc > best_acc1:
            best_acc1 = vl_acc
            torch.save(model.state_dict(), config.DISEASE_MODEL_PATH)
            tag = ' ← best'
        print(
            f'  Ep {ep:02d}/{epochs1} done in {time.time() - epoch_start:.1f}s | '
            f'tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | '
            f'val_loss {vl_loss:.4f} val_acc {vl_acc:.4f}{tag}'
        )
    print(f'\n  ✅ Stage 1 best val acc: {best_acc1:.4f}')

    # 4. Stage 2 — full fine-tune (320px, 3 epochs)
    print(f'\n  --- Stage 2: Full Fine-Tune | {img2}px | {epochs2} epochs ---')
    model.load_state_dict(torch.load(config.DISEASE_MODEL_PATH, map_location=config.DEVICE))
    for p in model.parameters():
        p.requires_grad = True

    tr_ds2  = PaddyDiseaseDataset(train_df, train_dir, train_tfm(img2))
    val_ds2 = PaddyDiseaseDataset(val_df,   train_dir, val_tfm(img2))
    tr_ld2  = DataLoader(
        tr_ds2,
        batch_size=bs2,
        shuffle=True,
        num_workers=config.DISEASE_DATALOADER_WORKERS,
        pin_memory=pin_memory,
    )
    val_ld2 = DataLoader(
        val_ds2,
        batch_size=bs2 * 2,
        shuffle=False,
        num_workers=config.DISEASE_DATALOADER_WORKERS,
        pin_memory=pin_memory,
    )

    opt2    = optim.AdamW(model.parameters(), lr=config.LR2, weight_decay=1e-4)
    sch2    = optim.lr_scheduler.OneCycleLR(opt2, max_lr=config.LR2,
                  steps_per_epoch=len(tr_ld2), epochs=epochs2, pct_start=0.1)
    scaler2 = AmpGradScaler(enabled=use_amp)

    best_acc2 = 0.0
    for ep in range(1, epochs2 + 1):
        epoch_start = time.time()
        tr_loss, tr_acc = _train_epoch(
            model,
            tr_ld2,
            opt2,
            criterion,
            scaler2,
            sch2,
            stage_label='S2',
            epoch_idx=ep,
            total_epochs=epochs2,
            progress_every=progress_every,
            use_amp=use_amp,
        )
        vl_loss, vl_acc = _val_epoch(model, val_ld2, criterion, use_amp=use_amp)
        tag = ''
        if vl_acc > best_acc2:
            best_acc2 = vl_acc
            torch.save(model.state_dict(), config.DISEASE_MODEL_PATH)
            tag = ' ← best'
        print(
            f'  Ep {ep:02d}/{epochs2} done in {time.time() - epoch_start:.1f}s | '
            f'tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | '
            f'val_loss {vl_loss:.4f} val_acc {vl_acc:.4f}{tag}'
        )
    print(f'\n  ✅ Stage 2 best val acc: {best_acc2:.4f}')

    # 5. Evaluate
    model.load_state_dict(torch.load(config.DISEASE_MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_ld2:
            with autocast():
                preds = model(imgs.to(config.DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    print(f'\n  Validation Accuracy: {val_acc:.2f}%')

    # 6. Save label encoder + config
    with open(config.DISEASE_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    disease_cfg = {
        'model_name'  : 'efficientnet_b3',
        'num_classes' : num_classes,
        'img_size'    : img2,
        'classes'     : list(le.classes_),
        'val_accuracy': round(val_acc, 4),
    }
    with open(config.DISEASE_CONFIG_PATH, 'w') as f:
        json.dump(disease_cfg, f, indent=2)

    print(f'  💾 Disease model saved → {config.DISEASE_MODEL_PATH}')
    print(f'  ✅ Done! (EfficientNet-B3  val acc={val_acc:.2f}%)')


if __name__ == '__main__':
    from setup_kaggle import setup_kaggle
    setup_kaggle()
    train()
