"""
train_disease.py — Module 2: Paddy Disease Classification
EfficientNet-B3 fine-tuned in 2 stages on the Kaggle paddy-disease-classification dataset.

Run:  python train_disease.py
Skip: Automatically skipped if saved_models/disease/best.pth already exists.
"""

import os
import pickle
import json
import zipfile
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

def _train_epoch(model, loader, optimizer, criterion, amp_scaler, scheduler):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp_scaler.step(optimizer); amp_scaler.update()
        scheduler.step()
        loss_sum += loss.item() * labels.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def _val_epoch(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * labels.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


# ──────────────────────── Download ───────────────────────────────────────────

def _download_dataset():
    train_csv = Path(config.DISEASE_CSV)
    if train_csv.exists():
        print(f'  ✅ Disease dataset already exists.')
        return
    print('  📥 Downloading paddy-disease-classification competition data...')

    # Try kaggle CLI first
    ret = os.system('kaggle competitions download -c paddy-disease-classification -q')
    if ret != 0:
        raise RuntimeError(
            'Kaggle download failed. Make sure you have accepted the competition rules at '
            'https://www.kaggle.com/competitions/paddy-disease-classification/rules'
        )

    Path(config.DISEASE_DATA_DIR).mkdir(exist_ok=True)
    zip_file = 'paddy-disease-classification.zip'
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(config.DISEASE_DATA_DIR)
        os.remove(zip_file)
    print('  ✅ Disease dataset extracted!')


# ──────────────────────── Main ───────────────────────────────────────────────

def train():
    """Full disease training pipeline. Skipped if model already saved."""
    os.makedirs(config.MODEL_DIR_DISEASE, exist_ok=True)

    if os.path.exists(config.DISEASE_MODEL_PATH):
        print('⏭️  Disease model already trained — skipping. Delete saved_models/disease/ to retrain.')
        return

    print('\n' + '='*60)
    print('🍄  MODULE 2 — Paddy Disease Classification (EfficientNet-B3)')
    print('='*60)
    print(f'  Device: {config.DEVICE}')

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
    print(f'\n  --- Stage 1: Head-Only Warm-Up | {config.IMG1}px | {config.EPOCHS1} epochs ---')
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
    model = model.to(config.DEVICE)

    for name, p in model.named_parameters():
        p.requires_grad = ('classifier' in name)

    tr_ds1  = PaddyDiseaseDataset(train_df, train_dir, train_tfm(config.IMG1))
    val_ds1 = PaddyDiseaseDataset(val_df,   train_dir, val_tfm(config.IMG1))
    tr_ld1  = DataLoader(tr_ds1,  batch_size=config.BS1,   shuffle=True,  num_workers=2, pin_memory=True)
    val_ld1 = DataLoader(val_ds1, batch_size=config.BS1*2, shuffle=False, num_workers=2, pin_memory=True)

    opt1  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=config.LR1, weight_decay=1e-4)
    sch1  = optim.lr_scheduler.OneCycleLR(opt1, max_lr=config.LR1,
                steps_per_epoch=len(tr_ld1), epochs=config.EPOCHS1, pct_start=0.3)
    scaler1 = AmpGradScaler()

    best_acc1 = 0.0
    for ep in range(1, config.EPOCHS1 + 1):
        tr_loss, tr_acc = _train_epoch(model, tr_ld1, opt1, criterion, scaler1, sch1)
        vl_loss, vl_acc = _val_epoch(model, val_ld1, criterion)
        tag = ''
        if vl_acc > best_acc1:
            best_acc1 = vl_acc
            torch.save(model.state_dict(), config.DISEASE_MODEL_PATH)
            tag = ' ← best'
        print(f'  Ep {ep:02d}/{config.EPOCHS1} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | val_acc {vl_acc:.4f}{tag}')
    print(f'\n  ✅ Stage 1 best val acc: {best_acc1:.4f}')

    # 4. Stage 2 — full fine-tune (320px, 3 epochs)
    print(f'\n  --- Stage 2: Full Fine-Tune | {config.IMG2}px | {config.EPOCHS2} epochs ---')
    model.load_state_dict(torch.load(config.DISEASE_MODEL_PATH, map_location=config.DEVICE))
    for p in model.parameters():
        p.requires_grad = True

    tr_ds2  = PaddyDiseaseDataset(train_df, train_dir, train_tfm(config.IMG2))
    val_ds2 = PaddyDiseaseDataset(val_df,   train_dir, val_tfm(config.IMG2))
    tr_ld2  = DataLoader(tr_ds2,  batch_size=config.BS2,   shuffle=True,  num_workers=2, pin_memory=True)
    val_ld2 = DataLoader(val_ds2, batch_size=config.BS2*2, shuffle=False, num_workers=2, pin_memory=True)

    opt2    = optim.AdamW(model.parameters(), lr=config.LR2, weight_decay=1e-4)
    sch2    = optim.lr_scheduler.OneCycleLR(opt2, max_lr=config.LR2,
                  steps_per_epoch=len(tr_ld2), epochs=config.EPOCHS2, pct_start=0.1)
    scaler2 = AmpGradScaler()

    best_acc2 = 0.0
    for ep in range(1, config.EPOCHS2 + 1):
        tr_loss, tr_acc = _train_epoch(model, tr_ld2, opt2, criterion, scaler2, sch2)
        vl_loss, vl_acc = _val_epoch(model, val_ld2, criterion)
        tag = ''
        if vl_acc > best_acc2:
            best_acc2 = vl_acc
            torch.save(model.state_dict(), config.DISEASE_MODEL_PATH)
            tag = ' ← best'
        print(f'  Ep {ep:02d}/{config.EPOCHS2} | tr_acc {tr_acc:.4f} | val_acc {vl_acc:.4f}{tag}')
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
        'img_size'    : config.IMG2,
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
