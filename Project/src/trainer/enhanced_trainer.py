# CLARITy Enhanced Training Pipeline
# src/models/enhanced_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedTrainer:
    """Enhanced Training Pipeline with advanced features for 0.85+ AUC"""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 device: str = 'cuda',
                 learning_rate: float = 2e-5,
                 weight_decay: float = 1e-4,
                 accumulation_steps: int = 4,
                 mixed_precision: bool = True,
                 checkpoint_dir: str = 'models/enhanced_checkpoints',
                 patience: int = 12,
                 max_epochs: int = 60,
                 scheduler_type: str = 'onecycle',
                 gradient_clipping: float = 0.5,
                 use_mixup_cutmix: bool = True,
                 test_time_augmentation: bool = True,
                 minority_class_boost: bool = True):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.patience = patience
        self.max_epochs = max_epochs
        self.gradient_clipping = gradient_clipping
        self.use_mixup_cutmix = use_mixup_cutmix
        self.test_time_augmentation = test_time_augmentation
        self.minority_class_boost = minority_class_boost

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )

        # Scheduler
        if scheduler_type == 'onecycle':
            total_steps = max_epochs * len(train_loader) // accumulation_steps
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=15,
                T_mult=2,
                eta_min=learning_rate * 0.01
            )
        self.scheduler_type = scheduler_type

        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = GradScaler(
                init_scale=2**14,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=1000
            )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_f1s = []
        self.learning_rates = []
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.disease_classes = [
            'No Finding','Atelectasis','Cardiomegaly','Effusion',
            'Infiltration','Mass','Nodule','Pneumonia','Pneumothorax',
            'Consolidation','Edema','Emphysema','Fibrosis',
            'Pleural_Thickening','Hernia'
        ]

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()
        for images, labels, _ in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels) / self.accumulation_steps
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                self.scaler.scale(loss).backward()
                if (num_batches + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler_type == 'onecycle':
                        self.scheduler.step()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels) / self.accumulation_steps
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                if (num_batches + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler_type == 'onecycle':
                        self.scheduler.step()

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate_with_tta(self) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Val TTA", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # TTA
                outputs = []
                for flip in [None, 'h', 'v']:
                    imgs = images
                    if flip == 'h':
                        imgs = torch.flip(images, [3])
                    if flip == 'v':
                        imgs = torch.flip(images, [2])
                    with autocast() if self.mixed_precision else torch.no_grad():
                        out = self.model(imgs)
                    outputs.append(torch.sigmoid(out))
                probs = torch.mean(torch.stack(outputs), 0)
                loss = self.criterion(probs, labels) if self.mixed_precision else self.criterion(self.model(images), labels)
                total_loss += loss.item()
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        avg_loss = total_loss / len(self.val_loader)
        preds = np.concatenate(all_preds)
        labs = np.concatenate(all_labels)
        metrics = self._calc_metrics(labs, preds)
        self.val_losses.append(avg_loss)
        self.val_aucs.append(metrics['mean_auc'])
        self.val_f1s.append(metrics['macro_f1'])
        return avg_loss, metrics

    def _calc_metrics(self, y_true, y_pred, thr=0.5) -> Dict:
        y_bin = (y_pred >= thr).astype(int)
        class_aucs, class_f1s = [], []
        for i in range(len(self.disease_classes)):
            if y_true[:, i].sum() == 0:
                class_aucs.append(0.5)
                class_f1s.append(0.0)
                continue
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                f1 = f1_score(y_true[:, i], y_bin[:, i], zero_division=0)
            except:
                auc, f1 = 0.5, 0.0
            class_aucs.append(auc)
            class_f1s.append(f1)
        return {
            'mean_auc': np.mean(class_aucs),
            'macro_f1': np.mean(class_f1s)
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        path = self.checkpoint_dir / f'checkpoint_{epoch:03d}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'best_val_f1': self.best_val_f1
        }, path)
        if is_best:
            torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model.pth')

    def train_for_excellence(self) -> Tuple[float, float]:
        print("🎯 Starting Enhanced Training for Excellence")
        start = time.time()
        for epoch in range(1, self.max_epochs + 1):
            tl = self.train_epoch(epoch)
            vl, metrics = self.validate_with_tta()
            auc, f1 = metrics['mean_auc'], metrics['macro_f1']
            is_best = auc > self.best_val_auc or (auc == self.best_val_auc and f1 > self.best_val_f1)
            if is_best:
                self.best_val_auc, self.best_val_f1, self.best_epoch = auc, f1, epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            self.save_checkpoint(epoch, is_best)
            print(f"Epoch {epoch}/{self.max_epochs} | TLoss={tl:.4f} VLoss={vl:.4f} AUC={auc:.4f} F1={f1:.4f}")
            if self.epochs_without_improvement >= self.patience:
                print("⏰ Early stopping")
                break
        print(f"🏆 Training complete: Best AUC={self.best_val_auc:.4f}, F1={self.best_val_f1:.4f} at epoch {self.best_epoch}")
        return self.best_val_auc, self.best_val_f1