# CLARITy Fast Stable Training Pipeline
# src/models/fast_stable_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FastStableTrainer:
    """Fast, stable training pipeline optimized for speed and gradient stability"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 device: str = 'cuda',
                 learning_rate: float = 3e-5,      # Conservative LR
                 weight_decay: float = 1e-4,       # Moderate regularization
                 accumulation_steps: int = 2,      # Lower accumulation for speed
                 mixed_precision: bool = True,
                 checkpoint_dir: str = 'models/fast_stable_checkpoints',
                 patience: int = 10,
                 max_epochs: int = 30,
                 gradient_clipping: float = 0.5):  # Strong clipping for stability
        
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
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Stable optimizer (AdamW with conservative settings)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Simple, stable scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=learning_rate * 0.01
        )
        
        # Mixed precision scaler with conservative settings
        if self.mixed_precision:
            self.scaler = GradScaler(
                init_scale=2**12,  # Lower initial scale for stability
                growth_factor=1.5,  # Conservative growth
                backoff_factor=0.8  # Aggressive backoff on overflow
            )
            print("✅ Conservative mixed precision enabled")
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_aps = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Disease classes
        self.disease_classes = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        
        print(f"🚀 Fast Stable Trainer initialized:")
        print(f"   Target: 0.80+ AUC (faster training)")
        print(f"   Max epochs: {max_epochs}")
        print(f"   Gradient clipping: {gradient_clipping} (strong)")
        print(f"   Learning rate: {learning_rate} (conservative)")
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        """Calculate key metrics efficiently"""
        
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        metrics = {}
        class_aucs = []
        class_aps = []
        class_f1s = []
        
        for i in range(len(self.disease_classes)):
            if y_true[:, i].sum() == 0:
                class_aucs.append(0.5)
                class_aps.append(0.0)
                class_f1s.append(0.0)
                continue
            
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                
                class_aucs.append(auc)
                class_aps.append(ap)
                class_f1s.append(f1)
                
            except Exception:
                class_aucs.append(0.5)
                class_aps.append(0.0)
                class_f1s.append(0.0)
        
        metrics['mean_auc'] = np.mean([auc for auc in class_aucs if auc > 0])
        metrics['mean_ap'] = np.mean([ap for ap in class_aps if ap > 0])
        metrics['mean_f1'] = np.mean(class_f1s)
        metrics['overall_accuracy'] = np.mean(y_pred_binary == y_true)
        metrics['class_aucs'] = class_aucs
        
        return metrics
    
    def train_epoch(self, epoch: int) -> float:
        """Fast, stable training epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {epoch:2d}/{self.max_epochs} [Fast Training]", 
                   leave=False)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps
                
                # Check for NaN loss immediately
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ NaN/Inf loss detected at batch {batch_idx}")
                    continue
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.gradient_clipping)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ NaN/Inf loss detected at batch {batch_idx}")
                    continue
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.gradient_clipping)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'LR': f'{current_lr:.1e}',
                'Target': '0.80+ AUC'
            })
            
            if batch_idx % 25 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """Fast validation"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.mixed_precision:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_predictions.append(probs)
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.calculate_metrics(all_labels, all_predictions)
        
        self.val_losses.append(avg_loss)
        self.val_aucs.append(metrics['mean_auc'])
        self.val_aps.append(metrics['mean_ap'])
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'current_metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_fast_stable_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💎 New best model saved: AUC = {self.best_val_auc:.4f}")
    
    def train_fast_stable(self) -> float:
        """Train with speed and stability"""
        
        print(f"🎯 Starting Fast Stable Training")
        print(f"   Target: 0.80+ AUC")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate()
            
            self.scheduler.step(metrics['mean_auc'])
            
            is_best = metrics['mean_auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = metrics['mean_auc']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            self.save_checkpoint(epoch, metrics, is_best)
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:2d}/{self.max_epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"AUC: {metrics['mean_auc']:.4f} | "
                  f"F1: {metrics['mean_f1']:.4f} | "
                  f"LR: {current_lr:.1e} | "
                  f"Time: {epoch_time:.0f}s")
            
            if metrics['mean_auc'] >= 0.80:
                print(f"🎉 TARGET ACHIEVED! AUC = {metrics['mean_auc']:.4f}")
            
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⏰ Early stopping")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n🏆 Training Complete!")
        print(f"Best AUC: {self.best_val_auc:.4f}")
        print(f"Time: {total_time/3600:.2f} hours")
        
        return self.best_val_auc