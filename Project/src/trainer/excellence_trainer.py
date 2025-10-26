# CLARITy Excellence Training Pipeline
# src/models/excellence_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ExcellenceTrainer:
    """Excellence Training Pipeline for 0.85+ AUC target"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 accumulation_steps: int = 4,
                 mixed_precision: bool = True,
                 checkpoint_dir: str = 'models/excellence_checkpoints',
                 patience: int = 15,
                 max_epochs: int = 100,
                 scheduler_type: str = 'cosine_warm_restarts',
                 warmup_epochs: int = 5,
                 test_time_augmentation: bool = True,
                 gradient_clipping: float = 1.0):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.patience = patience
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.test_time_augmentation = test_time_augmentation
        self.gradient_clipping = gradient_clipping
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Excellence optimizer with advanced settings
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # Enhanced AdamW
        )
        
        # Advanced learning rate scheduling
        self.scheduler_type = scheduler_type
        if scheduler_type == 'cosine_warm_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,  # First restart after 20 epochs
                T_mult=1,  # Restart period multiplier
                eta_min=learning_rate * 0.001
            )
        elif scheduler_type == 'onecycle':
            total_steps = max_epochs * len(train_loader) // accumulation_steps
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate * 5,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
        else:  # plateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=learning_rate * 0.0001
            )
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = GradScaler()
            print("✅ Mixed precision training enabled (FP16)")
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_aps = []
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
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
        
        print(f"🚀 Excellence Trainer initialized:")
        print(f"   Target: 0.85+ AUC")
        print(f"   Max epochs: {max_epochs}")
        print(f"   Batch accumulation: {accumulation_steps}")
        print(f"   Scheduler: {scheduler_type}")
        print(f"   Test-time augmentation: {test_time_augmentation}")
    
    def apply_test_time_augmentation(self, images):
        """Apply test-time augmentation for better performance"""
        if not self.test_time_augmentation:
            return self.model(images)
        
        # Original
        outputs = []
        outputs.append(self.model(images))
        
        # Horizontal flip
        outputs.append(self.model(torch.flip(images, dims=[3])))
        
        # Vertical flip
        outputs.append(self.model(torch.flip(images, dims=[2])))
        
        # Average predictions
        final_output = torch.mean(torch.stack(outputs), dim=0)
        return final_output
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, threshold=0.5):
        """Calculate comprehensive metrics including IoU/mIoU"""
        
        # Convert to binary
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        metrics = {}
        
        # Per-class metrics
        class_aucs = []
        class_aps = []
        class_f1s = []
        class_precisions = []
        class_recalls = []
        class_ious = []
        
        for i in range(len(self.disease_classes)):
            # Skip if no positive samples
            if y_true[:, i].sum() == 0:
                class_aucs.append(0.5)
                class_aps.append(0.0)
                class_f1s.append(0.0)
                class_precisions.append(0.0)
                class_recalls.append(0.0)
                class_ious.append(0.0)
                continue
            
            try:
                # AUC and AP
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                
                # Classification metrics
                f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                precision = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                recall = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
                
                # IoU calculation
                tp = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1))
                fp = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1))
                fn = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0))
                
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                
                class_aucs.append(auc)
                class_aps.append(ap)
                class_f1s.append(f1)
                class_precisions.append(precision)
                class_recalls.append(recall)
                class_ious.append(iou)
                
            except Exception as e:
                class_aucs.append(0.5)
                class_aps.append(0.0)
                class_f1s.append(0.0)
                class_precisions.append(0.0)
                class_recalls.append(0.0)
                class_ious.append(0.0)
        
        # Calculate means
        metrics['mean_auc'] = np.mean([auc for auc in class_aucs if auc > 0])
        metrics['mean_ap'] = np.mean([ap for ap in class_aps if ap > 0])
        metrics['mean_f1'] = np.mean(class_f1s)
        metrics['mean_precision'] = np.mean(class_precisions)
        metrics['mean_recall'] = np.mean(class_recalls)
        metrics['mean_iou'] = np.mean(class_ious)
        
        # Overall accuracy
        metrics['overall_accuracy'] = np.mean(y_pred_binary == y_true)
        
        # Class-wise metrics
        metrics['class_aucs'] = class_aucs
        metrics['class_aps'] = class_aps
        metrics['class_f1s'] = class_f1s
        metrics['class_precisions'] = class_precisions
        metrics['class_recalls'] = class_recalls
        metrics['class_ious'] = class_ious
        
        return metrics
    
    def train_epoch(self, epoch: int) -> float:
        """Enhanced training epoch for excellence performance"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        torch.cuda.empty_cache()
        
        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {epoch:3d}/{self.max_epochs} [Excellence Training]", 
                   leave=False)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision training
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.gradient_clipping)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler_type == 'onecycle':
                        self.scheduler.step()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.gradient_clipping)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler_type == 'onecycle':
                        self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            current_loss = loss.item() * self.accumulation_steps
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'LR': f'{current_lr:.2e}',
                'Target': '0.85+ AUC'
            })
            
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """Enhanced validation with comprehensive metrics"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for images, labels, metadata in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass with TTA if enabled
                if self.mixed_precision:
                    with autocast():
                        if self.test_time_augmentation:
                            logits = self.apply_test_time_augmentation(images)
                        else:
                            logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    if self.test_time_augmentation:
                        logits = self.apply_test_time_augmentation(images)
                    else:
                        logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Convert to probabilities and store
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                
                all_predictions.append(probs)
                all_labels.append(labels_cpu)
        
        # Concatenate predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate comprehensive metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.calculate_comprehensive_metrics(all_labels, all_predictions)
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_aucs.append(metrics['mean_auc'])
        self.val_aps.append(metrics['mean_ap'])
        self.val_f1s.append(metrics['mean_f1'])
        self.val_precisions.append(metrics['mean_precision'])
        self.val_recalls.append(metrics['mean_recall'])
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save comprehensive checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            
            # Performance metrics
            'best_val_auc': self.best_val_auc,
            'current_metrics': metrics,
            
            # Training history
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'val_aps': self.val_aps,
            'val_f1s': self.val_f1s,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'learning_rates': self.learning_rates
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_excellence_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💎 New best model saved: AUC = {self.best_val_auc:.4f}")
    
    def train_for_excellence(self) -> float:
        """Train for excellence performance (0.85+ AUC)"""
        
        print(f"🎯 Starting Excellence Training for 0.85+ AUC")
        print(f"   Max epochs: {self.max_epochs}")
        print(f"   Early stopping patience: {self.patience}")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            
            # Training step
            train_loss = self.train_epoch(epoch)
            
            # Validation step
            val_loss, metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler_type == 'cosine_warm_restarts':
                self.scheduler.step()
            elif self.scheduler_type == 'plateau':
                self.scheduler.step(metrics['mean_auc'])
            
            # Check if best model
            is_best = metrics['mean_auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = metrics['mean_auc']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best or epoch == self.max_epochs:
                self.save_checkpoint(epoch, metrics, is_best)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print comprehensive results
            print(f"Epoch {epoch:3d}/{self.max_epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"AUC: {metrics['mean_auc']:.4f} | "
                  f"F1: {metrics['mean_f1']:.4f} | "
                  f"IoU: {metrics['mean_iou']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Excellence milestone checks
            if metrics['mean_auc'] >= 0.85:
                print(f"🎉 EXCELLENCE ACHIEVED! AUC = {metrics['mean_auc']:.4f}")
                
            # Detailed progress every 10 epochs
            if epoch % 10 == 0:
                print(f"\n📊 Progress Report (Epoch {epoch}):")
                print(f"   Best AUC: {self.best_val_auc:.4f}")
                print(f"   Current AUC: {metrics['mean_auc']:.4f}")
                print(f"   Mean F1: {metrics['mean_f1']:.4f}")
                print(f"   Mean Precision: {metrics['mean_precision']:.4f}")
                print(f"   Mean Recall: {metrics['mean_recall']:.4f}")
                print(f"   Mean IoU: {metrics['mean_iou']:.4f}")
                print(f"   Overall Accuracy: {metrics['overall_accuracy']:.4f}")
                
                # Class performance highlights
                best_classes = np.argsort(metrics['class_aucs'])[-3:]
                print(f"   Top 3 classes:")
                for idx in best_classes:
                    print(f"     {self.disease_classes[idx]}: {metrics['class_aucs'][idx]:.3f}")
                print()
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⏰ Early stopping after {self.patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n🏆 Excellence Training Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best AUC: {self.best_val_auc:.4f} (Epoch {self.best_epoch})")
        print(f"Target achieved: {'✅ YES' if self.best_val_auc >= 0.85 else '🔸 Close' if self.best_val_auc >= 0.80 else '❌ No'}")
        
        return self.best_val_auc
