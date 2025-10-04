# CLARITy Enhanced Training Pipeline
# src/models/trainer_enhanced.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedRTX3060Trainer:
    """Enhanced Training pipeline optimized for RTX 3060 with advanced features"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 device: str = 'cuda',
                 learning_rate: float = 2e-4,  # Slightly higher LR
                 weight_decay: float = 1e-4,
                 accumulation_steps: int = 2,
                 mixed_precision: bool = True,
                 checkpoint_dir: str = 'models/checkpoints',
                 patience: int = 7,
                 scheduler_type: str = 'cosine',
                 warmup_epochs: int = 3,
                 max_epochs: int = 50,
                 label_smoothing: float = 0.1):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.label_smoothing = label_smoothing
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced Optimizer with better settings
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Advanced Learning Rate Scheduling
        self.scheduler_type = scheduler_type
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate * 3,
                total_steps=max_epochs * len(train_loader) // accumulation_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:  # plateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=learning_rate * 0.001
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
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Progressive training support
        self.progressive_stages = []
        self.current_stage = "standard"
        
        # Disease classes
        self.disease_classes = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        
        print(f"🚀 Enhanced RTX 3060 Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Effective batch size: {train_loader.batch_size * accumulation_steps}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Scheduler: {scheduler_type}")
        print(f"   Warmup epochs: {warmup_epochs}")
        print(f"   Mixed precision: {mixed_precision}")
        print(f"   Label smoothing: {label_smoothing}")
    
    def apply_label_smoothing(self, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        """Apply label smoothing to reduce overconfidence"""
        if smoothing <= 0:
            return targets
        
        # For multi-label: smooth positive labels towards (1-smoothing) and negative towards smoothing
        smoothed = targets * (1 - smoothing) + smoothing * 0.5
        return smoothed
    
    def train_epoch(self, epoch: int) -> float:
        """Enhanced training epoch with progressive features"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Warmup learning rate
        if epoch <= self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Progress bar with more info
        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {epoch:2d}/{self.max_epochs} [Stage: {self.current_stage}]", 
                   leave=False)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                labels = self.apply_label_smoothing(labels, self.label_smoothing)
            
            # Mixed precision forward pass
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps
                
                # Backward pass with gradient scaling and clipping
                self.scaler.scale(loss).backward()
                
                # Accumulate gradients
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping before step
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update scheduler for OneCycleLR
                    if self.scheduler_type == 'onecycle':
                        self.scheduler.step()
            else:
                # Standard training
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler_type == 'onecycle':
                        self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar with current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            current_loss = loss.item() * self.accumulation_steps
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'LR': f'{current_lr:.2e}',
                'Stage': self.current_stage
            })
            
            # Memory management
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if len(self.train_loader) % self.accumulation_steps != 0:
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float, float, List[float], List[float]]:
        """Enhanced validation with comprehensive metrics"""
        
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for images, labels, metadata in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Store predictions and labels
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Convert to probabilities
        all_probs = torch.sigmoid(all_logits).numpy()
        all_labels = all_labels.numpy()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        
        # Per-class metrics with better error handling
        class_aucs = []
        class_aps = []
        
        for i in range(len(self.disease_classes)):
            if all_labels[:, i].sum() > 0 and (1 - all_labels[:, i]).sum() > 0:
                try:
                    auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    ap = average_precision_score(all_labels[:, i], all_probs[:, i])
                    class_aucs.append(auc)
                    class_aps.append(ap)
                except Exception as e:
                    class_aucs.append(0.5)
                    class_aps.append(0.0)
            else:
                class_aucs.append(0.5)
                class_aps.append(0.0)
        
        # Mean metrics (excluding NaN)
        valid_aucs = [auc for auc in class_aucs if not np.isnan(auc)]
        valid_aps = [ap for ap in class_aps if not np.isnan(ap)]
        
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.5
        mean_ap = np.mean(valid_aps) if valid_aps else 0.0
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_aucs.append(mean_auc)
        self.val_aps.append(mean_ap)
        
        return avg_loss, mean_auc, mean_ap, class_aucs, class_aps
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, 
                       additional_info: Optional[Dict] = None) -> None:
        """Enhanced checkpoint saving with more metadata"""
        
        checkpoint = {
            # Model info
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            
            # Training config
            'training_config': {
                'learning_rate': self.learning_rates[0] if self.learning_rates else 0,
                'scheduler_type': self.scheduler_type,
                'warmup_epochs': self.warmup_epochs,
                'label_smoothing': self.label_smoothing,
                'accumulation_steps': self.accumulation_steps,
                'mixed_precision': self.mixed_precision,
            },
            
            # Performance metrics
            'best_val_auc': self.best_val_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'val_aps': self.val_aps,
            'learning_rates': self.learning_rates,
            
            # Model architecture info
            'model_info': {
                'architecture': 'DenseNet121Enhanced',
                'num_classes': self.model.num_classes,
                'use_attention': getattr(self.model, 'use_attention', False),
                'feature_fusion': getattr(self.model, 'feature_fusion', False),
            },
            
            # Training stage
            'current_stage': self.current_stage,
            'progressive_stages': self.progressive_stages,
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint['additional_info'] = additional_info
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model_enhanced.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved: AUC = {self.best_val_auc:.4f}")
        
        # Save latest model (for resuming training)
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True) -> Dict:
        """Load model from checkpoint with full state restoration"""
        
        print(f"📥 Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Load optimizer and scheduler states
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.mixed_precision and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore training metrics
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_aucs = checkpoint.get('val_aucs', [])
            self.val_aps = checkpoint.get('val_aps', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
            self.current_stage = checkpoint.get('current_stage', 'standard')
            self.progressive_stages = checkpoint.get('progressive_stages', [])
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Best AUC: {checkpoint.get('best_val_auc', 'N/A')}")
        print(f"   Stage: {checkpoint.get('current_stage', 'N/A')}")
        
        return checkpoint
    
    def train(self, num_epochs: int, save_every: int = 5, 
              progressive_training: bool = False) -> float:
        """Enhanced training loop with progressive training support"""
        
        print(f"🚀 Starting enhanced training for {num_epochs} epochs")
        print(f"   Progressive training: {progressive_training}")
        if progressive_training:
            print(f"   Stage 1: Epochs 1-{num_epochs//2} (Frozen features)")
            print(f"   Stage 2: Epochs {num_epochs//2+1}-{num_epochs} (Fine-tuning)")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Progressive training stage switching
            if progressive_training and epoch == num_epochs // 2 + 1:
                print(f"\n🔄 Switching to fine-tuning stage...")
                self.model.unfreeze_features()
                self.current_stage = "fine_tuning"
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"   Learning rate reduced to: {self.optimizer.param_groups[0]['lr']:.2e}")
                self.progressive_stages.append(f"Stage_2_started_epoch_{epoch}")
            
            # Training step
            train_loss = self.train_epoch(epoch)
            
            # Validation step
            val_loss, val_auc, val_ap, class_aucs, class_aps = self.validate()
            
            # Learning rate scheduling (for non-OneCycleLR schedulers)
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            elif self.scheduler_type == 'plateau':
                self.scheduler.step(val_auc)
            
            # Check if best model
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best or epoch == num_epochs:
                additional_info = {
                    'epoch_time': time.time() - epoch_start,
                    'per_class_aucs': dict(zip(self.disease_classes, class_aucs)),
                    'per_class_aps': dict(zip(self.disease_classes, class_aps))
                }
                self.save_checkpoint(epoch, is_best, additional_info)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch results
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val AP: {val_ap:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Print detailed metrics every 5 epochs
            if epoch % 5 == 0:
                print(f"\nPer-class AUC scores (Epoch {epoch}):")
                for disease, auc in zip(self.disease_classes, class_aucs):
                    status = "✅" if auc >= 0.8 else "🔶" if auc >= 0.7 else "🔸"
                    print(f"  {status} {disease:.<25} {auc:.3f}")
                print()
            
            # Early stopping with patience
            if self.epochs_without_improvement >= self.patience:
                print(f"\n🛑 Early stopping after {self.patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Enhanced training completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation AUC: {self.best_val_auc:.4f} (Epoch {self.best_epoch})")
        print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"Training stages: {len(self.progressive_stages) + 1}")
        
        return self.best_val_auc
    
    def plot_enhanced_training_curves(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive training visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('CLARITy Enhanced Training Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 1. Loss curves with LR overlay
        ax1 = axes[0, 0]
        ax1_lr = ax1.twinx()
        
        ax1.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        ax1_lr.plot(epochs, self.learning_rates, 'g--', linewidth=1, label='Learning Rate', alpha=0.6)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='black')
        ax1_lr.set_ylabel('Learning Rate', color='green')
        ax1.set_title('Training Progress with Learning Rate')
        ax1.legend(loc='upper left')
        ax1_lr.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. AUC progression with targets
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.val_aucs, 'g-', linewidth=3, marker='o', markersize=4, 
                 label='Validation AUC', alpha=0.8)
        ax2.axhline(y=self.best_val_auc, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best AUC: {self.best_val_auc:.3f}')
        ax2.axhline(y=0.8, color='orange', linestyle=':', alpha=0.7, label='Clinical Target (0.80)')
        ax2.axhline(y=0.75, color='yellow', linestyle=':', alpha=0.7, label='Good Performance (0.75)')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('Model Performance Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.5, 1.0])
        
        # 3. Loss vs AUC relationship
        ax3 = axes[0, 2]
        scatter = ax3.scatter(self.val_losses, self.val_aucs, c=epochs, cmap='viridis', 
                             alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Validation Loss')
        ax3.set_ylabel('Validation AUC')
        ax3.set_title('Loss-Performance Relationship')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Epoch')
        
        # 4. Learning rate schedule
        ax4 = axes[1, 0]
        ax4.semilogy(epochs, self.learning_rates, 'purple', linewidth=2, marker='o', markersize=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate (log scale)')
        ax4.set_title(f'Learning Rate Schedule ({self.scheduler_type})')
        ax4.grid(True, alpha=0.3)
        
        # 5. Training stability (loss smoothing)
        ax5 = axes[1, 1]
        if len(self.train_losses) > 5:
            # Moving average for smoothing
            window = min(5, len(self.train_losses) // 4)
            train_smooth = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            smooth_epochs = range(window, len(self.train_losses) + 1)
            
            ax5.plot(smooth_epochs, train_smooth, 'b-', linewidth=2, label=f'Train (MA-{window})', alpha=0.8)
            ax5.plot(smooth_epochs, val_smooth, 'r-', linewidth=2, label=f'Val (MA-{window})', alpha=0.8)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Smoothed Loss')
            ax5.set_title('Training Stability (Moving Average)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Performance summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate improvement metrics
        initial_auc = self.val_aucs[0] if self.val_aucs else 0
        improvement = self.best_val_auc - initial_auc
        final_lr = self.learning_rates[-1] if self.learning_rates else 0
        
        summary_text = f"""
Enhanced Training Summary

🎯 Performance:
• Best AUC: {self.best_val_auc:.4f}
• Initial AUC: {initial_auc:.4f}
• Improvement: +{improvement:.4f}
• Best Epoch: {self.best_epoch}

📈 Training Details:
• Total Epochs: {len(self.train_losses)}
• Final Loss: {self.train_losses[-1]:.4f}
• Scheduler: {self.scheduler_type}
• Final LR: {final_lr:.2e}

🔧 Configuration:
• Stage: {self.current_stage}
• Mixed Precision: {self.mixed_precision}
• Label Smoothing: {self.label_smoothing}
• Warmup: {self.warmup_epochs} epochs

🏥 Clinical Readiness:
• Target (0.80): {"✅ Achieved" if self.best_val_auc >= 0.8 else "🔸 Not yet"}
• Good (0.75): {"✅ Achieved" if self.best_val_auc >= 0.75 else "🔸 Not yet"}
• Improvement: {"📈 Significant" if improvement > 0.05 else "📊 Moderate"}
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                 verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Enhanced training curves saved to: {save_path}")
        
        plt.show()

# Utility Functions for Enhanced Training

def create_enhanced_trainer(model, train_loader, val_loader, criterion, 
                          config: Optional[Dict] = None) -> EnhancedRTX3060Trainer:
    """Factory function to create enhanced trainer with optimal defaults"""
    
    default_config = {
        'learning_rate': 2e-4,
        'scheduler_type': 'cosine',
        'warmup_epochs': 3,
        'max_epochs': 50,
        'patience': 8,
        'label_smoothing': 0.1,
        'accumulation_steps': 2,
        'mixed_precision': True
    }
    
    if config:
        default_config.update(config)
    
    trainer = EnhancedRTX3060Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        **default_config
    )
    
    return trainer

def setup_progressive_training(model, train_loader, val_loader, class_weights,
                             total_epochs: int = 50) -> Tuple[EnhancedRTX3060Trainer, Dict]:
    """Setup progressive training with two-stage approach"""
    
    from .densenet121_enhanced import create_progressive_model
    
    # Stage 1: Initial training with frozen features
    model_stage1, criterion_stage1 = create_progressive_model(
        num_classes=15,
        class_weights=class_weights,
        stage="initial"
    )
    
    config_stage1 = {
        'learning_rate': 3e-4,  # Higher LR for classifier-only training
        'scheduler_type': 'onecycle',
        'warmup_epochs': 2,
        'max_epochs': total_epochs // 2,
        'patience': 6,
        'label_smoothing': 0.05,  # Less smoothing for initial training
    }
    
    trainer_stage1 = EnhancedRTX3060Trainer(
        model=model_stage1,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_stage1,
        **config_stage1
    )
    
    return trainer_stage1, config_stage1