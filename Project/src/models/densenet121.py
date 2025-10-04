import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import matplotlib.pyplot as plt

class RTX3060Trainer:
    """Training pipeline optimized for RTX 3060 with 12GB VRAM"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 accumulation_steps: int = 2,  # Effective batch size = 8 * 2 = 16
                 mixed_precision: bool = True,
                 checkpoint_dir: str = 'models/checkpoints',
                 patience: int = 5):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.patience = patience
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer with weight decay for generalization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize validation AUC
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-7
        )
        
        # Mixed precision scaler for RTX 3060
        if self.mixed_precision:
            self.scaler = GradScaler()
            print("✅ Mixed precision training enabled (FP16)")
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_aps = []  # Average precision scores
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Disease classes for metrics
        self.disease_classes = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        
        print(f"🚀 RTX 3060 Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Accumulation steps: {accumulation_steps}")
        print(f"   Effective batch size: {train_loader.batch_size * accumulation_steps}")
        print(f"   Mixed precision: {mixed_precision}")
        print(f"   Learning rate: {learning_rate}")
    
    def train_epoch(self):
        """Train for one epoch with memory optimization"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Clear GPU cache before training
        torch.cuda.empty_cache()
        
        # Initialize progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps  # Scale loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Accumulate gradients
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_loss = loss.item() * self.accumulation_steps
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
            
            # Periodic GPU cache clearing for RTX 3060
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Handle any remaining gradients
        if (len(self.train_loader)) % self.accumulation_steps != 0:
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validation with comprehensive multi-label metrics"""
        
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        
        # Clear GPU cache
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
        
        # AUC per class (handle classes with no positive samples)
        class_aucs = []
        class_aps = []
        
        for i in range(len(self.disease_classes)):
            if all_labels[:, i].sum() > 0:  # Only if positive samples exist
                try:
                    auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    ap = average_precision_score(all_labels[:, i], all_probs[:, i])
                    class_aucs.append(auc)
                    class_aps.append(ap)
                except:
                    class_aucs.append(0.5)  # Random performance
                    class_aps.append(0.0)
            else:
                class_aucs.append(0.5)  # No positive samples
                class_aps.append(0.0)
        
        # Mean metrics
        mean_auc = np.mean(class_aucs)
        mean_ap = np.mean(class_aps)
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_aucs.append(mean_auc)
        self.val_aps.append(mean_ap)
        
        return avg_loss, mean_auc, mean_ap, class_aucs, class_aps
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'val_aps': self.val_aps,
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved: AUC = {self.best_val_auc:.4f}")
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Complete training loop with early stopping"""
        
        print(f"🚀 Starting training for {num_epochs} epochs")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_auc, val_ap, class_aucs, class_aps = self.validate()
            
            # Learning rate scheduling
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
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Print epoch results
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val AP: {val_ap:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Print detailed class metrics every 5 epochs
            if epoch % 5 == 0:
                print("\nPer-class AUC scores:")
                for disease, auc in zip(self.disease_classes, class_aucs):
                    print(f"  {disease:.<25} {auc:.3f}")
                print()
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n🛑 Early stopping after {self.patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Best validation AUC: {self.best_val_auc:.4f} (Epoch {self.best_epoch})")
        
        return self.best_val_auc
    
    def plot_training_curves(self, save_path=None):
        """Plot training and validation curves"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes.plot(epochs, self.train_losses, label='Train Loss', color='blue')
        axes.plot(epochs, self.val_losses, label='Val Loss', color='red')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.set_title('Training and Validation Loss')
        axes.legend()
        axes.grid(True)
        
        # AUC curve
        axes.plot(epochs, self.val_aucs, label='Val AUC', color='green')
        axes.axhline(y=self.best_val_auc, color='red', linestyle='--', 
                       label=f'Best AUC: {self.best_val_auc:.3f}')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('AUC')
        axes.set_title('Validation AUC')
        axes.legend()
        axes.grid(True)
        
        # Average Precision curve
        axes.plot(epochs, self.val_aps, label='Val AP', color='orange')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Average Precision')
        axes.set_title('Validation Average Precision')
        axes.legend()
        axes.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()