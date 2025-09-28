import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def create_rtx3060_dataloaders(train_dataset, val_dataset, test_dataset=None,
                              batch_size=8, num_workers=2, use_weighted_sampling=True):
    """Create optimized dataloaders for RTX 3060"""
    
    print(f"Creating RTX 3060 optimized dataloaders:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Weighted sampling: {use_weighted_sampling}")
    
    # Create weighted sampler for training if requested
    train_sampler = None
    if use_weighted_sampling:
        # Calculate sample weights based on rarest positive class in each sample
        labels = train_dataset.labels
        pos_counts = labels.sum(axis=0) + 1  # +1 to avoid division by zero
        
        sample_weights = []
        for sample_labels in labels:
            pos_indices = np.where(sample_labels > 0)
            if len(pos_indices) > 0:
                # Weight by inverse of rarest positive class frequency
                rarest_count = pos_counts[pos_indices].min()
                weight = 1.0 / rarest_count
            else:
                weight = 1.0
            sample_weights.append(weight)
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"  Weighted sampler created with {len(sample_weights)} samples")
    
    # Training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Consistent batch sizes for training
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Test dataloader (optional)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches") 
    if test_loader:
        print(f"  Test:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader