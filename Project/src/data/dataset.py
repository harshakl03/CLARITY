import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional

class ChestXrayDataset(Dataset):
    """Custom dataset for NIH Chest X-ray multi-label classification"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 image_mapping: Dict[str, Path], 
                 image_size: int = 512,
                 is_training: bool = True,
                 augmentation_prob: float = 0.8):
        """
        Args:
            df: DataFrame with image metadata
            image_mapping: Dict mapping image names to file paths
            image_size: Target image size (square)
            is_training: Whether this is training set (enables augmentations)
            augmentation_prob: Probability of applying augmentations
        """
        self.df = df.reset_index(drop=True)
        self.image_mapping = image_mapping
        self.image_size = image_size
        self.is_training = is_training
        self.augmentation_prob = augmentation_prob
        
        # Define the 14 NIH disease classes + No Finding
        self.disease_classes = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        
        print(f"Dataset created with {len(self.df)} samples")
        print(f"Training mode: {is_training}")
        print(f"Image size: {image_size}x{image_size}")
        
        # Create label matrix
        self._create_label_matrix()
        
        # Define transforms
        self._create_transforms()
    
    def _create_label_matrix(self):
        """Create binary label matrix for multi-label classification"""
        
        self.labels = np.zeros((len(self.df), len(self.disease_classes)), 
                              dtype=np.float32)
        
        for idx, findings in enumerate(self.df['Finding Labels']):
            findings_str = str(findings)
            
            if findings_str == 'No Finding':
                self.labels[idx, 0] = 1.0  # No Finding is index 0
            else:
                # Handle multiple findings
                finding_list = findings_str.split('|')
                for finding in finding_list:
                    finding = finding.strip()  # Remove whitespace
                    if finding in self.disease_classes:
                        class_idx = self.disease_classes.index(finding)
                        self.labels[idx, class_idx] = 1.0
        
        # Print label statistics
        pos_counts = self.labels.sum(axis=0)
        print(f"\nLabel matrix created: {self.labels.shape}")
        print(f"Positive samples per class:")
        for i, disease in enumerate(self.disease_classes):
            print(f"  {disease:.<25} {pos_counts[i]:>6.0f}")
    
    def _create_transforms(self):
        """Create image preprocessing transforms optimized for chest X-rays"""
        
        if self.is_training:
            # Training augmentations - medical imaging specific
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                
                # Geometric augmentations (conservative for medical images)
                A.OneOf([
                    A.HorizontalFlip(p=0.5),  # Chest X-rays can be horizontally flipped
                    A.Rotate(limit=5, p=0.3), # Very small rotation only
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                                     rotate_limit=3, p=0.3),
                ], p=self.augmentation_prob),
                
                # Intensity augmentations (important for X-ray contrast)
                A.OneOf([
                    A.CLAHE(clip_limit=2.0, p=0.5),  # Contrast Limited AHE
                    A.RandomBrightnessContrast(brightness_limit=0.1, 
                                             contrast_limit=0.1, p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ], p=self.augmentation_prob),
                
                # Noise and blur (subtle)
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
                ], p=0.2),
                
                # Normalization using ImageNet statistics
                A.Normalize(mean=[0.485, 0.456, 0.406],  
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Validation/test transforms (no augmentation)
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        print(f"Transforms created for {'training' if self.is_training else 'validation'}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample with error handling"""
        
        # Get image info
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        
        # Load image with error handling
        try:
            if image_name not in self.image_mapping:
                # Create dummy image if missing
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                print(f"⚠️  Missing image: {image_name}")
            else:
                image_path = self.image_mapping[image_name]
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    # Handle corrupted images
                    image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                    print(f"⚠️  Corrupted image: {image_name}")
                
                # Convert grayscale to RGB for pretrained models
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        except Exception as e:
            print(f"❌ Error loading {image_name}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Apply transforms
        try:
            transformed = self.transform(image=image)
            image = transformed['image']
        except Exception as e:
            print(f"❌ Transform error for {image_name}: {e}")
            # Return zero tensor if transform fails
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Get labels
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Additional metadata for debugging/analysis
        metadata = {
            'image_name': image_name,
            'patient_id': row.get('Patient ID', 'Unknown'),
            'age': row.get('Patient Age', -1),
            'gender': row.get('Patient Gender', 'Unknown'),
            'view_position': row.get('View Position', 'Unknown'),
            'findings': row['Finding Labels']
        }
        
        return image, labels, metadata

def create_data_splits(df: pd.DataFrame, 
                      test_size: float = 0.2, 
                      val_size: float = 0.1, 
                      random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits with patient-level separation"""
    
    np.random.seed(random_seed)
    
    # Get unique patients to avoid data leakage
    unique_patients = df['Patient ID'].unique()
    np.random.shuffle(unique_patients)
    
    # Calculate split sizes
    n_patients = len(unique_patients)
    n_test = int(n_patients * test_size)
    n_val = int(n_patients * val_size)
    n_train = n_patients - n_test - n_val
    
    # Split patients
    test_patients = set(unique_patients[:n_test])
    val_patients = set(unique_patients[n_test:n_test+n_val])
    train_patients = set(unique_patients[n_test+n_val:])
    
    # Create dataframe splits
    train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    test_df = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)
    
    print(f"Patient-level data splits:")
    print(f"  Train: {len(train_df):,} images from {len(train_patients):,} patients ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} images from {len(val_patients):,} patients ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} images from {len(test_patients):,} patients ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no patient overlap
    assert len(train_patients & val_patients) == 0, "Patient overlap between train and val!"
    assert len(train_patients & test_patients) == 0, "Patient overlap between train and test!"
    assert len(val_patients & test_patients) == 0, "Patient overlap between val and test!"
    
    print("✅ No patient overlap verified - clean splits!")
    
    return train_df, val_df, test_df

def calculate_class_weights(labels: np.ndarray, method: str = 'inverse_freq') -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    
    pos_counts = labels.sum(axis=0)
    total_samples = len(labels)
    
    if method == 'inverse_freq':
        # Inverse frequency weighting
        weights = total_samples / (2 * (pos_counts + 1))  # +1 to avoid division by zero
    elif method == 'effective_num':
        # Effective number of samples weighting
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, pos_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        weights = np.ones(len(pos_counts))
    
    # Normalize weights to have mean = 1.0
    weights = weights / weights.mean()
    
    disease_classes = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 
        'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
        'Pleural_Thickening', 'Hernia'
    ]
    
    print(f"\nClass weights ({method}):")
    print("-" * 50)
    for i, (disease, weight) in enumerate(zip(disease_classes, weights)):
        print(f"{disease:.<25} {weight:>8.3f} (pos: {pos_counts[i]:>6.0f})")
    
    return torch.tensor(weights, dtype=torch.float32)