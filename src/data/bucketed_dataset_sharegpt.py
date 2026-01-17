"""
Bucketed dataset loader for ShareGPT-4o-Image format.
"""

import json
import random
from typing import List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T


class BucketedShareGPTDataset(Dataset):
    """
    Dataset that loads ShareGPT-4o-Image from bucketed metadata.
    Each sample is resized/cropped to match its assigned bucket.
    """
    
    def __init__(
        self,
        metadata_path: str,
        dataset_name: str = 'FreedomIntelligence/ShareGPT-4o-Image',
        split: str = 'train',
        image_dir: str = None,  # Optional: local directory with images
        transform: Optional[callable] = None,
    ):
        """
        Args:
            metadata_path: Path to bucket_metadata.json
            dataset_name: HuggingFace dataset identifier
            split: Dataset split
            image_dir: Local directory containing images (if None, loads from HF)
            transform: Additional transforms to apply
        """
        with open(metadata_path, 'r') as f:
            self.bucket_metadata = json.load(f)
        
        # Flatten all samples across buckets
        self.samples = []
        for bucket_key, bucket_samples in self.bucket_metadata.items():
            for sample in bucket_samples:
                sample['bucket_key'] = bucket_key
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {len(self.bucket_metadata)} buckets")
        
        # Load base dataset
        self.dataset = load_dataset(dataset_name,'1_text_to_image', split=split, streaming=False)
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        
        # Get image path
        dataset_idx = sample_meta['index']
        image_path = sample_meta['image_path']  # e.g., "image/8483.png"
        
        # Load image
        if self.image_dir:
            # Load from local directory
            img_file = self.image_dir / image_path
            image = Image.open(img_file).convert('RGB')
        else:
            # Note: ShareGPT-4o-Image doesn't include actual images in HF dataset
            # You need to download images separately or provide local path
            # For now, create a placeholder
            # In production, you must download images to image_dir
            raise ValueError(
                f"Images not available in HF dataset. "
                f"Please download images and provide image_dir parameter. "
                f"Expected image at: {image_path}"
            )
        
        # Resize and crop to bucket size
        resize_w = sample_meta['resize_width']
        resize_h = sample_meta['resize_height']
        crop_l = sample_meta['crop_left']
        crop_t = sample_meta['crop_top']
        crop_r = sample_meta['crop_right']
        crop_b = sample_meta['crop_bottom']
        
        # Resize
        image = image.resize((resize_w, resize_h), Image.LANCZOS)
        
        # Crop
        image = image.crop((crop_l, crop_t, crop_r, crop_b))
        
        # Apply additional transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = T.ToTensor()(image)
            image = T.Normalize([0.5], [0.5])(image)  # [-1, 1]
        
        return {
            'image': image,
            'text': sample_meta['text'],
            'bucket_key': sample_meta['bucket_key'],
            'height': sample_meta['bucket_height'],
            'width': sample_meta['bucket_width'],
            'z': torch.tensor([]),  # Placeholder for compatibility
        }


class BucketBatchSampler(Sampler):
    """
    Sampler that ensures each batch contains samples from the same bucket.
    """
    
    def __init__(
        self,
        dataset: BucketedShareGPTDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Group indices by bucket
        self.bucket_indices = {}
        for idx, sample in enumerate(dataset.samples):
            bucket_key = sample['bucket_key']
            if bucket_key not in self.bucket_indices:
                self.bucket_indices[bucket_key] = []
            self.bucket_indices[bucket_key].append(idx)
        
        print(f"BucketBatchSampler initialized:")
        for bucket_key, indices in self.bucket_indices.items():
            num_batches = len(indices) // batch_size
            print(f"  {bucket_key}: {len(indices)} samples, {num_batches} batches")
    
    def __iter__(self):
        # Shuffle within each bucket if needed
        bucket_batches = []
        
        for bucket_key, indices in self.bucket_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    bucket_batches.append(batch)
        
        # Shuffle bucket order
        if self.shuffle:
            random.shuffle(bucket_batches)
        
        # Yield batches
        for batch in bucket_batches:
            yield batch
    
    def __len__(self):
        total_batches = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches