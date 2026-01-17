"""
Final preprocessing for ShareGPT-4o-Image dataset.
Handles the specific structure: input_prompt, output_image, output_image_resolution
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import yaml


def calculate_aspect_ratio(width: int, height: int) -> float:
    """Calculate aspect ratio (height/width)"""
    return height / width


def find_best_bucket(
    img_width: int,
    img_height: int,
    buckets: List[Dict],
    fallback: str = "closest"
) -> Optional[Dict]:
    """Find the best bucket for an image using intelligent matching."""
    img_ar = calculate_aspect_ratio(img_width, img_height)
    img_pixels = img_width * img_height
    
    # Find all matching buckets within tolerance
    matches = []
    for bucket in buckets:
        bucket_ar = bucket['aspect_ratio']
        tolerance = bucket['tolerance']
        
        ar_min = bucket_ar * (1 - tolerance)
        ar_max = bucket_ar * (1 + tolerance)
        
        if ar_min <= img_ar <= ar_max:
            bucket_pixels = bucket['height'] * bucket['width']
            scale_factor = (bucket_pixels / img_pixels) ** 0.5
            
            matches.append({
                'bucket': bucket,
                'ar_diff': abs(img_ar - bucket_ar),
                'scale_factor': scale_factor,
                'pixel_diff': abs(bucket_pixels - img_pixels),
            })
    
    if matches:
        matches.sort(key=lambda x: (abs(x['scale_factor'] - 1.0), x['ar_diff']))
        return matches[0]['bucket']
    
    if fallback == "closest":
        closest = min(buckets, key=lambda b: abs(b['aspect_ratio'] - img_ar))
        return closest
    elif fallback == "crop_to_square":
        return next((b for b in buckets if b['name'] == '1:1'), buckets[0])
    else:
        return None


def calculate_smart_crop(
    img_width: int,
    img_height: int,
    target_width: int,
    target_height: int,
    strategy: str = "center_weighted"
) -> Tuple[int, int, int, int, int, int]:
    """
    Calculate smart crop that preserves important content.
    Returns: (resize_w, resize_h, crop_l, crop_t, crop_r, crop_b)
    """
    img_ar = img_width / img_height
    target_ar = target_width / target_height
    
    if img_ar > target_ar:
        # Image is wider - fit height, crop width
        new_height = target_height
        new_width = int(target_height * img_ar)
        
        if strategy == "center_weighted":
            crop_left = int((new_width - target_width) * 0.5)
        else:
            crop_left = (new_width - target_width) // 2
        
        crop_top = 0
        crop_right = crop_left + target_width
        crop_bottom = target_height
        
    else:
        # Image is taller - fit width, crop height
        new_width = target_width
        new_height = int(target_width / img_ar)
        
        if strategy == "center_weighted":
            # Slightly prefer upper-center for portraits
            crop_top = int((new_height - target_height) * 0.45)
        else:
            crop_top = (new_height - target_height) // 2
        
        crop_left = 0
        crop_right = target_width
        crop_bottom = crop_top + target_height
    
    return (new_width, new_height, crop_left, crop_top, crop_right, crop_bottom)


def preprocess_dataset(
    dataset_name: str,
    output_dir: str,
    bucket_config_path: str,
    split: str = "train",
    max_samples: int = None,
):
    """Preprocess dataset with intelligent bucketing."""
    
    # Load bucket config
    with open(bucket_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    buckets = config['buckets']
    fallback_strategy = config.get('fallback_strategy', 'closest')
    min_bucket_size = config.get('min_bucket_size', 50)
    max_scale_factor = config.get('max_scale_factor', 1.5)
    crop_strategy = config.get('crop_strategy', 'center_weighted')
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name,'1_text_to_image', split=split, streaming=False)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples: {len(dataset)}")
    print(f"Buckets: {len(buckets)}")
    print(f"Fallback strategy: {fallback_strategy}")
    
    # Initialize storage
    bucket_data = defaultdict(list)
    bucket_stats = defaultdict(lambda: {
        'count': 0,
        'total_pixels': 0,
        'scale_samples': []
    })
    
    skipped_extreme_scale = 0
    skipped_no_bucket = 0
    skipped_missing_data = 0
    skipped_errors = 0
    
    # Process each sample
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        try:
            sample = dataset[idx]
            
            # Extract data from ShareGPT-4o-Image format
            # Format: {'input_prompt': str, 'output_image': str, 'output_image_resolution': [w, h]}
            
            if 'input_prompt' not in sample or 'output_image_resolution' not in sample:
                skipped_missing_data += 1
                continue
            
            text = sample['input_prompt']
            image_path = sample['output_image']  # e.g., "image/8483.png"
            resolution = sample['output_image_resolution']  # [width, height]
            
            if len(resolution) != 2:
                skipped_missing_data += 1
                continue
            
            img_width, img_height = resolution[0], resolution[1]
            
            # Validate dimensions
            if img_width <= 0 or img_height <= 0:
                skipped_missing_data += 1
                continue
            
            # Find best bucket
            best_bucket = find_best_bucket(
                img_width, img_height, buckets, fallback_strategy
            )
            
            if best_bucket is None:
                skipped_no_bucket += 1
                continue
            
            target_width = best_bucket['width']
            target_height = best_bucket['height']
            
            # Check scale factor
            img_pixels = img_width * img_height
            target_pixels = target_width * target_height
            scale_factor = (target_pixels / img_pixels) ** 0.5
            
            if scale_factor > max_scale_factor:
                skipped_extreme_scale += 1
                continue
            
            # Calculate smart crop parameters
            resize_w, resize_h, crop_l, crop_t, crop_r, crop_b = calculate_smart_crop(
                img_width, img_height,
                target_width, target_height,
                strategy=crop_strategy
            )
            
            # Store metadata
            bucket_key = best_bucket['name']
            bucket_data[bucket_key].append({
                'index': idx,
                'text': text,
                'image_path': image_path,  # Store the path for loading later
                'original_width': img_width,
                'original_height': img_height,
                'bucket_name': bucket_key,
                'bucket_height': target_height,
                'bucket_width': target_width,
                'resize_width': resize_w,
                'resize_height': resize_h,
                'crop_left': crop_l,
                'crop_top': crop_t,
                'crop_right': crop_r,
                'crop_bottom': crop_b,
                'scale_factor': scale_factor,
            })
            
            # Update stats
            bucket_stats[bucket_key]['count'] += 1
            bucket_stats[bucket_key]['total_pixels'] += target_pixels
            bucket_stats[bucket_key]['scale_samples'].append(scale_factor)
            
        except Exception as e:
            skipped_errors += 1
            if idx < 5:  # Debug first few errors
                print(f"Error processing sample {idx}: {e}")
            continue
    
    # Calculate average scale factors
    for bucket_key in bucket_stats:
        scales = bucket_stats[bucket_key]['scale_samples']
        if scales:
            bucket_stats[bucket_key]['avg_scale'] = np.mean(scales)
            bucket_stats[bucket_key]['std_scale'] = np.std(scales)
            del bucket_stats[bucket_key]['scale_samples']
    
    # Filter small buckets
    filtered_buckets = {
        k: v for k, v in bucket_data.items()
        if len(v) >= min_bucket_size
    }
    
    # Print statistics
    print("\n" + "="*80)
    print("BUCKET STATISTICS")
    print("="*80)
    print(f"{'Bucket':<10} {'Size':<8} {'Resolution':<12} {'Pixels':<10} {'Avg Scale':<12}")
    print("-"*80)
    
    total_kept = 0
    for bucket_key in sorted(filtered_buckets.keys()):
        samples = filtered_buckets[bucket_key]
        count = len(samples)
        total_kept += count
        
        h = samples[0]['bucket_height']
        w = samples[0]['bucket_width']
        pixels = h * w
        
        avg_scale = bucket_stats[bucket_key]['avg_scale']
        std_scale = bucket_stats[bucket_key].get('std_scale', 0)
        
        print(f"{bucket_key:<10} {count:<8} {w}x{h:<7} {pixels:<10} "
              f"{avg_scale:.3f}±{std_scale:.3f}")
    
    print("-"*80)
    print(f"Total kept: {total_kept}")
    print(f"Skipped (extreme scale): {skipped_extreme_scale}")
    print(f"Skipped (no bucket): {skipped_no_bucket}")
    print(f"Skipped (missing data): {skipped_missing_data}")
    print(f"Skipped (errors): {skipped_errors}")
    total_skipped = skipped_extreme_scale + skipped_no_bucket + skipped_missing_data + skipped_errors
    print(f"Total skipped: {total_skipped}")
    if len(dataset) > 0:
        print(f"Keep rate: {100 * total_kept / len(dataset):.2f}%")
    print("="*80)
    
    # Save metadata
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = os.path.join(output_dir, 'bucket_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(filtered_buckets, f, indent=2)
    
    stats_path = os.path.join(output_dir, 'bucket_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'total_samples': total_kept,
            'num_buckets': len(filtered_buckets),
            'buckets': {k: {
                'count': len(v),
                'resolution': f"{v[0]['bucket_width']}x{v[0]['bucket_height']}",
                'avg_scale': bucket_stats[k]['avg_scale'],
            } for k, v in filtered_buckets.items()},
            'skipped': {
                'extreme_scale': skipped_extreme_scale,
                'no_bucket': skipped_no_bucket,
                'missing_data': skipped_missing_data,
                'errors': skipped_errors,
            },
            'keep_rate': 100 * total_kept / len(dataset) if len(dataset) > 0 else 0,
        }, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Stats saved to: {stats_path}")
    
    return filtered_buckets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                       default='FreedomIntelligence/ShareGPT-4o-Image')
    parser.add_argument('--output_dir', type=str,
                       default='./data/sharegpt4o_smart_bucketed')
    parser.add_argument('--bucket_config', type=str,
                       default='configs/bucket_config.yaml')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    preprocess_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        bucket_config_path=args.bucket_config,
        split=args.split,
        max_samples=args.max_samples,
    )