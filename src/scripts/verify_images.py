"""
Verify extracted images match dataset metadata.
"""

import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def verify_images(image_dir: str, max_check: int = 100):
    """Check if extracted images match dataset paths."""
    
    print("Loading dataset metadata...")
    dataset = load_dataset('FreedomIntelligence/ShareGPT-4o-Image', '1_text_to_image', split='train')
    
    image_dir = Path(image_dir)
    
    print(f"Checking first {max_check} samples...")
    missing = []
    found = 0
    
    for i in tqdm(range(min(max_check, len(dataset)))):
        sample = dataset[i]
        image_path = sample['output_image']  # e.g., "image/8483.png"
        
        # Try different possible paths
        possible_paths = [
            image_dir / image_path,  # Direct: ./images/image/8483.png
            image_dir / Path(image_path).name,  # Flat: ./images/8483.png
            image_dir / "image" / Path(image_path).name,  # With subdir
        ]
        
        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break
        
        if found_path:
            found += 1
        else:
            missing.append((i, image_path))
            if len(missing) <= 5:  # Show first 5 missing
                print(f"Missing: {image_path}")
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Found: {found}/{max_check}")
    print(f"Missing: {len(missing)}/{max_check}")
    
    if found > 0:
        # Determine the correct path structure
        sample = dataset[0]
        image_path = sample['output_image']
        for path in possible_paths:
            if path.exists():
                print(f"\n✓ Correct image directory structure:")
                print(f"  {path.parent}")
                print(f"\nUse this in preprocessing:")
                print(f"  --image_dir {path.parent}")
                break
    
    return found, missing


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--max_check', type=int, default=100)
    args = parser.parse_args()
    
    verify_images(args.image_dir, args.max_check)