"""
Download and extract ShareGPT-4o-Image dataset.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tarfile
from huggingface_hub import hf_hub_download
import concurrent.futures


def download_tar_file(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a single tar file from HuggingFace."""
    print(f"Downloading {filename}...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return local_path


def extract_tar_file(tar_path: str, extract_dir: str):
    """Extract a tar file."""
    print(f"Extracting {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_dir)
    print(f"✓ Extracted {os.path.basename(tar_path)}")


def download_and_extract_dataset(
    output_dir: str,
    num_parts: int = 10,
    parallel_downloads: int = 3,
):
    """
    Download and extract ShareGPT-4o-Image dataset.
    
    Args:
        output_dir: Where to extract images
        num_parts: Number of tar parts (0-9 = 10 parts)
        parallel_downloads: Number of parallel downloads
    """
    
    repo_id = "FreedomIntelligence/ShareGPT-4o-Image"
    cache_dir = os.path.join(output_dir, "hf_cache")
    extract_dir = os.path.join(output_dir, "images")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Generate filenames
    tar_files = [f"text_to_image_part_{i}.tar" for i in range(num_parts)]
    
    print("="*80)
    print(f"Downloading {num_parts} tar files from {repo_id}")
    print(f"Cache directory: {cache_dir}")
    print(f"Extract directory: {extract_dir}")
    print("="*80)
    
    # Download tar files (with limited parallelism to avoid rate limits)
    downloaded_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_downloads) as executor:
        future_to_file = {
            executor.submit(download_tar_file, repo_id, filename, cache_dir): filename
            for filename in tar_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                local_path = future.result()
                downloaded_paths.append(local_path)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
    
    print("\n" + "="*80)
    print(f"Downloaded {len(downloaded_paths)}/{num_parts} files")
    print("="*80)
    
    # Extract tar files sequentially (extraction is I/O bound)
    print("\nExtracting tar files...")
    for tar_path in tqdm(downloaded_paths, desc="Extracting"):
        try:
            extract_tar_file(tar_path, extract_dir)
        except Exception as e:
            print(f"✗ Failed to extract {os.path.basename(tar_path)}: {e}")
    
    print("\n" + "="*80)
    print("DOWNLOAD AND EXTRACTION COMPLETE")
    print("="*80)
    print(f"Images extracted to: {extract_dir}")
    
    # Count extracted images
    image_files = list(Path(extract_dir).rglob("*.png")) + list(Path(extract_dir).rglob("*.jpg"))
    print(f"Total images extracted: {len(image_files)}")
    
    return extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/sharegpt4o_raw',
        help='Directory to store downloaded and extracted data'
    )
    parser.add_argument(
        '--num_parts',
        type=int,
        default=10,
        help='Number of tar parts to download (default: 10 for parts 0-9)'
    )
    parser.add_argument(
        '--parallel_downloads',
        type=int,
        default=2,
        help='Number of parallel downloads (default: 2, be careful with rate limits)'
    )
    
    args = parser.parse_args()
    
    extract_dir = download_and_extract_dataset(
        output_dir=args.output_dir,
        num_parts=args.num_parts,
        parallel_downloads=args.parallel_downloads,
    )
    
    print(f"\n✓ Ready for preprocessing!")
    print(f"  Image directory: {extract_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/preprocess_sharegpt4o_final.py \\")
    print(f"    --dataset FreedomIntelligence/ShareGPT-4o-Image \\")
    print(f"    --output_dir ./data/sharegpt4o_bucketed \\")
    print(f"    --bucket_config configs/bucket_config.yaml")