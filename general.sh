python scripts/download_sharegpt4o.py \
  --output_dir ./data/sharegpt4o_raw

# 2. Verify images
python scripts/verify_images.py \
  --image_dir ./data/sharegpt4o_raw/images

# 3. Preprocess buckets
python scripts/preprocess_sharegpt4o_final.py \
  --dataset FreedomIntelligence/ShareGPT-4o-Image \
  --output_dir ./data/sharegpt4o_bucketed \
  --bucket_config configs/bucket_config.yaml
