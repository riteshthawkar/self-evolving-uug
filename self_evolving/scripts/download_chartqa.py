"""
Download and prepare ChartQA dataset for testing.
Saves images locally to avoid re-downloading.
"""

import os
import sys
from PIL import Image
from tqdm import tqdm

# Set cache directory
os.environ['HF_HOME'] = '/home/omkar/ritesh/cache'
os.environ['HF_DATASETS_CACHE'] = '/home/omkar/ritesh/cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/omkar/ritesh/cache'

from datasets import load_dataset
import json

def download_chartqa(output_dir: str, n_samples: int = 500):
    """Download ChartQA and save images locally."""
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Downloading ChartQA dataset (first {n_samples} samples)...")
    
    try:
        # Try loading with streaming to avoid full download
        ds = load_dataset(
            'ahmed-masry/ChartQA', 
            split=f'train[:{n_samples}]',
            cache_dir='/home/omkar/ritesh/cache',
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative: HuggingFace datasets with streaming...")
        ds = load_dataset(
            'ahmed-masry/ChartQA', 
            split='train',
            streaming=True,
            cache_dir='/home/omkar/ritesh/cache',
            trust_remote_code=True,
        )
        # Take first n_samples
        samples = []
        for i, sample in enumerate(ds):
            if i >= n_samples:
                break
            samples.append(sample)
        ds = samples
    
    print(f"Processing {len(ds)} samples...")
    
    # Save samples
    metadata = []
    
    for i, sample in enumerate(tqdm(ds, desc="Saving images")):
        try:
            # Get image
            if 'image' in sample:
                img = sample['image']
                if isinstance(img, Image.Image):
                    # Save image
                    img_path = os.path.join(images_dir, f"chart_{i:04d}.png")
                    img.save(img_path)
                    
                    # Create metadata entry
                    entry = {
                        'id': i,
                        'image_path': img_path,
                        'question': sample.get('query', sample.get('question', '')),
                        'answer': sample.get('label', sample.get('answer', '')),
                    }
                    metadata.append(entry)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved {len(metadata)} samples to {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Images: {images_dir}")
    
    return metadata


if __name__ == "__main__":
    output_dir = "/home/omkar/ritesh/data/chartqa"
    n_samples = 500
    
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    
    download_chartqa(output_dir, n_samples)
