"""
Generate synthetic test images for pipeline validation.
This allows testing the full pipeline without waiting for large dataset downloads.
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont
import json


def generate_test_images(output_dir: str, n_samples: int = 100):
    """Generate simple synthetic images with geometric shapes for testing."""
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    shapes = ['circle', 'square', 'triangle', 'rectangle']
    backgrounds = ['white', 'lightgray', 'lightblue', 'lightyellow']
    
    metadata = []
    
    print(f"Generating {n_samples} synthetic test images...")
    
    for i in range(n_samples):
        # Random properties
        bg_color = random.choice(backgrounds)
        shape = random.choice(shapes)
        color = random.choice(colors)
        size = random.randint(50, 150)
        
        # Create image
        img = Image.new('RGB', (384, 384), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Center position
        cx, cy = 192, 192
        
        # Draw shape
        if shape == 'circle':
            draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=color)
        elif shape == 'square':
            draw.rectangle([cx-size, cy-size, cx+size, cy+size], fill=color)
        elif shape == 'triangle':
            points = [(cx, cy-size), (cx-size, cy+size), (cx+size, cy+size)]
            draw.polygon(points, fill=color)
        elif shape == 'rectangle':
            draw.rectangle([cx-size, cy-size//2, cx+size, cy+size//2], fill=color)
        
        # Add some text
        try:
            draw.text((10, 10), f"Test {i}", fill='black')
        except:
            pass
        
        # Save image
        img_path = os.path.join(images_dir, f"test_{i:04d}.png")
        img.save(img_path)
        
        # Create QA pairs
        questions = [
            f"What shape is in this image?",
            f"What color is the {shape}?",
            f"What is the background color?",
        ]
        answers = [
            shape,
            color,
            bg_color,
        ]
        
        entry = {
            'id': i,
            'image_path': img_path,
            'shape': shape,
            'color': color,
            'background': bg_color,
            'questions': questions,
            'answers': answers,
            # Use first question as default
            'question': questions[0],
            'answer': answers[0],
        }
        metadata.append(entry)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {n_samples} images in {images_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    return metadata


if __name__ == "__main__":
    import sys
    
    output_dir = "/home/omkar/ritesh/data/synthetic_test"
    n_samples = 100
    
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    
    generate_test_images(output_dir, n_samples)
