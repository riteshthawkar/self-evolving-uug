"""
Quick pipeline validation test using synthetic data.
Tests the core understanding loop without full training.
"""

import os
import sys
import json
import torch
from PIL import Image
from tqdm import tqdm

# Monkeypatch torch.is_autocast_enabled to handle device_type arg
# Transformers calls is_autocast_enabled("cuda"), but Torch 2.3 implementation takes no args
_original_is_autocast_enabled = torch.is_autocast_enabled
def _patched_is_autocast_enabled(device=None):
    return _original_is_autocast_enabled()
torch.is_autocast_enabled = _patched_is_autocast_enabled

# Set environment
os.environ['HF_HOME'] = '/home/omkar/ritesh/cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/omkar/ritesh/cache'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add the project root
sys.path.insert(0, "/home/omkar/ritesh")


def test_model_loading(cuda_device: int = 0):
    """Test that model loads correctly."""
    print("\n=== Test 1: Model Loading ===")
    
    # Add BLIP3o to path
    blip3o_path = "/home/omkar/ritesh/BLIP3o"
    if blip3o_path not in sys.path:
        sys.path.insert(0, blip3o_path)
    
    from transformers import AutoTokenizer, AutoProcessor
    
    # Try to load BLIP3o from local path
    model_path = "/fsx/home/jiuhai.chen/BLIP3o-NEXT/models/debug"  # Original path
    
    # Check if model exists locally, otherwise try HF
    if not os.path.exists(model_path):
        # Try Qwen3-VL as fallback for testing
        print("BLIP3o-NEXT not available locally, using Qwen3-VL-2B for testing...")
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir='/home/omkar/ritesh/cache',
            )
            print("  ✓ Processor loaded")
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": cuda_device},
                cache_dir='/home/omkar/ritesh/cache',
            )
            print(f"  ✓ Model loaded on GPU {cuda_device}")
            return model, processor
            
        except Exception as e:
            print(f"  ✗ Qwen2.5-VL failed: {e}")
            
            # Try even simpler model
            print("  Trying BLIP-2 as fallback...")
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            model_name = "Salesforce/blip2-opt-2.7b"
            # Disable fast tokenizer to avoid compatibility issues
            processor = Blip2Processor.from_pretrained(
                model_name,
                cache_dir='/home/omkar/ritesh/cache',
                use_fast=False,
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": cuda_device},
                cache_dir='/home/omkar/ritesh/cache',
            )
            print(f"  ✓ BLIP-2 loaded as fallback")
            return model, processor
    else:
        # Load BLIP3o-NEXT
        print(f"Loading BLIP3o-NEXT from {model_path}...")
        try:
            from blip3o.model import blip3oQwenForInferenceLM
            
            model = blip3oQwenForInferenceLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16
            ).to(f"cuda:{cuda_device}")
            
            processor = AutoTokenizer.from_pretrained(model_path)
            print(f"  ✓ BLIP3o-NEXT loaded on GPU {cuda_device}")
            return model, processor
            
        except Exception as e:
            print(f"  ✗ BLIP3o-NEXT failed: {e}")
            return None, None
    
    return None, None



def test_image_pool():
    """Test image pool with synthetic data."""
    print("\n=== Test 2: Image Pool ===")
    
    from self_evolving.data.image_pool import ImagePool, ImagePoolConfig, BatchDataLoader
    
    data_dir = "/home/omkar/ritesh/data/synthetic_test/images"
    
    config = ImagePoolConfig(
        data_dir=data_dir,
        max_images=50,
    )
    
    pool = ImagePool(config)
    print(f"  ✓ Loaded {len(pool)} images")
    
    # Test batch loading
    dataloader = BatchDataLoader(pool, batch_size=4)
    batch = next(iter(dataloader))
    images, metas = batch
    print(f"  ✓ Batch loading works: {len(images)} images")
    
    return pool


def test_roles(model, processor, device: str):
    """Test proposer and solver roles."""
    print("\n=== Test 3: Roles ===")
    
    from self_evolving.roles.proposer import ProposerRole
    from self_evolving.roles.solver import SolverRole
    
    # Load a test image
    img_path = "/home/omkar/ritesh/data/synthetic_test/images/test_0000.png"
    if not os.path.exists(img_path):
        print("  ✗ Test image not found")
        return None, None
    
    img = Image.open(img_path)
    print(f"  ✓ Loaded test image: {img.size}")
    
    # Test proposer
    print("  Testing Proposer...")
    try:
        proposer = ProposerRole(model, processor, device=device)
        questions = proposer.propose_questions(img, n_questions=2)
        print(f"    ✓ Proposer generated {len(questions)} questions")
        for q in questions[:2]:
            print(f"      - {q}")
    except Exception as e:
        print(f"    ✗ Proposer failed: {e}")
        proposer = None
    
    # Test solver
    print("  Testing Solver...")
    try:
        solver = SolverRole(model, processor, device=device)
        answers, agreement = solver.solve(img, "What shape is in this image?", n_samples=3)
        print(f"    ✓ Solver generated {len(answers)} answers, agreement={agreement:.2f}")
        for a in answers[:3]:
            print(f"      - {a}")
    except Exception as e:
        print(f"    ✗ Solver failed: {e}")
        solver = None
    
    return proposer, solver


def test_understanding_step(proposer, solver, img):
    """Test a full understanding step."""
    print("\n=== Test 4: Understanding Step ===")
    
    if proposer is None or solver is None:
        print("  ✗ Skipped (roles not available)")
        return
    
    try:
        # Proposer generates questions
        questions = proposer.propose_questions(img, n_questions=2)
        
        total_agreement = 0.0
        for q in questions:
            # Solver answers each question
            answers, agreement = solver.solve(img, q, n_samples=5)
            total_agreement += agreement
            print(f"  Q: {q[:50]}...")
            print(f"    Agreement: {agreement:.2f}")
        
        avg_agreement = total_agreement / len(questions) if questions else 0
        print(f"  ✓ Understanding step complete, avg agreement={avg_agreement:.2f}")
        
    except Exception as e:
        print(f"  ✗ Understanding step failed: {e}")


def test_internal_rewards(model, processor):
    """Test internal reward functions."""
    print("\n=== Test 5: Internal Rewards ===")
    
    # Load test image
    img = Image.open("/home/omkar/ritesh/data/synthetic_test/images/test_0000.png")
    
    # Test cycle consistency
    try:
        from self_evolving.rewards import CycleConsistencyReward
        cycle_reward = CycleConsistencyReward(model, processor)
        # CycleConsistencyReward is callable directly
        cycle_score = cycle_reward("a red circle", img)
        print(f"  ✓ Cycle consistency: {cycle_score:.3f}")
    except Exception as e:
        print(f"  ✗ Cycle consistency failed: {e}")
    
    try:
        # Test diversity (need multiple images)
        from self_evolving.rewards import DiversityReward
        
        # Qwen2-VL uses 'visual', others use 'vision_model'
        if hasattr(model, 'visual'):
            vision_encoder = model.visual
        elif hasattr(model, 'vision_model'):
            vision_encoder = model.vision_model
        else:
            vision_encoder = model
            
        diversity_reward = DiversityReward(vision_encoder=vision_encoder, processor=processor)
        images = [Image.open(f"/home/omkar/ritesh/data/synthetic_test/images/test_{i:04d}.png") 
                  for i in range(5)]
        div_score = diversity_reward(images)
        print(f"  ✓ Diversity: {div_score:.3f}")
    except Exception as e:
        print(f"  ✗ Diversity failed: {e}")


def main():
    print("=" * 60)
    print("Self-Evolving Pipeline Validation Test")
    print("=" * 60)
    
    cuda_device = 0  # Use GPU 0 (free)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nUsing GPU {cuda_device}: {torch.cuda.get_device_name(cuda_device)}")
    else:
        print("\nWARNING: No GPU available, using CPU")
    
    # Run tests
    model, processor = test_model_loading(cuda_device)
    
    if model is None:
        print("\n✗ Model loading failed, cannot continue")
        return
    
    pool = test_image_pool()
    
    device = f"cuda:{cuda_device}"
    proposer, solver = test_roles(model, processor, device)
    
    # Load test image
    img = Image.open("/home/omkar/ritesh/data/synthetic_test/images/test_0000.png")
    test_understanding_step(proposer, solver, img)
    
    test_internal_rewards(model, processor)
    
    print("\n" + "=" * 60)
    print("Pipeline Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
