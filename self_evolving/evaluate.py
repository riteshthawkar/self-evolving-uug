"""
Benchmark validation script for self-evolving BLIP3o-family models.
Evaluates trained models on standard understanding and generation benchmarks.

Benchmarks:
- Understanding: ChartQA, MathVista, DocVQA (EvoLMM suite)
- Generation: GenEval, prompt-following tests (BLIP3o-family suite)
"""

import os
import sys
import argparse
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from PIL import Image

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    # Model
    model_path: str = ""
    base_model: str = "BLIP3o/BLIP3o-Model-8B"
    
    # Benchmarks to run
    run_understanding: bool = True
    run_generation: bool = True
    
    # Understanding benchmarks
    understanding_benchmarks: List[str] = field(default_factory=lambda: [
        "chartqa",
        "mathvista",
        "docvqa",
    ])
    
    # Generation benchmarks
    generation_benchmarks: List[str] = field(default_factory=lambda: [
        "geneval",
        "prompt_following",
    ])
    
    # Evaluation settings
    max_samples: Optional[int] = None
    batch_size: int = 4
    
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Output
    output_dir: str = "./eval_results"
    
    # Device
    cuda_device: int = 0


class UnderstandingBenchmark:
    """
    Benchmark for understanding tasks (VQA).
    Uses internal self-consistency as evaluation signal.
    """
    
    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Import solver for evaluation
        from self_evolving.roles.solver import SolverRole
        self.solver = SolverRole(model, processor, device=device)

    def evaluate_sample(
        self,
        image: Image.Image,
        question: str,
        ground_truth: Optional[str] = None,
        n_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate a single VQA sample.
        
        Uses self-consistency (agreement among samples) as primary metric.
        If ground truth is provided, also computes accuracy.
        """
        # Get multiple answers
        answers, agreement = self.solver.solve(
            image=image,
            question=question,
            n_samples=n_samples,
        )
        
        # Get most common answer
        from collections import Counter
        counter = Counter(answers)
        predicted = counter.most_common(1)[0][0] if counter else ""
        
        result = {
            'question': question,
            'predicted': predicted,
            'agreement': agreement,
            'all_answers': answers,
        }
        
        # Check accuracy if ground truth provided
        if ground_truth is not None:
            # Normalize for comparison
            pred_norm = predicted.lower().strip()
            gt_norm = ground_truth.lower().strip()
            
            # Exact match
            exact_match = pred_norm == gt_norm
            
            # Relaxed match (contains)
            relaxed_match = gt_norm in pred_norm or pred_norm in gt_norm
            
            result['ground_truth'] = ground_truth
            result['exact_match'] = exact_match
            result['relaxed_match'] = relaxed_match
        
        return result

    def evaluate_dataset(
        self,
        samples: List[Dict],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset of samples.
        
        Each sample should have: image, question, (optional) ground_truth
        """
        if max_samples:
            samples = samples[:max_samples]
        
        results = []
        total_agreement = 0.0
        exact_matches = 0
        relaxed_matches = 0
        has_gt = False
        
        for sample in tqdm(samples, desc="Evaluating understanding"):
            result = self.evaluate_sample(
                image=sample['image'],
                question=sample['question'],
                ground_truth=sample.get('ground_truth'),
            )
            results.append(result)
            total_agreement += result['agreement']
            
            if 'exact_match' in result:
                has_gt = True
                if result['exact_match']:
                    exact_matches += 1
                if result['relaxed_match']:
                    relaxed_matches += 1
        
        n = len(results)
        metrics = {
            'num_samples': n,
            'avg_agreement': total_agreement / n if n > 0 else 0.0,
        }
        
        if has_gt:
            metrics['exact_accuracy'] = exact_matches / n if n > 0 else 0.0
            metrics['relaxed_accuracy'] = relaxed_matches / n if n > 0 else 0.0
        
        return {
            'metrics': metrics,
            'results': results,
        }


class GenerationBenchmark:
    """
    Benchmark for generation tasks.
    Uses internal rewards (cycle consistency, verification) as evaluation.
    """
    
    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Import roles
        from self_evolving.roles.generator import GeneratorRole
        from self_evolving.roles.solver import SolverRole
        from self_evolving.rewards.cycle_consistency import CycleConsistencyReward
        from self_evolving.rewards.verification import VerificationReward
        
        self.generator = GeneratorRole(
            model, processor, device=device, freeze_diffusion=True
        )
        self.solver = SolverRole(model, processor, device=device)
        self.cycle_reward = CycleConsistencyReward(model, processor)
        self.verification_reward = VerificationReward(model, processor)

    def evaluate_prompt(
        self,
        prompt: str,
        verification_questions: Optional[List[str]] = None,
        n_samples: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """
        Evaluate generation for a single prompt.
        
        Metrics:
        - Cycle consistency: prompt -> image -> caption similarity
        - Verification: answers to verification questions
        - Diversity: variance across samples
        """
        # Generate images
        images = self.generator.generate(
            prompt=prompt,
            n_samples=n_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        if not images:
            return {
                'prompt': prompt,
                'cycle_consistency': 0.0,
                'verification': 0.0,
                'num_generated': 0,
            }
        
        # Compute cycle consistency for each image
        cycle_scores = []
        for img in images:
            score = self.cycle_reward.compute_cycle_consistency(prompt, img)
            cycle_scores.append(score)
        
        # Compute verification if questions provided
        verif_scores = []
        if verification_questions:
            for img in images:
                for q in verification_questions:
                    answers, agreement = self.solver.solve(img, q, n_samples=5)
                    verif_scores.append(agreement)
        
        result = {
            'prompt': prompt,
            'cycle_consistency': sum(cycle_scores) / len(cycle_scores),
            'num_generated': len(images),
            'per_image_cycle': cycle_scores,
        }
        
        if verif_scores:
            result['verification'] = sum(verif_scores) / len(verif_scores)
        
        return result

    def evaluate_dataset(
        self,
        prompts: List[Dict],
        max_samples: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset of prompts.
        
        Each prompt should have: prompt, (optional) verification_questions
        """
        if max_samples:
            prompts = prompts[:max_samples]
        
        results = []
        total_cycle = 0.0
        total_verif = 0.0
        verif_count = 0
        
        for sample in tqdm(prompts, desc="Evaluating generation"):
            result = self.evaluate_prompt(
                prompt=sample['prompt'],
                verification_questions=sample.get('verification_questions'),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            results.append(result)
            total_cycle += result['cycle_consistency']
            
            if 'verification' in result:
                total_verif += result['verification']
                verif_count += 1
        
        n = len(results)
        metrics = {
            'num_prompts': n,
            'avg_cycle_consistency': total_cycle / n if n > 0 else 0.0,
        }
        
        if verif_count > 0:
            metrics['avg_verification'] = total_verif / verif_count
        
        return {
            'metrics': metrics,
            'results': results,
        }


def create_synthetic_understanding_benchmark(n_samples: int = 100) -> List[Dict]:
    """
    Create synthetic understanding samples for testing.
    In practice, use real benchmark datasets.
    """
    # Placeholder - in real implementation, load from actual benchmarks
    questions = [
        "What is in this image?",
        "Describe the main object.",
        "What color is the background?",
        "How many objects are visible?",
        "What is the mood of this image?",
    ]
    
    samples = []
    for i in range(n_samples):
        # Create placeholder image
        img = Image.new('RGB', (224, 224), color=(i % 256, (i * 2) % 256, (i * 3) % 256))
        samples.append({
            'image': img,
            'question': questions[i % len(questions)],
        })
    
    return samples


def create_synthetic_generation_benchmark(n_samples: int = 50) -> List[Dict]:
    """
    Create synthetic generation prompts for testing.
    In practice, use GenEval or similar benchmarks.
    """
    prompts = [
        {
            'prompt': "A red apple on a wooden table",
            'verification_questions': ["Is there an apple?", "Is the apple red?", "Is there a table?"],
        },
        {
            'prompt': "A cat sitting on a windowsill at sunset",
            'verification_questions': ["Is there a cat?", "Is it sunset?"],
        },
        {
            'prompt': "A futuristic city with flying cars",
            'verification_questions': ["Is this a city?", "Are there flying vehicles?"],
        },
        {
            'prompt': "A peaceful mountain lake with snow-capped peaks",
            'verification_questions': ["Is there a lake?", "Are there mountains?"],
        },
    ]
    
    samples = []
    for i in range(n_samples):
        samples.append(prompts[i % len(prompts)])
    
    return samples


def run_evaluation(config: BenchmarkConfig):
    """Run full benchmark evaluation."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {config.model_path or config.base_model}")
    
    device = torch.device(f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)
    
    try:
        model = AutoModel.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": config.cuda_device},
            trust_remote_code=True,
        )
        if not hasattr(model, "generate"):
            raise RuntimeError("AutoModel result has no `.generate` method")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": config.cuda_device},
            trust_remote_code=True,
        )
    
    # Load LoRA weights if provided
    if config.model_path and os.path.exists(config.model_path):
        print(f"Loading LoRA weights from: {config.model_path}")
        model = PeftModel.from_pretrained(model, config.model_path)
    
    results = {}
    
    # Understanding benchmarks
    if config.run_understanding:
        print("\n=== Running Understanding Benchmarks ===")
        
        understanding_bench = UnderstandingBenchmark(model, processor, str(device))
        
        # Create synthetic samples (replace with real benchmark loading)
        samples = create_synthetic_understanding_benchmark(
            n_samples=config.max_samples or 100
        )
        
        understanding_results = understanding_bench.evaluate_dataset(
            samples, max_samples=config.max_samples
        )
        
        results['understanding'] = understanding_results
        
        print(f"\nUnderstanding Results:")
        for k, v in understanding_results['metrics'].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Generation benchmarks
    if config.run_generation:
        print("\n=== Running Generation Benchmarks ===")
        
        generation_bench = GenerationBenchmark(model, processor, str(device))
        
        # Create synthetic prompts (replace with real benchmark loading)
        prompts = create_synthetic_generation_benchmark(
            n_samples=config.max_samples or 50
        )
        
        generation_results = generation_bench.evaluate_dataset(
            prompts,
            max_samples=config.max_samples,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        )
        
        results['generation'] = generation_results
        
        print(f"\nGeneration Results:")
        for k, v in generation_results['metrics'].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    results_path = os.path.join(config.output_dir, "benchmark_results.json")
    
    # Convert results to serializable format
    def make_serializable(obj):
        if isinstance(obj, Image.Image):
            return "<PIL.Image>"
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for self-evolving BLIP3o-family models")
    parser.add_argument("--model_path", type=str, default="", help="Path to trained LoRA weights")
    parser.add_argument("--base_model", type=str, default="BLIP3o/BLIP3o-Model-8B")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--run_understanding", action="store_true", default=True)
    parser.add_argument("--run_generation", action="store_true", default=True)
    parser.add_argument("--skip_understanding", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_path=args.model_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        cuda_device=args.cuda_device,
        run_understanding=not args.skip_understanding,
        run_generation=not args.skip_generation,
    )
    
    run_evaluation(config)


if __name__ == "__main__":
    main()
