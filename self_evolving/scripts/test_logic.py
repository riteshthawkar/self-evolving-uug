"""
Minimal pipeline validation without model inference.
Tests the logic of all components without requiring model weights.
"""

import os
import sys
import json
from PIL import Image

sys.path.insert(0, "/home/omkar/ritesh")


def test_image_pool():
    """Test image pool loading."""
    print("\n=== Test 1: Image Pool ===")
    
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
    
    # Verify images are PIL
    assert all(isinstance(img, Image.Image) for img in images)
    print(f"  ✓ All images are PIL.Image")
    
    return True


def test_reward_logic():
    """Test reward computation logic without model."""
    print("\n=== Test 2: Reward Logic ===")
    
    # Test agreement computation
    from self_evolving.roles.solver import SolverRole
    
    # Mock solver - just test the logic
    class MockModel:
        pass
    
    class MockProcessor:
        pass
    
    mock_solver = SolverRole(MockModel(), MockProcessor(), device="cpu")
    
    # Test agreement reward
    answers1 = ["cat", "cat", "cat", "dog", "cat"]
    reward1 = mock_solver.compute_agreement_reward(answers1)
    print(f"  Agreement for ['cat'x4, 'dog'x1]: {reward1:.2f}")
    assert 0.7 <= reward1 <= 0.9
    
    answers2 = ["red", "blue", "green", "yellow", "purple"]
    reward2 = mock_solver.compute_agreement_reward(answers2)
    print(f"  Agreement for all different: {reward2:.2f}")
    assert reward2 <= 0.3
    
    answers3 = ["yes", "yes", "yes", "yes", "yes"]
    reward3 = mock_solver.compute_agreement_reward(answers3)
    print(f"  Agreement for all same: {reward3:.2f}")
    assert reward3 == 1.0
    
    print(f"  ✓ Agreement reward logic works")
    
    # Test entropy
    entropy1 = mock_solver.compute_entropy(answers1)
    entropy2 = mock_solver.compute_entropy(answers2)
    entropy3 = mock_solver.compute_entropy(answers3)
    print(f"  Entropy: high agreement={entropy1:.2f}, all different={entropy2:.2f}, all same={entropy3:.2f}")
    assert entropy3 < entropy1 < entropy2
    print(f"  ✓ Entropy logic works")
    
    return True


def test_proposer_parsing():
    """Test proposer output parsing."""
    print("\n=== Test 3: Proposer Parsing ===")
    
    from self_evolving.roles.proposer import ProposerRole
    
    class MockModel:
        pass
    
    class MockProcessor:
        pass
    
    mock_proposer = ProposerRole(MockModel(), MockProcessor(), device="cpu")
    
    # Test question parsing
    test_output = """
    1. What color is the main object?
    2. How many items are visible in the image?
    3. Is there a person in the scene?
    4. What is the weather like?
    Not a question without ending mark
    5. Where is the object located?
    """
    
    questions = mock_proposer._parse_questions(test_output)
    print(f"  Parsed {len(questions)} questions from test output")
    for q in questions:
        print(f"    - {q}")
    
    assert len(questions) >= 4
    assert all(q.endswith('?') for q in questions)
    print(f"  ✓ Question parsing works")
    
    # Test creative output parsing
    creative_output = """
    PROMPT: A red car on a sunny day
    Q1: Is there a car?
    A1: Yes
    Q2: What color is the car?
    A2: Red
    Q3: What is the weather?
    A3: Sunny
    """
    
    prompt, spec = mock_proposer._parse_creative_output(creative_output)
    print(f"  Parsed prompt: {prompt}")
    print(f"  Parsed {len(spec.questions)} verification questions")
    assert prompt == "A red car on a sunny day"
    assert len(spec.questions) >= 2
    print(f"  ✓ Creative output parsing works")
    
    return True


def test_edit_instruction_parsing():
    """Test edit instruction parsing."""
    print("\n=== Test 4: Edit Instruction Parsing ===")
    
    from self_evolving.roles.editor import EditorRole, EditInstruction
    
    class MockModel:
        pass
    
    class MockProcessor:
        pass
    
    # Test EditInstruction dataclass
    instruction = EditInstruction(
        instruction="Change the bird color from blue to red",
        target_attribute="bird color",
        old_value="blue",
        new_value="red",
        edit_success_questions=["Is the bird red?", "Has the color changed?"],
        preservation_questions=["Is the background the same?", "Is the bird still present?"],
    )
    
    print(f"  EditInstruction: {instruction.instruction}")
    print(f"  Target: {instruction.target_attribute}: {instruction.old_value} → {instruction.new_value}")
    print(f"  Success questions: {len(instruction.edit_success_questions)}")
    print(f"  Preservation questions: {len(instruction.preservation_questions)}")
    
    assert instruction.target_attribute == "bird color"
    assert len(instruction.edit_success_questions) == 2
    print(f"  ✓ EditInstruction dataclass works")
    
    return True


def test_rl_controller_logic():
    """Test RL controller logic."""
    print("\n=== Test 5: RL Controller Logic ===")
    
    from self_evolving.rl_controller import RLControllerConfig, EvoLMMReward
    
    # Test config
    config = RLControllerConfig(
        kl_coeff=0.01,
        kl_target=1.0,
        kl_adaptive=True,
    )
    print(f"  RLControllerConfig: β={config.kl_coeff}, target={config.kl_target}")
    
    # Test EvoLMM reward
    reward_fn = EvoLMMReward(
        entropy_target=1.0,
        entropy_sigma=0.5,
    )
    
    # Test solver reward (uses answers list)
    agreement_reward, entropy = reward_fn.compute_solver_reward(["yes", "yes", "yes", "no", "yes"])
    print(f"  Solver reward (4/5 agree): reward={agreement_reward:.2f}, entropy={entropy:.2f}")
    assert 0.7 <= agreement_reward <= 0.9
    
    # Test proposer reward (uses entropy band)
    proposer_reward_good = reward_fn.compute_proposer_reward(solver_entropy=1.0)  # At target
    proposer_reward_low = reward_fn.compute_proposer_reward(solver_entropy=0.0)  # Too easy
    proposer_reward_high = reward_fn.compute_proposer_reward(solver_entropy=3.0) # Too hard
    
    print(f"  Proposer reward: at target={proposer_reward_good:.2f}, too low={proposer_reward_low:.2f}, too high={proposer_reward_high:.2f}")
    assert proposer_reward_good > proposer_reward_low
    assert proposer_reward_good > proposer_reward_high
    print(f"  ✓ RL Controller logic works")
    
    return True


def test_crossplay_logic():
    """Test cross-play evaluator logic."""
    print("\n=== Test 6: Cross-Play Logic ===")
    
    from self_evolving.judge import CrossPlayEvaluator
    
    evaluator = CrossPlayEvaluator(
        max_snapshots=5,
        divergence_threshold=0.3,
    )
    
    print(f"  Created CrossPlayEvaluator with max_snapshots={evaluator.max_snapshots}")
    
    # Test divergence rate
    evaluator.evaluation_count = 100
    evaluator.divergence_history = [
        {'step': 80, 'divergence': 0.4, 'current': 0.9, 'historical_avg': 0.5},
        {'step': 85, 'divergence': 0.35, 'current': 0.85, 'historical_avg': 0.5},
        {'step': 95, 'divergence': 0.5, 'current': 0.95, 'historical_avg': 0.45},
    ]
    
    div_rate = evaluator.get_divergence_rate(last_n=20)
    print(f"  Divergence rate (last 20): {div_rate:.2f}")
    
    should_reset = evaluator.should_reset_judge()
    print(f"  Should reset judge: {should_reset}")
    
    print(f"  ✓ Cross-play logic works")
    
    return True


def test_internal_reward_modules():
    """Test internal reward module imports."""
    print("\n=== Test 7: Internal Reward Modules ===")
    
    try:
        from self_evolving.rewards import CycleConsistencyReward, VerificationReward, DiversityReward
        print(f"  ✓ CycleConsistencyReward imported")
        print(f"  ✓ VerificationReward imported")
        print(f"  ✓ DiversityReward imported")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("Self-Evolving Pipeline Logic Validation")
    print("(Tests core logic without model inference)")
    print("=" * 60)
    
    tests = [
        ("Image Pool", test_image_pool),
        ("Reward Logic", test_reward_logic),
        ("Proposer Parsing", test_proposer_parsing),
        ("Edit Instruction Parsing", test_edit_instruction_parsing),
        ("RL Controller Logic", test_rl_controller_logic),
        ("Cross-Play Logic", test_crossplay_logic),
        ("Internal Reward Modules", test_internal_reward_modules),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
