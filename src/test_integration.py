"""
Dry-run test to validate TwinFlow + RL integration without full training.
Tests all code paths with minimal data/computation.
"""

import torch
import sys
from omegaconf import OmegaConf
from services.tools import create_logger
from networks import MODELS
from methodes import METHODES

logger = create_logger(__name__)

def test_reward_model():
    """Test reward model loading and inference"""
    logger.info("=" * 50)
    logger.info("TEST 1: Reward Model Loading")
    logger.info("=" * 50)
    
    try:
        from services.reward_models import RewardModelWrapper
        
        # Test with dummy data
        reward_model = RewardModelWrapper(
            reward_type="hpsv2",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Create dummy image and prompt
        dummy_image = torch.randn(2, 3, 256, 256).cuda()  # 2 images
        dummy_prompts = ["a photo of a cat", "a photo of a dog"]
        
        rewards = reward_model.compute_reward(dummy_image, dummy_prompts)
        
        assert rewards.shape == (2,), f"Expected shape (2,), got {rewards.shape}"
        assert torch.isfinite(rewards).all(), "Rewards contain NaN/Inf"
        
        logger.info(f"✓ Reward model loaded: {reward_model.reward_type}")
        logger.info(f"✓ Reward scores: {rewards.cpu().numpy()}")
        logger.info(f"✓ Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        
        del reward_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Reward model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_twinflow_init():
    """Test TwinFlow initialization with RL params"""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: TwinFlow Initialization")
    logger.info("=" * 50)
    
    try:
        method_config = {
            "method_type": "TwinFlow",
            "consistc_ratio": 1.0,
            "enhanced_ratio": 0.8,
            "ema_decay_rate": 0.0,
            "enhanced_range": [0.0, 1.0],
            "time_dist_ctrl": [1.0, 1.0, 1.0],
            "estimate_order": 2,
            "loss_func_type": {"type": "barron_reweighting", "alpha": 1.0},
            "dist_match_cof": 0.5,
            "use_image_free": False,
            # RL params
            "use_rl": True,
            "rl_warmup_steps": 100,  # Low for testing
            "rl_weight": 0.05,
            "reward_model_type": "hpsv2",
            "reward_model_cache": "./test_cache",
            # Dynamic renoise
            "use_dynamic_renoise": True,
            "renoise_schedule": [4.0, 1.0, 200],
        }
        
        method = METHODES["TwinFlow"](**method_config)
        
        assert hasattr(method, 'use_rl'), "Missing use_rl attribute"
        assert hasattr(method, 'rl_warmup_steps'), "Missing rl_warmup_steps"
        assert hasattr(method, 'compute_rl_gradients'), "Missing compute_rl_gradients method"
        assert hasattr(method, 'get_dynamic_renoise_bias'), "Missing dynamic renoise method"
        
        logger.info("✓ TwinFlow initialized with RL params")
        logger.info(f"  - use_rl: {method.use_rl}")
        logger.info(f"  - rl_warmup_steps: {method.rl_warmup_steps}")
        logger.info(f"  - rl_weight: {method.rl_weight}")
        logger.info(f"  - use_dynamic_renoise: {method.use_dynamic_renoise}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TwinFlow init test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_renoise():
    """Test dynamic renoise scheduling"""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 3: Dynamic Renoise Scheduling")
    logger.info("=" * 50)
    
    try:
        method_config = {
            "use_dynamic_renoise": True,
            "renoise_schedule": [4.0, 1.0, 1000],
            "time_dist_ctrl": [1.0, 1.0, 1.0],
            "consistc_ratio": 1.0,
            "enhanced_ratio": 0.0,
            "ema_decay_rate": 0.0,
            "enhanced_range": [0.0, 1.0],
            "estimate_order": 2,
            "loss_func_type": {"type": "l2"},
            "dist_match_cof": 0.5,
        }
        
        method = METHODES["TwinFlow"](**method_config)
        
        # Test bias schedule
        steps = [0, 500, 1000, 2000]
        biases = [method.get_dynamic_renoise_bias(s) for s in steps]
        
        logger.info("✓ Dynamic renoise bias schedule:")
        for s, b in zip(steps, biases):
            logger.info(f"  Step {s:4d}: bias = {b:.4f}")
        
        # Should decay from 4.0 -> 1.0
        assert biases[0] == 4.0, f"Expected 4.0 at step 0, got {biases[0]}"
        assert biases[-1] == 1.0, f"Expected 1.0 at step 2000, got {biases[-1]}"
        assert biases[0] > biases[1] > biases[2], "Bias should monotonically decrease"
        
        # Test sampling with dynamic bias
        dummy_x = torch.randn(4, 3, 64, 64)
        t_early = method.sample_renoise_time_dynamic(dummy_x, step=0)
        t_late = method.sample_renoise_time_dynamic(dummy_x, step=2000)
        
        logger.info(f"✓ Early step t mean: {t_early.mean():.4f} (should be > 0.5)")
        logger.info(f"✓ Late step t mean: {t_late.mean():.4f} (should be ~0.5)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dynamic renoise test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step_forward_pass():
    """Test full training_step with dummy model and data"""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 4: Training Step Forward Pass")
    logger.info("=" * 50)
    
    try:
        # Create minimal config
        config = {
            "model": {
                "model_name": "QwenImage",
                "aux_time_embed": True,
                "model_path": "Qwen/Qwen-Image-2512",  # Small model for testing
            },
            "method": {
                "method_type": "TwinFlow",
                "consistc_ratio": 1.0,
                "enhanced_ratio": 0.0,  # Disable for speed
                "ema_decay_rate": 0.0,
                "enhanced_range": [0.0, 1.0],
                "time_dist_ctrl": [1.0, 1.0, 1.0],
                "estimate_order": 0,  # Simplest mode
                "loss_func_type": {"type": "l2"},
                "dist_match_cof": 0.5,
                "use_rl": True,
                "rl_warmup_steps": 50,
                "rl_weight": 0.05,
                "reward_model_type": "hpsv2",
                "use_dynamic_renoise": False,  # Disable for speed
            }
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model (you may need to adjust this based on your MODELS dict)
        logger.info("Loading model (this may take a moment)...")
        
        wrapped_model = MODELS[config["model"]["model_name"]](
            model_id=config["model"]["model_path"],
            aux_time_embed=config["model"]["aux_time_embed"],
            text_dtype=torch.float32,      # ← changed
            imgs_dtype=torch.float32,      # ← changed
        )
        wrapped_model.transformer.to(device)
        wrapped_model.transformer.eval()  # Eval mode for testing
        
        # Initialize method
        method = METHODES["TwinFlow"](**config["method"],
            vae_for_rl=wrapped_model.vae if hasattr(wrapped_model, 'vae') else None,)
        
        # Create dummy batch
        batch_size = 4
        # Create dummy text embeddings
        seq_len = 77
        hidden_dim = 3584  # Adjust based on your model
        
        latents = torch.randn(batch_size, 16, 64, 64, dtype=torch.float32).to(device)  # ← float32
        prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).to(device)
        prompt_mask = torch.ones(batch_size, seq_len, dtype=torch.float32).to(device)
        uncond_embeds = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).to(device)
        uncond_mask = torch.ones(batch_size, seq_len, dtype=torch.float32).to(device)

        # Dummy prompts
        prompts = [
            "a photo of a cat",
            "a photo of a dog",
            "a beautiful landscape",
            "a portrait of a person"
        ]
        
        # Test different steps (before and after RL warmup)
        test_steps = [10, 60, 100]  # Before warmup, after warmup, later
        
        for step in test_steps:
            logger.info(f"\n--- Testing step {step} ---")
            
            with torch.no_grad():
                loss = method.training_step(
                    wrapped_model,
                    latents,
                    c=[prompt_embeds, prompt_mask],
                    e=[uncond_embeds, uncond_mask],
                    step=step,
                    v=None,
                    prompts=prompts,
                )
            
            assert isinstance(loss, torch.Tensor), f"Loss should be tensor, got {type(loss)}"
            assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
            assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss}"
            
            logger.info(f"✓ Step {step}: loss = {loss.item():.6f}")
            
            # Check if RL activated
            if step >= method.rl_warmup_steps:
                logger.info(f"  RL should be active (step {step} >= warmup {method.rl_warmup_steps})")
            else:
                logger.info(f"  RL should be inactive (step {step} < warmup {method.rl_warmup_steps})")
        
        logger.info("\n✓ All training steps completed successfully")
        
        del wrapped_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow properly through RL integration"""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 5: Gradient Flow Check")
    logger.info("=" * 50)
    
    try:
        from services.reward_models import compute_reward_gradients, RewardModelWrapper
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create reward model
        reward_model = RewardModelWrapper(reward_type="hpsv2", device=device)
        
        # Create dummy samples requiring grad
        #fake_samples = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
        fake_samples = torch.randn(2, 3, 256, 256, dtype=torch.float32, device=device, requires_grad=True)
        prompts = ["a cat", "a dog"]
        
        # Compute RL gradients
        rl_grads = compute_reward_gradients(reward_model, fake_samples, prompts)
        
        assert rl_grads.shape == fake_samples.shape, "Gradient shape mismatch"
        assert not rl_grads.requires_grad, "RL grads should be detached"
        assert torch.isfinite(rl_grads).all(), "RL grads contain NaN/Inf"
        
        logger.info("✓ RL gradients computed successfully")
        logger.info(f"  Gradient norm: {rl_grads.norm():.6f}")
        logger.info(f"  Gradient range: [{rl_grads.min():.6f}, {rl_grads.max():.6f}]")
        
        del reward_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests"""
    logger.info("\n" + "=" * 70)
    logger.info("TWINFLOW + RL INTEGRATION VALIDATION SUITE")
    logger.info("=" * 70 + "\n")
    
    tests = [
        ("Reward Model", test_reward_model),
        ("TwinFlow Init", test_twinflow_init),
        ("Dynamic Renoise", test_dynamic_renoise),
        ("Gradient Flow", test_gradient_flow),
        ("Training Step", test_training_step_forward_pass),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Fatal error in {test_name}: {e}")
            results[test_name] = False
        
        # Clear GPU memory between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status:8s} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALL TESTS PASSED - READY FOR TRAINING 🎉")
        logger.info("=" * 70)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("❌ SOME TESTS FAILED - FIX BEFORE TRAINING ❌")
        logger.error("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)