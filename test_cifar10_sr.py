#!/usr/bin/env python3
"""
Test script for CIFAR-10 Super-Resolution setup
Validates datasets, models, and pipeline configuration
"""
import torch
import sys


def test_datasets():
    """Test SR dataset loading"""
    print("\n" + "="*60)
    print("Testing CIFAR-10 Super-Resolution Datasets")
    print("="*60)

    try:
        from src.data.dataset.cifar10_sr import (
            CIFAR10SuperResDataset,
            CIFAR10SuperResEvalDataset
        )

        # Test training dataset
        print("\n1. Testing CIFAR10SuperResDataset (training)...")
        train_dataset = CIFAR10SuperResDataset(
            root='./data/cifar10',
            train=True,
            download=True,
            lr_size=32,
            hr_size=128
        )

        hr_img, label, metadata = train_dataset[0]

        print(f"   ✓ Dataset loaded: {len(train_dataset)} samples")
        print(f"   ✓ HR image shape: {hr_img.shape} (expected: [3, 128, 128])")
        print(f"   ✓ Class label: {label}")
        print(f"   ✓ Metadata keys: {list(metadata.keys())}")
        print(f"   ✓ LR upsampled shape: {metadata['lr_upsampled'].shape}")

        assert hr_img.shape == (3, 128, 128), f"Wrong HR shape: {hr_img.shape}"
        assert metadata['lr_upsampled'].shape == (3, 128, 128), "Wrong LR upsampled shape"
        assert 'raw_lr' in metadata, "Missing raw_lr in metadata"
        assert metadata['raw_lr'].shape == (3, 32, 32), "Wrong raw LR shape"

        # Test evaluation dataset
        print("\n2. Testing CIFAR10SuperResEvalDataset (evaluation)...")
        eval_dataset = CIFAR10SuperResEvalDataset(
            root='./data/cifar10',
            hr_size=128,
            max_num_instances=100
        )

        hr_img, label, metadata = eval_dataset[0]

        print(f"   ✓ Eval dataset loaded: {len(eval_dataset)} samples")
        print(f"   ✓ HR image shape: {hr_img.shape}")
        print(f"   ✓ Metadata contains lr_upsampled: {'lr_upsampled' in metadata}")

        assert len(eval_dataset) == 100, "Wrong eval dataset size"

        print("\n✅ Dataset tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conditioner():
    """Test image conditioners"""
    print("\n" + "="*60)
    print("Testing Image Conditioners")
    print("="*60)

    try:
        from src.models.conditioner.image_cond import (
            ImageConditioner,
            ConcatImageConditioner
        )

        # Test ConcatImageConditioner
        print("\n1. Testing ConcatImageConditioner...")
        conditioner = ConcatImageConditioner(image_key='lr_upsampled')

        # Create dummy data
        batch_size = 4
        y = [0, 1, 2, 3]  # Class labels
        metadata = {
            'lr_upsampled': torch.randn(batch_size, 3, 128, 128)
        }

        condition, uncondition = conditioner(y, metadata)

        print(f"   ✓ Condition shape: {condition.shape}")
        print(f"   ✓ Uncondition shape: {uncondition.shape}")

        assert condition.shape == (batch_size, 3, 128, 128), "Wrong condition shape"
        assert uncondition.shape == (batch_size, 3, 128, 128), "Wrong uncondition shape"
        assert torch.all(uncondition == 0), "Uncondition should be zeros"

        print("\n✅ Conditioner tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Conditioner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_config():
    """Test model configuration loading"""
    print("\n" + "="*60)
    print("Testing Model Configuration")
    print("="*60)

    try:
        import yaml

        print("\n1. Loading SR config...")
        with open('configs_c2i/cifar_sr_128.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("   ✓ Config loaded successfully")

        # Validate key parameters
        assert config['model']['denoiser']['init_args']['in_channels'] == 6, \
            "Denoiser should have in_channels=6"
        assert config['model']['denoiser']['init_args']['patch_size'] == 8, \
            "Patch size should be 8 for 128×128"
        assert config['data']['train_dataset']['class_path'] == \
            'src.data.dataset.cifar10_sr.CIFAR10SuperResDataset', \
            "Wrong dataset class"

        print(f"   ✓ Denoiser in_channels: {config['model']['denoiser']['init_args']['in_channels']}")
        print(f"   ✓ Patch size: {config['model']['denoiser']['init_args']['patch_size']}")
        print(f"   ✓ Hidden size: {config['model']['denoiser']['init_args']['hidden_size']}")
        print(f"   ✓ Num blocks: {config['model']['denoiser']['init_args']['num_blocks']}")
        print(f"   ✓ Resolution: {config['data']['train_dataset']['init_args']['hr_size']}")

        print("\n✅ Config validation passed!")
        return True

    except Exception as e:
        print(f"\n❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_sampler():
    """Test SR trainer and sampler imports"""
    print("\n" + "="*60)
    print("Testing SR Trainer and Sampler")
    print("="*60)

    try:
        print("\n1. Importing SuperResolutionFlowMatchingTrainer...")
        from src.diffusion.flow_matching.training_sr import SuperResolutionFlowMatchingTrainer
        print("   ✓ Trainer imported successfully")

        print("\n2. Importing SuperResolutionEulerSampler...")
        from src.diffusion.flow_matching.sampling_sr import SuperResolutionEulerSampler
        print("   ✓ Sampler imported successfully")

        print("\n3. Importing SuperResolutionLightningModel...")
        from src.lightning_model_sr import SuperResolutionLightningModel
        print("   ✓ Lightning model imported successfully")

        print("\n✅ Trainer/Sampler tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Trainer/Sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CIFAR-10 Super-Resolution Setup Validation")
    print("="*60)

    results = []

    # Run tests
    results.append(("Datasets", test_datasets()))
    results.append(("Conditioners", test_conditioner()))
    results.append(("Config", test_model_config()))
    results.append(("Trainer/Sampler", test_trainer_sampler()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<30} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed! Ready for SR training.")
        print("\nTo start training:")
        print("  python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
