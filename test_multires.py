#!/usr/bin/env python3
"""
Test script for CIFAR-10 Multi-Resolution Setup
Validates the PixNerd paper's approach: single model, multiple resolutions
"""
import torch
import sys


def test_dataset():
    """Test multi-resolution dataset"""
    print("\n" + "="*60)
    print("Testing Multi-Resolution Dataset")
    print("="*60)

    try:
        from src.data.dataset.cifar10_multires import (
            CIFAR10MultiResDataset,
            CIFAR10MultiResRandomDataset,
            CIFAR10MultiResEvalDataset
        )

        print("\n1. Testing CIFAR10MultiResDataset at 128×128...")
        train_dataset = CIFAR10MultiResDataset(
            root='./data/cifar10',
            train=True,
            download=True,
            target_size=128
        )

        img, label, metadata = train_dataset[0]

        print(f"   ✓ Dataset loaded: {len(train_dataset)} samples")
        print(f"   ✓ Image shape: {img.shape} (expected: [3, 128, 128])")
        print(f"   ✓ Class label: {label}")
        print(f"   ✓ Resolution: {metadata['resolution']}")

        assert img.shape == (3, 128, 128), f"Wrong shape: {img.shape}"
        assert metadata['resolution'] == 128, "Wrong resolution"

        # Test at different resolutions
        print("\n2. Testing at 64×64...")
        dataset_64 = CIFAR10MultiResDataset(
            root='./data/cifar10',
            train=True,
            target_size=64
        )
        img_64, _, meta_64 = dataset_64[0]
        print(f"   ✓ 64×64 image shape: {img_64.shape}")
        assert img_64.shape == (3, 64, 64)

        print("\n3. Testing at 32×32...")
        dataset_32 = CIFAR10MultiResDataset(
            root='./data/cifar10',
            train=True,
            target_size=32
        )
        img_32, _, meta_32 = dataset_32[0]
        print(f"   ✓ 32×32 image shape: {img_32.shape}")
        assert img_32.shape == (3, 32, 32)

        # Test random dataset
        print("\n4. Testing CIFAR10MultiResRandomDataset...")
        random_dataset = CIFAR10MultiResRandomDataset(
            num_classes=10,
            max_num_instances=100,
            target_size=128
        )

        noise, label, metadata = random_dataset[0]
        print(f"   ✓ Random dataset size: {len(random_dataset)}")
        print(f"   ✓ Noise shape: {noise.shape}")
        print(f"   ✓ Auto-adjusted to resolution: {metadata['resolution']}")

        assert noise.shape == (3, 128, 128)
        assert metadata['resolution'] == 128

        print("\n✅ Dataset tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test multi-resolution config"""
    print("\n" + "="*60)
    print("Testing Multi-Resolution Config")
    print("="*60)

    try:
        import yaml

        print("\n1. Loading multi-res config...")
        with open('configs_c2i/cifar_multires_128.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("   ✓ Config loaded successfully")

        # Validate key parameters
        assert config['model']['denoiser']['init_args']['in_channels'] == 3, \
            "Should be 3 channels (no concatenation)"
        assert config['model']['denoiser']['init_args']['patch_size'] == 8, \
            "Patch size should be 8 for 128×128"
        assert config['data']['train_dataset']['init_args']['target_size'] == 128, \
            "Training resolution should be 128×128"

        print(f"   ✓ In channels: {config['model']['denoiser']['init_args']['in_channels']}")
        print(f"   ✓ Patch size: {config['model']['denoiser']['init_args']['patch_size']}")
        print(f"   ✓ Hidden size: {config['model']['denoiser']['init_args']['hidden_size']}")
        print(f"   ✓ Num blocks: {config['model']['denoiser']['init_args']['num_blocks']}")
        print(f"   ✓ Training resolution: {config['data']['train_dataset']['init_args']['target_size']}")

        print("\n✅ Config validation passed!")
        return True

    except Exception as e:
        print(f"\n❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_field_flexibility():
    """Test that neural field encoding works at different resolutions"""
    print("\n" + "="*60)
    print("Testing Neural Field Resolution Flexibility")
    print("="*60)

    try:
        from src.models.transformer.pixnerd_c2i import PixNerDiT

        print("\n1. Creating PixNerDiT model...")
        model = PixNerDiT(
            in_channels=3,
            patch_size=8,
            hidden_size=768,
            hidden_size_x=64,
            num_blocks=20,
            num_cond_blocks=16,
            nerf_mlpratio=2,
            num_classes=10
        )

        print("   ✓ Model created")

        # Test forward pass at different resolutions
        batch_size = 2
        num_classes = 10

        print("\n2. Testing forward pass at 128×128...")
        x_128 = torch.randn(batch_size, 3, 128, 128)
        t_128 = torch.rand(batch_size)
        y_128 = torch.randint(0, num_classes, (batch_size,))

        with torch.no_grad():
            out_128 = model(x_128, t_128, y_128)

        print(f"   ✓ Input shape: {x_128.shape}")
        print(f"   ✓ Output shape: {out_128.shape}")
        assert out_128.shape == x_128.shape

        print("\n3. Testing SAME model at 64×64...")
        x_64 = torch.randn(batch_size, 3, 64, 64)
        t_64 = torch.rand(batch_size)
        y_64 = torch.randint(0, num_classes, (batch_size,))

        with torch.no_grad():
            out_64 = model(x_64, t_64, y_64)

        print(f"   ✓ Input shape: {x_64.shape}")
        print(f"   ✓ Output shape: {out_64.shape}")
        assert out_64.shape == x_64.shape

        print("\n4. Testing SAME model at 32×32...")
        x_32 = torch.randn(batch_size, 3, 32, 32)
        t_32 = torch.rand(batch_size)
        y_32 = torch.randint(0, num_classes, (batch_size,))

        with torch.no_grad():
            out_32 = model(x_32, t_32, y_32)

        print(f"   ✓ Input shape: {x_32.shape}")
        print(f"   ✓ Output shape: {out_32.shape}")
        assert out_32.shape == x_32.shape

        print("\n🎯 Key Result: SAME model works at all resolutions!")
        print("   This validates the neural field approach from the paper.")

        print("\n✅ Neural field flexibility test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Neural field test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CIFAR-10 Multi-Resolution Setup Validation")
    print("Following PixNerd Paper Approach")
    print("="*60)

    results = []

    # Run tests
    results.append(("Dataset", test_dataset()))
    results.append(("Config", test_config()))
    results.append(("Neural Field Flexibility", test_neural_field_flexibility()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed! Ready for multi-resolution training.")
        print("\nTo start training:")
        print("  python main.py fit -c configs_c2i/cifar_multires_128.yaml")
        print("\nAfter training, generate at multiple resolutions:")
        print("  python generate_multires.py --ckpt <checkpoint> --multi")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
