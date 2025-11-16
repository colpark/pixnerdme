#!/usr/bin/env python3
"""
Quick test script to verify CIFAR-10 setup works correctly.
Tests dataset loading and model instantiation.
"""

import torch
from src.data.dataset.cifar10 import PixCIFAR10, CIFAR10RandomNDataset
from src.models.transformer.pixnerd_c2i import PixNerDiT
from src.models.autoencoder.pixel import PixelAE
from src.models.conditioner.class_label import LabelConditioner


def test_dataset():
    """Test CIFAR-10 dataset loading"""
    print("Testing PixCIFAR10 dataset...")

    # Test training dataset
    dataset = PixCIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        random_crop=True
    )

    print(f"✓ Training dataset loaded: {len(dataset)} samples")

    # Test getting an item
    image, label, metadata = dataset[0]

    assert image.shape == (3, 32, 32), f"Expected shape (3, 32, 32), got {image.shape}"
    assert label >= 0 and label < 10, f"Invalid label: {label}"
    assert "raw_image" in metadata, "Missing raw_image in metadata"
    assert "class" in metadata, "Missing class in metadata"

    print(f"✓ Sample shape: {image.shape}, label: {label}")
    print(f"✓ Image range: [{image.min():.2f}, {image.max():.2f}]")

    # Test random noise dataset
    print("\nTesting CIFAR10RandomNDataset...")
    rand_dataset = CIFAR10RandomNDataset(
        num_classes=10,
        max_num_instances=100,
        latent_shape=(3, 32, 32)
    )

    latent, label, metadata = rand_dataset[0]
    assert latent.shape == (3, 32, 32), f"Expected shape (3, 32, 32), got {latent.shape}"

    print(f"✓ Random dataset loaded: {len(rand_dataset)} samples")
    print(f"✓ Random sample shape: {latent.shape}, label: {label}")


def test_model():
    """Test model instantiation and forward pass"""
    print("\n" + "="*60)
    print("Testing PixNerDiT model for CIFAR-10...")

    # Model parameters from config
    model = PixNerDiT(
        in_channels=3,
        patch_size=4,
        num_groups=8,
        hidden_size=384,
        hidden_size_x=32,
        num_blocks=12,
        num_cond_blocks=10,
        nerf_mlpratio=2,
        num_classes=10,
    )

    print(f"✓ Model instantiated")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.rand(batch_size)
    y = torch.randint(0, 10, (batch_size,))

    print(f"\nTesting forward pass...")
    print(f"  Input shape: {x.shape}")
    print(f"  Timestep shape: {t.shape}")
    print(f"  Label shape: {y.shape}")

    with torch.no_grad():
        output = model(x, t, y)

    assert output.shape == (batch_size, 3, 32, 32), f"Expected output shape (4, 3, 32, 32), got {output.shape}"

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Forward pass successful!")


def test_autoencoder():
    """Test PixelAE"""
    print("\n" + "="*60)
    print("Testing PixelAE...")

    ae = PixelAE(scale=1.0)

    x = torch.randn(2, 3, 32, 32)

    # Encode
    z = ae.encode(x)
    print(f"✓ Encoded shape: {z.shape}")

    # Decode
    x_recon = ae.decode(z)
    print(f"✓ Decoded shape: {x_recon.shape}")

    # Verify it's approximately identity
    diff = (x - x_recon).abs().mean()
    print(f"✓ Reconstruction error: {diff:.6f}")
    assert diff < 1e-5, "PixelAE should be near-identity mapping"


def test_conditioner():
    """Test LabelConditioner"""
    print("\n" + "="*60)
    print("Testing LabelConditioner...")

    conditioner = LabelConditioner(num_classes=10)

    # Conditional
    y = torch.randint(0, 10, (4,))
    cond_emb = conditioner(y, unconditional=False)
    print(f"✓ Conditional embedding shape: {cond_emb.shape}")

    # Unconditional
    uncond_emb = conditioner(y, unconditional=True)
    print(f"✓ Unconditional embedding shape: {uncond_emb.shape}")

    # Verify they're different
    diff = (cond_emb - uncond_emb).abs().mean()
    print(f"✓ Conditional vs unconditional difference: {diff:.4f}")
    assert diff > 0, "Conditional and unconditional should differ"


if __name__ == "__main__":
    print("="*60)
    print("CIFAR-10 Setup Verification")
    print("="*60)

    try:
        test_dataset()
        test_autoencoder()
        test_conditioner()
        test_model()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run training with:")
        print("  python main.py fit -c configs_c2i/cifar_repa_v1.yaml")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
