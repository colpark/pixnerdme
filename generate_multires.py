#!/usr/bin/env python3
"""
Multi-Resolution Generation with Single PixNerd Checkpoint

Demonstrates PixNerd's key feature: ONE model can generate at multiple resolutions
thanks to neural field positional encoding with normalized [0,1] coordinates.

Usage:
    # Generate at 32×32
    python generate_multires.py --ckpt path/to/checkpoint.ckpt --resolution 32 --num-images 100

    # Generate at 64×64 (SAME checkpoint)
    python generate_multires.py --ckpt path/to/checkpoint.ckpt --resolution 64 --num-images 100

    # Generate at 128×128 (SAME checkpoint)
    python generate_multires.py --ckpt path/to/checkpoint.ckpt --resolution 128 --num-images 100
"""
import argparse
import torch
import yaml
import numpy as np
from pathlib import Path

from src.lightning_model import LightningModel
from src.models.autoencoder.base import fp2uint8


def load_model(ckpt_path, config_path):
    """Load PixNerd model from checkpoint"""
    print(f"Loading checkpoint: {ckpt_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model = LightningModel.load_from_checkpoint(
        ckpt_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        strict=False
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, config


def generate_at_resolution(model, resolution, num_images=100, num_classes=10,
                          guidance=2.5, num_steps=50):
    """
    Generate images at specified resolution using the same model.

    Args:
        model: Loaded PixNerd model
        resolution: Target resolution (32, 64, 128, etc.)
        num_images: Number of images to generate
        num_classes: Number of classes (10 for CIFAR-10)
        guidance: CFG guidance scale
        num_steps: Number of sampling steps

    Returns:
        Generated images as numpy array [N, H, W, C]
    """
    print(f"\nGenerating {num_images} images at {resolution}×{resolution}...")

    # Update sampler parameters
    model.diffusion_sampler.guidance = guidance
    model.diffusion_sampler.num_steps = num_steps

    generated = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            current_batch_size = min(batch_size, num_images - i)

            # Generate random noise at target resolution
            noise = torch.randn(current_batch_size, 3, resolution, resolution)

            # Class labels (cycle through classes)
            class_labels = torch.tensor([j % num_classes for j in range(i, i + current_batch_size)])

            # Metadata
            metadata = {
                'resolution': resolution,
                'class': class_labels.tolist()
            }

            if torch.cuda.is_available():
                noise = noise.cuda()
                class_labels = class_labels.cuda()

            # Get conditioning
            condition, uncondition = model.conditioner(class_labels.tolist(), metadata)

            # Sample images
            samples = model.diffusion_sampler(
                model.ema_denoiser,
                noise,
                condition,
                uncondition
            )

            # Decode (identity for pixel-space)
            samples = model.vae.decode(samples)

            # Convert to uint8 [0, 255]
            samples = fp2uint8(samples)

            # Move to CPU and append
            generated.append(samples.cpu().numpy())

            print(f"  Generated {min(i + batch_size, num_images)}/{num_images}...")

    # Concatenate all batches
    generated = np.concatenate(generated, axis=0)

    # Convert from [N, C, H, W] to [N, H, W, C]
    generated = generated.transpose(0, 2, 3, 1)

    print(f"✓ Generated {len(generated)} images with shape {generated.shape}")

    return generated


def save_images(images, output_path, resolution):
    """Save generated images to .npz file"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_file = output_path / f"cifar10_{resolution}x{resolution}.npz"

    np.savez(npz_file, arr_0=images)

    print(f"\n✓ Saved to: {npz_file}")
    print(f"  Shape: {images.shape}")
    print(f"  Size: {npz_file.stat().st_size / (1024*1024):.2f} MB")

    return npz_file


def main():
    parser = argparse.ArgumentParser(description='Multi-resolution generation with PixNerd')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       default='configs_c2i/cifar_multires_128.yaml',
                       help='Path to config file')
    parser.add_argument('--resolution', type=int, default=128,
                       choices=[32, 64, 96, 128, 160, 192, 224, 256],
                       help='Target resolution (default: 128)')
    parser.add_argument('--num-images', type=int, default=10000,
                       help='Number of images to generate (default: 10000)')
    parser.add_argument('--guidance', type=float, default=2.5,
                       help='CFG guidance scale (default: 2.5)')
    parser.add_argument('--num-steps', type=int, default=50,
                       help='Number of sampling steps (default: 50)')
    parser.add_argument('--output-dir', type=str,
                       default='./generated_multires',
                       help='Output directory for generated images')
    parser.add_argument('--multi', action='store_true',
                       help='Generate at multiple resolutions (32, 64, 128) with same checkpoint')

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.ckpt, args.config)

    print("\n" + "="*60)
    print("PixNerd Multi-Resolution Generation")
    print("="*60)
    print(f"Model: {args.ckpt}")
    print(f"Target resolution(s): {args.resolution if not args.multi else '32, 64, 128'}")
    print(f"Number of images: {args.num_images}")
    print(f"Guidance scale: {args.guidance}")
    print(f"Sampling steps: {args.num_steps}")
    print("="*60)

    if args.multi:
        # Generate at multiple resolutions with SAME checkpoint
        resolutions = [32, 64, 128]
        print("\n🎯 Demonstrating resolution flexibility:")
        print("   Using SAME checkpoint to generate at 3 different resolutions!\n")

        for res in resolutions:
            images = generate_at_resolution(
                model, res,
                num_images=min(1000, args.num_images),  # Generate 1K each for demo
                guidance=args.guidance,
                num_steps=args.num_steps
            )
            save_images(images, args.output_dir, res)

        print("\n" + "="*60)
        print("✅ Multi-resolution generation complete!")
        print("   Same checkpoint successfully generated at:")
        for res in resolutions:
            print(f"   - {res}×{res}")
        print("\nThis demonstrates PixNerd's key feature:")
        print("Neural field positional encoding enables resolution flexibility!")
        print("="*60)

    else:
        # Generate at single resolution
        images = generate_at_resolution(
            model, args.resolution,
            num_images=args.num_images,
            guidance=args.guidance,
            num_steps=args.num_steps
        )
        npz_file = save_images(images, args.output_dir, args.resolution)

        print("\n" + "="*60)
        print(f"✅ Generation complete at {args.resolution}×{args.resolution}!")
        print(f"\nTo visualize:")
        print(f"  python view_generated_images.py \\")
        print(f"    --workdir {args.output_dir} \\")
        print(f"    --step {args.resolution}")
        print("="*60)


if __name__ == '__main__':
    main()
