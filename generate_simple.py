#!/usr/bin/env python3
"""
Simple multi-resolution generation script using Lightning CLI pattern.

Usage:
    # Generate at 128×128 using existing 32×32 checkpoint
    python generate_simple.py \
        --ckpt ./workdirs/exp_cifar10_basic_pixnerd_base/last.ckpt \
        --resolution 128 \
        --num-images 10000
"""
import argparse
import sys
import yaml
import subprocess
from pathlib import Path


def create_temp_config(base_config_path, resolution, num_images, output_path):
    """Create temporary config with modified resolution and dataset size"""

    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify pred_dataset for target resolution
    if 'data' not in config:
        config['data'] = {}

    # Use RandomNDataset for generation at target resolution
    config['data']['pred_dataset'] = {
        'class_path': 'src.data.dataset.cifar10.CIFAR10RandomNDataset',
        'init_args': {
            'num_classes': 10,
            'max_num_instances': num_images,
            'latent_shape': [3, resolution, resolution]
        }
    }

    # Set pred_batch_size
    config['data']['pred_batch_size'] = 64

    # Set prediction output path
    if 'trainer' not in config:
        config['trainer'] = {}
    config['trainer']['default_root_dir'] = str(output_path)

    # Save temporary config
    temp_config = Path(f'temp_gen_{resolution}.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_config


def main():
    parser = argparse.ArgumentParser(description='Simple multi-resolution generation')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Base config (auto-detects if not specified)')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Target resolution')
    parser.add_argument('--num-images', type=int, default=10000,
                       help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='./generated_multires',
                       help='Output directory')

    args = parser.parse_args()

    # Auto-detect config
    if args.config is None:
        if 'exp_cifar10_basic' in args.ckpt:
            args.config = 'configs_c2i/cifar_basic_v1.yaml'
            print(f"Auto-detected config: {args.config}")
        elif 'exp_cifar10_multires' in args.ckpt:
            args.config = 'configs_c2i/cifar_multires_128.yaml'
            print(f"Auto-detected config: {args.config}")
        else:
            print("ERROR: Could not auto-detect config. Please specify --config")
            sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir) / f"{args.resolution}x{args.resolution}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create temporary config
    print(f"\nCreating temporary config for {args.resolution}×{args.resolution} generation...")
    temp_config = create_temp_config(args.config, args.resolution, args.num_images, output_path)

    print(f"Temporary config: {temp_config}")
    print(f"Output will be saved to: {output_path}")

    # Run prediction using Lightning CLI
    print(f"\nGenerating {args.num_images} images at {args.resolution}×{args.resolution}...")
    cmd = [
        'python', 'main.py', 'predict',
        '-c', str(temp_config),
        '--ckpt_path', args.ckpt
    ]

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=True)

    # Clean up temp config
    temp_config.unlink()

    print(f"\n✅ Generation complete!")
    print(f"Output saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
