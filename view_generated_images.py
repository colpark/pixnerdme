#!/usr/bin/env python3
"""
Visualize generated CIFAR-10 images from .npz files
Usage: python view_generated_images.py [--step STEP] [--num-images N]
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_npz(npz_path):
    """Load images from .npz file"""
    data = np.load(npz_path)
    images = data['arr_0']  # Shape: [N, H, W, C]
    return images

def center_crop(image, crop_size=28):
    """Center crop image to remove edge artifacts from random crop augmentation"""
    h, w = image.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return image[start_h:start_h+crop_size, start_w:start_w+crop_size]

def visualize_grid(images, num_images=100, grid_size=10, save_path=None, apply_center_crop=False):
    """Display images in a grid"""
    num_images = min(num_images, len(images))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    title = f'Generated CIFAR-10 Images (showing {num_images} samples)'
    if apply_center_crop:
        title += ' - Center Cropped'
    fig.suptitle(title, fontsize=16)

    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]

        if idx < num_images:
            img = images[idx]
            # Optionally apply center crop to remove translation artifacts
            if apply_center_crop:
                img = center_crop(img, crop_size=28)
            # Images are already uint8 [0, 255]
            ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

def find_latest_checkpoint(workdir="./workdirs/exp_cifar10_basic_pixnerd_base/val"):
    """Find the latest validation checkpoint"""
    val_dir = Path(workdir)
    if not val_dir.exists():
        return None

    # Find all iter_* directories
    iter_dirs = sorted(val_dir.glob("iter_*"), key=lambda p: int(p.name.split('_')[1]))

    if not iter_dirs:
        return None

    return iter_dirs[-1] / "output.npz"

def list_checkpoints(workdir="./workdirs/exp_cifar10_basic_pixnerd_base/val"):
    """List all available checkpoints"""
    val_dir = Path(workdir)
    if not val_dir.exists():
        print(f"Validation directory not found: {workdir}")
        return []

    iter_dirs = sorted(val_dir.glob("iter_*"), key=lambda p: int(p.name.split('_')[1]))

    print("\nAvailable validation checkpoints:")
    print("=" * 50)
    for iter_dir in iter_dirs:
        step = int(iter_dir.name.split('_')[1])
        npz_file = iter_dir / "output.npz"
        if npz_file.exists():
            size_mb = npz_file.stat().st_size / (1024 * 1024)
            print(f"  Step {step:>6}: {npz_file} ({size_mb:.2f} MB)")
    print("=" * 50)

    return iter_dirs

def main():
    parser = argparse.ArgumentParser(description='Visualize generated CIFAR-10 images')
    parser.add_argument('npz_file', type=str, nargs='?', default=None,
                       help='Direct path to .npz file (e.g., output.npz or generated_multires/128x128/val/predict/output.npz)')
    parser.add_argument('--step', type=int, default=None,
                       help='Training step to visualize (e.g., 19500). If not specified, uses latest.')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of images to display (default: 100)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size for display (default: 10x10)')
    parser.add_argument('--workdir', type=str,
                       default='./workdirs/exp_cifar10_basic_pixnerd_base/val',
                       help='Path to validation directory')
    parser.add_argument('--save', type=str, default=None,
                       help='Save visualization to file instead of displaying')
    parser.add_argument('--list', action='store_true',
                       help='List all available checkpoints')
    parser.add_argument('--center-crop', action='store_true',
                       help='Apply center crop (28x28) to remove translation artifacts from augmentation')

    args = parser.parse_args()

    # List checkpoints if requested
    if args.list:
        list_checkpoints(args.workdir)
        return

    # Determine which checkpoint to load
    if args.npz_file is not None:
        # Direct npz file path provided
        npz_path = Path(args.npz_file)
        if not npz_path.exists():
            print(f"Error: File not found: {npz_path}")
            return
        print(f"Loading from direct path: {npz_path}")
    elif args.step is not None:
        npz_path = Path(args.workdir) / f"iter_{args.step}" / "output.npz"
        if not npz_path.exists():
            print(f"Error: Checkpoint not found at step {args.step}")
            print(f"Looking for: {npz_path}")
            print("\nUse --list to see available checkpoints")
            return
    else:
        print("No step specified, finding latest checkpoint...")
        npz_path = find_latest_checkpoint(args.workdir)
        if npz_path is None:
            print(f"Error: No checkpoints found in {args.workdir}")
            print("Make sure training has completed at least one validation.")
            return
        step = int(npz_path.parent.name.split('_')[1])
        print(f"Found latest checkpoint at step {step}")

    # Load and visualize
    print(f"Loading images from: {npz_path}")
    images = load_npz(npz_path)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Image range: [{images.min()}, {images.max()}]")

    # Visualize
    visualize_grid(images, args.num_images, args.grid_size, args.save, args.center_crop)

if __name__ == '__main__':
    main()
