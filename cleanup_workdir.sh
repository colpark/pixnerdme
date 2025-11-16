#!/bin/bash
# Cleanup script for PixNerd CIFAR-10 training
# Removes old experiment directories to allow fresh training

set -e  # Exit on error

WORKDIR="./workdirs/exp_cifar10_basic_pixnerd_base"

echo "PixNerd CIFAR-10 Cleanup Script"
echo "================================"
echo ""

# Check if workdir exists
if [ -d "$WORKDIR" ]; then
    echo "Found existing experiment directory: $WORKDIR"

    # Check if it has checkpoints
    if [ -d "$WORKDIR/checkpoints" ]; then
        echo ""
        echo "Checkpoints found:"
        ls -lh "$WORKDIR/checkpoints/" 2>/dev/null || echo "  (empty)"
        echo ""
        echo "WARNING: This will delete all checkpoints and training data!"
        echo ""
        read -p "Do you want to continue? (yes/no): " confirm

        if [ "$confirm" != "yes" ]; then
            echo "Cleanup cancelled."
            exit 0
        fi
    fi

    echo ""
    echo "Removing $WORKDIR..."
    rm -rf "$WORKDIR"
    echo "✓ Cleanup complete!"
else
    echo "No existing experiment directory found at: $WORKDIR"
    echo "✓ Ready for fresh training!"
fi

echo ""
echo "You can now start training with:"
echo "  python main.py fit -c configs_c2i/cifar_basic_v1.yaml"
