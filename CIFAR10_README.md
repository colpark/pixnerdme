# PixNerd for CIFAR-10

This repository extends PixNerd to support CIFAR-10 dataset training and evaluation.

## Overview

CIFAR-10 is a well-established computer vision dataset consisting of 60,000 32×32 color images across 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

This adaptation maintains PixNerd's pixel-space diffusion approach with Neural Field representations, optimized for the smaller 32×32 resolution.

## Key Differences from ImageNet Setup

| Parameter | ImageNet | CIFAR-10 |
|-----------|----------|----------|
| Resolution | 256×256 | 32×32 |
| Classes | 1,000 | 10 |
| Patch Size | 16 | 4 |
| Hidden Size | 1,152 | 384 |
| Num Blocks | 30 | 12 |
| Training Steps | 800,000 | 200,000 |
| Batch Size | 32 | 128 |
| Sampling Steps | 100 | 50 |
| CFG Guidance | 3.5 | 2.0 |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Dataset

The CIFAR-10 dataset will be automatically downloaded on first run to `./data/cifar10/`.

## Training

```bash
# Single GPU training
python main.py fit -c configs_c2i/cifar_basic_v1.yaml

# Multi-GPU training (DDP)
python main.py fit -c configs_c2i/cifar_basic_v1.yaml --trainer.devices=4
```

### Training Configuration

The configuration file `configs_c2i/cifar_basic_v1.yaml` includes:

- **Model Architecture**: PixNerDiT with 4×4 patches (8×8 patch grid)
- **Diffusion Trainer**: FlowMatchingTrainer (basic flow matching without auxiliary encoder)
- **Training**: 200,000 steps with validation every 50 epochs (~19,500 steps)
- **Optimizer**: AdamW with lr=1e-4
- **Augmentation**: Random crop with padding + random horizontal flip
- **Checkpoints**: Saved every 5,000 steps to `./workdirs/`

**Note**: With 50K training images and batch size 128, each epoch is ~390 steps. Validation runs every 50 epochs for efficiency.

**Note**: This uses basic flow matching. For REPA training with DINOv2 auxiliary encoder (better quality but requires additional setup), see the Advanced Training section below.

## Inference

### Generate Images from Random Noise

```bash
python main.py predict -c configs_c2i/cifar_basic_v1.yaml \
    --ckpt_path ./workdirs/cifar10_basic_pixnerd_base/checkpoints/last.ckpt
```

This generates 10,000 images (1,000 per class) for evaluation.

### Generate Specific Class

You can modify the prediction dataset in the config to focus on specific classes:

```yaml
pred_dataset:
  class_path: src.data.dataset.cifar10.CIFAR10RandomNDataset
  init_args:
    num_classes: 10
    max_num_instances: 1000  # Generate 1000 images total
```

## Model Architecture

### PixNerDiT for CIFAR-10

```
Input: [B, 3, 32, 32]
  ↓
Patchify (4×4 patches): [B, 64, 48]  # 8×8 = 64 patches, 3×4×4 = 48 dims
  ↓
NerfEmbedder: Neural field positional encoding
  ↓
12 Transformer Blocks:
  - 10 FlattenDiTBlock (conditional processing)
  - 2 NerfBlock (neural field refinement)
  ↓
NerfFinalLayer: [B, 64, 3] → unpatch → [B, 3, 32, 32]
```

### Key Components

1. **PixelAE**: Identity mapping (no VAE compression)
2. **NerfEmbedder**: Frequency-based positional encoding for high-frequency modeling
3. **FlowMatchingTrainer**: Basic flow matching diffusion training
4. **EulerSampler**: ODE-based sampling with Classifier-Free Guidance

## Monitoring

Training progress is logged to Weights & Biases:

- **Project**: `pixnerd_cifar10`
- **Run Name**: `cifar10_basic_pixnerd_base`

Validation images are saved to `./workdirs/cifar10_basic_pixnerd_base/val/`

## Visualizing Generated Images

Every 50 epochs (~19,500 steps), the model generates validation images saved as `.npz` files:

```
./workdirs/exp_cifar10_basic_pixnerd_base/val/
├── iter_19500/
│   └── output.npz  # First validation (50 epochs)
├── iter_39000/
│   └── output.npz  # Second validation (100 epochs)
├── iter_58500/
│   └── output.npz  # Third validation (150 epochs)
└── ...
```

### Quick Visualization

```bash
# List all available checkpoints
python view_generated_images.py --list

# View latest generated images (10x10 grid)
python view_generated_images.py

# View specific checkpoint
python view_generated_images.py --step 19500

# Save visualization to file instead of displaying
python view_generated_images.py --step 39000 --save progress_step39000.png

# Show more images (15x15 grid = 225 images)
python view_generated_images.py --num-images 225 --grid-size 15
```

### On Cluster

Since the cluster may not have display capabilities, save visualizations to PNG:

```bash
# On cluster - save to file
python view_generated_images.py --step 19500 --save validation_19500.png

# Then download to local machine
scp your-cluster:/pscratch/sd/d/dpark1/Claude/PixNerd/validation_19500.png .
```

### Understanding the Output

Each `.npz` file contains:
- **10,000 generated images** (1,000 per class)
- **Image format**: 32×32 RGB, uint8 [0, 255]
- **File size**: ~30-40 MB per checkpoint

The visualization script shows how well the model is learning to generate CIFAR-10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks) at different training stages.

## Expected Performance

Training on CIFAR-10 is significantly faster than ImageNet:

- **Training Time**: ~24-48 hours on 1×A100 GPU
- **Memory**: ~8-12GB GPU memory per device
- **Convergence**: Observable quality improvements by 50,000 steps

Target metrics (after full training):
- **FID**: < 10.0 (competitive with other pixel-space diffusion models)
- **Inception Score**: > 8.0

## Advanced Training with REPA (Optional)

For improved quality using REPA training with DINOv2 auxiliary encoder, you need to first download the DINOv2 model locally:

### Step 1: Download DINOv2

```bash
# Create torch hub directory
mkdir -p torch_hub
cd torch_hub

# Clone DINOv2 repository
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2

# The model will be in: torch_hub/dinov2/
```

### Step 2: Create REPA Config

Create `configs_c2i/cifar_repa_v1.yaml` based on `cifar_basic_v1.yaml`, but replace the diffusion_trainer section:

```yaml
diffusion_trainer:
  class_path: src.diffusion.flow_matching.training_repa.REPATrainer
  init_args:
    lognorm_t: true
    timeshift: 1.0
    feat_loss_weight: 0.5
    encoder:
      class_path: src.models.encoder.DINOv2
      init_args:
        weight_path: /path/to/torch_hub/dinov2/dinov2_vitb14  # Update this path
    align_layer: 8
    proj_denoiser_dim: 384
    proj_hidden_dim: 384
    proj_encoder_dim: 768
    scheduler: &scheduler src.diffusion.flow_matching.scheduling.LinearScheduler
```

### Step 3: Train with REPA

```bash
python main.py fit -c configs_c2i/cifar_repa_v1.yaml
```

REPA training uses auxiliary DINOv2 encoder features to improve high-frequency detail generation, potentially improving FID by 10-20%.

## Customization

### Adjust Model Size

Edit `configs_c2i/cifar_basic_v1.yaml`:

```yaml
denoiser:
  init_args:
    hidden_size: 512  # Increase for larger model
    num_blocks: 18    # More transformer blocks
```

### Modify Training Schedule

```yaml
trainer:
  max_steps: 300000            # Train longer
  check_val_every_n_epoch: 25  # Validate more frequently (every 25 epochs ≈ 9,750 steps)
```

### Change Sampling Parameters

```yaml
diffusion_sampler:
  init_args:
    num_steps: 100   # More steps = higher quality, slower
    guidance: 3.0    # Higher CFG = stronger class conditioning
```

## File Structure

```
PixNerd/
├── configs_c2i/
│   └── cifar_basic_v1.yaml          # CIFAR-10 training config (basic)
├── src/
│   └── data/
│       └── dataset/
│           └── cifar10.py           # CIFAR-10 dataset implementation
├── test_cifar10.py                  # Validation script
├── view_generated_images.py         # Visualization tool for .npz files
├── cleanup_workdir.sh               # Cleanup script for experiment dirs
├── data/
│   └── cifar10/                     # Auto-downloaded CIFAR-10 data
└── workdirs/
    └── exp_cifar10_basic_pixnerd_base/  # Checkpoints and logs
        ├── checkpoints/             # Model checkpoints (every 5K steps)
        └── val/                     # Validation images (.npz files)
            ├── iter_19500/
            │   └── output.npz
            ├── iter_39000/
            │   └── output.npz
            └── ...
```

## Troubleshooting

### Experiment Directory Already Exists

**Error**: `FileExistsError: ./workdirs/exp_cifar10_basic_pixnerd_base already exists`

**Cause**: Safety check prevents overwriting previous experiments.

**Solutions**:

**Option 1 - Clean Start (Recommended)**:
```bash
# Use cleanup script
bash cleanup_workdir.sh

# Or manually remove
rm -rf ./workdirs/exp_cifar10_basic_pixnerd_base

# Then train
python main.py fit -c configs_c2i/cifar_basic_v1.yaml
```

**Option 2 - Resume Training**:
```bash
# Continue from last checkpoint
python main.py fit -c configs_c2i/cifar_basic_v1.yaml \
    --ckpt_path ./workdirs/exp_cifar10_basic_pixnerd_base/checkpoints/last.ckpt
```

**Option 3 - Change Experiment Name**:
Edit `configs_c2i/cifar_basic_v1.yaml` and change line 5:
```yaml
exp: &exp cifar10_basic_pixnerd_v2  # Add version number
```

### Out of Memory

Reduce batch size in config:
```yaml
data:
  train_batch_size: 64  # Reduce from 128
```

### Slow Training

Enable mixed precision (already enabled by default):
```yaml
trainer:
  precision: bf16-mixed
```

### Configuration Errors

If you get errors about missing encoder or DINOv2, make sure you're using `cifar_basic_v1.yaml` (not `cifar_repa_v1.yaml` which requires additional DINOv2 setup).

### Validation Interval Errors

If you get `val_check_interval should be an integer` errors, this is due to Lightning version compatibility. The config uses `check_val_every_n_epoch` (epoch-based validation) which is more stable across Lightning versions than step-based `val_check_interval`.

### Torch Compile Errors

**Error**: `TypeError: Invalid NaN comparison` or `BackendCompilerFailed: backend='inductor'`

**Cause**: PyTorch 2.5 `torch.compile()` has compatibility issues with the model architecture.

**Fix**: `torch.compile()` has been disabled in `src/lightning_model.py` (lines 65-66 commented out). Training will run without compilation (slightly slower but stable).

**Note**: This doesn't significantly impact training speed for CIFAR-10's small 32×32 images.

## Citation

If you use this CIFAR-10 extension, please cite the original PixNerd paper:

```bibtex
@article{pixnerd2024,
  title={PixNerd: Pixel-Space Diffusion Transformers with Neural Field Representations},
  author={...},
  journal={arXiv preprint arXiv:2507.23268},
  year={2024}
}
```

## License

Same as original PixNerd repository.
