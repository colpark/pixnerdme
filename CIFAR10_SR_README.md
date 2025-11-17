# CIFAR-10 Super-Resolution: 32×32 → 128×128

This extension enables super-resolution from CIFAR-10's native 32×32 resolution to 128×128 using cascaded diffusion.

## Overview

**Pipeline**:
1. **Stage 1** (Already trained): Generate 32×32 CIFAR-10 images with pretrained model
2. **Stage 2** (New): Super-resolve 32×32 → 128×128 with SR diffusion model

**Approach**: Image-conditioned diffusion
- Input: Bicubic-upsampled 32×32 image (at 128×128 resolution)
- Output: High-quality 128×128 super-resolved image
- Conditioning: Concatenate upsampled LR with noisy HR (6 channels total)

## Architecture

**Super-Resolution Model**:
- **Input channels**: 6 (3 for noisy HR + 3 for upsampled LR conditioning)
- **Resolution**: 128×128
- **Patch size**: 8 (creates 16×16 patch grid)
- **Hidden size**: 512 (larger than 32×32 model's 384)
- **Blocks**: 16 (more capacity for higher resolution)
- **Classes**: 10 (CIFAR-10 class conditioning)

**Key Differences from 32×32 Model**:
| Parameter | 32×32 Model | 128×128 SR Model |
|-----------|-------------|------------------|
| Resolution | 32×32 | 128×128 |
| Input channels | 3 | 6 (concat conditioning) |
| Patch size | 4 | 8 |
| Patches | 8×8 = 64 | 16×16 = 256 |
| Hidden size | 384 | 512 |
| Blocks | 12 | 16 |
| Batch size | 128 | 64 |
| Training steps | 200K | 100K |

## Dataset

**CIFAR10SuperResDataset**:
- **Training data**: CIFAR-10 train set (50K images)
- **LR images**: Original 32×32 CIFAR-10 images
- **HR targets**: Bicubic upsampled to 128×128
- **Conditioning**: LR upsampled to 128×128 (bicubic)
- **No augmentation**: Uses clean, centered images

**CIFAR10SuperResEvalDataset**:
- **Evaluation data**: CIFAR-10 test set (10K images)
- **Same structure as training dataset**
- **No augmentation for fair evaluation**

## Training

### Prerequisites

1. **Install dependencies** (if not already done):
```bash
pip install -r requirements.txt
```

2. **CIFAR-10 data** will auto-download on first run to `./data/cifar10/`

### Training Command

```bash
# Single GPU
python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml

# Multi-GPU (4 GPUs with DDP)
python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml --trainer.devices=4
```

### Training Configuration

**Config**: `configs_c2i/cifar_sr_128.yaml`

Key settings:
- **Max steps**: 100,000 (faster than 32×32 training)
- **Batch size**: 64 (smaller due to 128×128 resolution)
- **Validation**: Every 25 epochs (~4,900 steps)
- **Checkpoints**: Saved every 2,500 steps
- **Optimizer**: AdamW, lr=1e-4
- **Precision**: bfloat16-mixed (memory efficient)
- **Logger**: Weights & Biases (`pixnerd_cifar10_sr` project)

**Output directory**:
```
./workdirs/exp_cifar10_sr_128x128/
├── checkpoints/         # Model checkpoints (every 2,500 steps)
├── val/                # Validation images (.npz files)
└── config-*.yaml       # Training configuration snapshots
```

### Expected Training Time

- **Single A100 GPU**: ~48-72 hours for 100K steps
- **Memory usage**: ~16-24GB per GPU
- **Convergence**: Observable quality by 25K-50K steps

## Inference / Evaluation

### Generate SR Images from Test Set

```bash
# Generate 128×128 super-resolved images from CIFAR-10 test set
python main_sr.py predict -c configs_c2i/cifar_sr_128.yaml \
    --ckpt_path ./workdirs/exp_cifar10_sr_128x128/checkpoints/last.ckpt
```

**Output**: 10,000 super-resolved 128×128 images saved to:
```
./workdirs/exp_cifar10_sr_128x128/val/predict/output.npz
```

### Full Pipeline: 32×32 Generation → 128×128 SR

To use the complete two-stage pipeline:

**Step 1**: Generate 32×32 images with pretrained CIFAR-10 model
```bash
python main.py predict -c configs_c2i/cifar_basic_v1.yaml \
    --ckpt_path ./workdirs/exp_cifar10_basic_pixnerd_base/checkpoints/last.ckpt
```

**Step 2**: Super-resolve generated 32×32 → 128×128
```python
# Use CIFAR10GeneratedSuperResDataset in your evaluation script
from src.data.dataset.cifar10_sr import CIFAR10GeneratedSuperResDataset

# Load generated 32×32 images
dataset = CIFAR10GeneratedSuperResDataset(
    generated_lr_path='./workdirs/exp_cifar10_basic_pixnerd_base/val/predict/output.npz',
    hr_size=128
)

# Then run SR prediction
python main_sr.py predict -c configs_c2i/cifar_sr_128.yaml \
    --ckpt_path ./workdirs/exp_cifar10_sr_128x128/checkpoints/last.ckpt \
    --data.pred_dataset=... # Configure to use generated dataset
```

## Visualization

Visualize SR results using the same tool as 32×32 generation:

```bash
# View latest SR checkpoint (128×128 images)
python view_generated_images.py \
    --workdir ./workdirs/exp_cifar10_sr_128x128/val \
    --step 25000 \
    --save sr_results_25000.png

# Compare with bicubic upsampling
python view_generated_images.py \
    --workdir ./workdirs/exp_cifar10_sr_128x128/val \
    --step 50000 \
    --num-images 64 \
    --grid-size 8 \
    --save sr_final.png
```

## Model Architecture Details

### SuperResolutionFlowMatchingTrainer

Custom trainer that handles image conditioning:
- Concatenates upsampled LR image with noisy HR input
- Input to denoiser: `[noisy_hr, lr_upsampled]` → [B, 6, H, W]
- Maintains standard flow matching loss on HR target

### SuperResolutionEulerSampler

Custom sampler for SR generation:
- Concatenates conditioning image at each denoising step
- Supports classifier-free guidance for quality control
- ODE-based sampling (50 steps default)

### Image Conditioning

**ConcatImageConditioner**:
- Extracts `lr_upsampled` from metadata
- Returns conditioning image for channel-wise concatenation
- Supports unconditional generation (zeros) for CFG

## Evaluation Metrics

**Recommended metrics**:
1. **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level fidelity
2. **SSIM** (Structural Similarity): Perceptual quality
3. **LPIPS** (Learned Perceptual Image Patch Similarity): Deep learning-based quality
4. **FID** (Fréchet Inception Distance): Distribution matching

**Evaluation script** (example):
```python
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Load SR outputs
sr_images = np.load('./workdirs/exp_cifar10_sr_128x128/val/predict/output.npz')['arr_0']

# Load ground truth (bicubic upsampled CIFAR-10 test set)
# Calculate metrics...
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```yaml
# In configs_c2i/cifar_sr_128.yaml
data:
  train_batch_size: 32  # Reduce from 64
```

### Slow Training

Enable torch.compile (if compatible):
```python
# In src/lightning_model_sr.py
# Uncomment lines for torch.compile()
```

Or use smaller model:
```yaml
denoiser:
  init_args:
    hidden_size: 384  # Reduce from 512
    num_blocks: 12    # Reduce from 16
```

### Poor SR Quality

**Increase training steps**:
```yaml
trainer:
  max_steps: 150000  # Train longer
```

**Increase guidance scale**:
```yaml
diffusion_sampler:
  init_args:
    guidance: 3.0  # Increase from 2.0
```

**More sampling steps**:
```yaml
diffusion_sampler:
  init_args:
    num_steps: 100  # Increase from 50
```

## Advanced: Using Pretrained 32×32 Model as Initialization

To warm-start SR training with 32×32 checkpoint weights:

```bash
# This is experimental - requires manual weight adaptation
# The pretrained model has in_channels=3, SR model needs in_channels=6

# You can initialize the first 3 channels from pretrained weights
# See src/lightning_model_sr.py for implementation details
```

## Cluster Deployment

For training on SLURM clusters:

```bash
# Single GPU job
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

cd /pscratch/sd/d/dpark1/Claude/pixnerdme
source venv/bin/activate
python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml
```

**Multi-GPU job**:
```bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4  # MUST match GPU count
#SBATCH --time=48:00:00

cd /pscratch/sd/d/dpark1/Claude/pixnerdme
source venv/bin/activate
srun python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml
```

## File Structure

```
PixNerd/
├── configs_c2i/
│   └── cifar_sr_128.yaml                    # SR training config
├── src/
│   ├── data/dataset/
│   │   └── cifar10_sr.py                    # SR datasets
│   ├── models/conditioner/
│   │   └── image_cond.py                    # Image conditioners
│   ├── diffusion/flow_matching/
│   │   ├── training_sr.py                   # SR trainer
│   │   └── sampling_sr.py                   # SR sampler
│   ├── lightning_model_sr.py                # SR Lightning model
├── main_sr.py                               # SR training entry point
├── view_generated_images.py                 # Visualization tool
└── workdirs/
    └── exp_cifar10_sr_128x128/              # SR outputs
```

## Citation

If you use this super-resolution extension, please cite the original PixNerd paper:

```bibtex
@article{pixnerd2024,
  title={PixNerd: Pixel-Space Diffusion Transformers with Neural Field Representations},
  author={...},
  journal={arXiv preprint arXiv:2507.23268},
  year={2024}
}
```

## Acknowledgments

This SR extension builds upon the PixNerd architecture for image-conditioned diffusion super-resolution.
