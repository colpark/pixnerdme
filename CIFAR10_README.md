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
python main.py fit -c configs_c2i/cifar_repa_v1.yaml

# Multi-GPU training (DDP)
python main.py fit -c configs_c2i/cifar_repa_v1.yaml --trainer.devices=4
```

### Training Configuration

The configuration file `configs_c2i/cifar_repa_v1.yaml` includes:

- **Model Architecture**: PixNerDiT with 4×4 patches (8×8 patch grid)
- **Auxiliary Encoder**: DINOv2-base for feature alignment
- **Training**: 200,000 steps with validation every 20,000 steps
- **Optimizer**: AdamW with lr=1e-4
- **Augmentation**: Random crop with padding + random horizontal flip
- **Checkpoints**: Saved every 5,000 steps to `./workdirs/`

## Inference

### Generate Images from Random Noise

```bash
python main.py predict -c configs_c2i/cifar_repa_v1.yaml \
    --ckpt_path ./workdirs/cifar10_repa_pixnerd_base/checkpoints/last.ckpt
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
3. **REPATrainer**: Flow matching with DINOv2 feature alignment
4. **EulerSampler**: ODE-based sampling with Classifier-Free Guidance

## Monitoring

Training progress is logged to Weights & Biases:

- **Project**: `pixnerd_cifar10`
- **Run Name**: `cifar10_repa_pixnerd_base`

Validation images are saved to `./workdirs/cifar10_repa_pixnerd_base/val/`

## Expected Performance

Training on CIFAR-10 is significantly faster than ImageNet:

- **Training Time**: ~24-48 hours on 1×A100 GPU
- **Memory**: ~8-12GB GPU memory per device
- **Convergence**: Observable quality improvements by 50,000 steps

Target metrics (after full training):
- **FID**: < 10.0 (competitive with other pixel-space diffusion models)
- **Inception Score**: > 8.0

## Customization

### Adjust Model Size

Edit `configs_c2i/cifar_repa_v1.yaml`:

```yaml
denoiser:
  init_args:
    hidden_size: 512  # Increase for larger model
    num_blocks: 18    # More transformer blocks
```

### Modify Training Schedule

```yaml
trainer:
  max_steps: 300000          # Train longer
  val_check_interval: 10000  # Validate more frequently
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
│   └── cifar_repa_v1.yaml           # CIFAR-10 training config
├── src/
│   └── data/
│       └── dataset/
│           └── cifar10.py           # CIFAR-10 dataset implementation
├── data/
│   └── cifar10/                     # Auto-downloaded CIFAR-10 data
└── workdirs/
    └── cifar10_repa_pixnerd_base/  # Checkpoints and logs
```

## Troubleshooting

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

### DINOv2 Download Issues

If automatic download fails, manually download:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-base")
```

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
