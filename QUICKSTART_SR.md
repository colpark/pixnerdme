# CIFAR-10 Super-Resolution Quick Start

Super-resolve CIFAR-10 from 32×32 to 128×128 using pretrained checkpoint.

## Setup (One-Time)

```bash
# On cluster
cd /pscratch/sd/d/dpark1/Claude/pixnerdme
git pull

# Validate setup
python test_cifar10_sr.py
```

## Training SR Model (New)

Train the super-resolution model:

```bash
# Single GPU
python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml

# Multi-GPU (4 GPUs)
python main_sr.py fit -c configs_c2i/cifar_sr_128.yaml --trainer.devices=4
```

**Training time**: ~48-72 hours on A100
**Output**: `./workdirs/exp_cifar10_sr_128x128/`

## Evaluation (Two Options)

### Option 1: SR on CIFAR-10 Test Set

Super-resolve real CIFAR-10 test images:

```bash
python main_sr.py predict -c configs_c2i/cifar_sr_128.yaml \
    --ckpt_path ./workdirs/exp_cifar10_sr_128x128/checkpoints/last.ckpt
```

**Output**: 10,000 SR images at `./workdirs/exp_cifar10_sr_128x128/val/predict/output.npz`

### Option 2: Full Pipeline (Generation + SR)

**Step 1**: Generate 32×32 with pretrained model
```bash
python main.py predict -c configs_c2i/cifar_basic_v1.yaml \
    --ckpt_path ./workdirs/exp_cifar10_basic_pixnerd_base/checkpoints/last.ckpt
```

**Step 2**: Super-resolve generated 32×32 → 128×128
```bash
# Modify config to use generated images as input
# Then run SR prediction
python main_sr.py predict -c configs_c2i/cifar_sr_128.yaml \
    --ckpt_path ./workdirs/exp_cifar10_sr_128x128/checkpoints/last.ckpt
```

## Visualization

View SR results:

```bash
# View SR outputs (128×128)
python view_generated_images.py \
    --workdir ./workdirs/exp_cifar10_sr_128x128/val \
    --step 25000 \
    --save sr_results.png
```

## Key Differences from 32×32 Training

| Aspect | 32×32 Model | 128×128 SR Model |
|--------|-------------|------------------|
| **Training script** | `main.py` | `main_sr.py` |
| **Config** | `cifar_basic_v1.yaml` | `cifar_sr_128.yaml` |
| **Input channels** | 3 | 6 (noisy HR + LR conditioning) |
| **Resolution** | 32×32 | 128×128 |
| **Batch size** | 128 | 64 |
| **Training steps** | 200K | 100K |

## Architecture

**Conditioning approach**:
- Input: `concat([noisy_hr, lr_upsampled], dim=1)` → [B, 6, 128, 128]
- Denoiser processes 6-channel input
- Output: Clean HR image [B, 3, 128, 128]

**Model size**: ~150M parameters (vs ~80M for 32×32 model)

## Files Created

```
PixNerd/
├── configs_c2i/cifar_sr_128.yaml              # SR config
├── src/
│   ├── data/dataset/cifar10_sr.py             # SR datasets
│   ├── models/conditioner/image_cond.py       # Image conditioners
│   ├── diffusion/flow_matching/
│   │   ├── training_sr.py                     # SR trainer
│   │   └── sampling_sr.py                     # SR sampler
│   └── lightning_model_sr.py                  # SR Lightning model
├── main_sr.py                                 # SR entry point
├── test_cifar10_sr.py                         # Validation script
├── CIFAR10_SR_README.md                       # Detailed docs
└── QUICKSTART_SR.md                           # This file
```

## Troubleshooting

**Out of Memory**: Reduce batch size to 32 in config
**Slow Training**: Use multi-GPU or smaller model
**Poor Quality**: Train longer or increase guidance scale

See [CIFAR10_SR_README.md](CIFAR10_SR_README.md) for full documentation.
