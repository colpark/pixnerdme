# Multi-Resolution Quick Start (PixNerd Paper Approach)

Train ONE model, generate at MULTIPLE resolutions (32×32, 64×64, 128×128+).

## What Makes This Special?

**PixNerd's Key Innovation**: Neural field positional encoding with normalized [0,1] coordinates enables resolution flexibility.

**Result**: Train at 128×128, use the SAME checkpoint to generate at any resolution!

## Quick Setup

```bash
# On cluster
cd /pscratch/sd/d/dpark1/Claude/pixnerdme
git pull

# Validate setup
python test_multires.py
```

## Training

Train a single model at 128×128:

```bash
# Single GPU
python main.py fit -c configs_c2i/cifar_multires_128.yaml

# Multi-GPU (4 GPUs)
python main.py fit -c configs_c2i/cifar_multires_128.yaml --trainer.devices=4
```

**Training Time**: ~48-72 hours on A100
**Output**: `./workdirs/exp_cifar10_multires_128/checkpoints/last.ckpt`

## Multi-Resolution Generation

### Demo: Generate at 3 Resolutions with SAME Checkpoint

```bash
# Generate 1K images each at 32×32, 64×64, and 128×128
# ALL using the SAME checkpoint!
python generate_multires.py \
    --ckpt ./workdirs/exp_cifar10_multires_128/checkpoints/last.ckpt \
    --multi \
    --num-images 1000
```

**Output**:
```
./generated_multires/
├── cifar10_32x32.npz
├── cifar10_64x64.npz
└── cifar10_128x128.npz
```

### Generate at Specific Resolution

```bash
# 10K images at 32×32
python generate_multires.py \
    --ckpt <checkpoint> \
    --resolution 32 \
    --num-images 10000

# 10K images at 64×64 (SAME checkpoint)
python generate_multires.py \
    --ckpt <checkpoint> \
    --resolution 64 \
    --num-images 10000

# 10K images at 128×128 (SAME checkpoint)
python generate_multires.py \
    --ckpt <checkpoint> \
    --resolution 128 \
    --num-images 10000
```

## Visualization

```bash
# View 32×32 generations
python view_generated_images.py \
    ./generated_multires/cifar10_32x32.npz

# View 128×128 generations
python view_generated_images.py \
    ./generated_multires/cifar10_128x128.npz \
    --num-images 64 \
    --grid-size 8
```

## Key Differences from Original 32×32 Model

| Aspect | Original 32×32 | Multi-Res 128×128 |
|--------|----------------|-------------------|
| **Resolutions** | 32×32 only | 32, 64, 128+ |
| **Patch size** | 4 | 8 |
| **Hidden size** | 384 | 768 |
| **Num blocks** | 12 | 20 |
| **Flexibility** | Fixed | Resolution-flexible |

## Comparison: PixNerd vs Cascaded SR

| Aspect | PixNerd (This) | Cascaded SR |
|--------|----------------|-------------|
| **Paper approach** | ✅ Yes | ❌ No |
| **Models needed** | 1 | 2 |
| **Training stages** | 1 | 2 |
| **Inference** | 1 pass | 2 passes |
| **Conditioning** | Class only | Class + image |
| **Efficiency** | High | Medium |

## How It Works

**Neural Field Magic**:
```python
# Normalized coordinates [0, 1] - resolution-independent!
pos_x = torch.linspace(0, 1, num_patches)
pos_y = torch.linspace(0, 1, num_patches)

# Works for ANY number of patches (any resolution)
dct = cos(pos_x * freqs * π) * cos(pos_y * freqs * π)
```

**Result**: Train once, generate at any resolution!

## Files

```
PixNerd/
├── configs_c2i/cifar_multires_128.yaml    # Training config
├── src/data/dataset/cifar10_multires.py   # Multi-res datasets
├── generate_multires.py                   # Generation script
├── test_multires.py                       # Validation
├── MULTIRES_README.md                     # Full docs
└── QUICKSTART_MULTIRES.md                 # This file
```

## Next Steps

1. ✅ Pull code on cluster
2. ✅ Run validation: `python test_multires.py`
3. 🎯 Train: `python main.py fit -c configs_c2i/cifar_multires_128.yaml`
4. 🎨 Generate: `python generate_multires.py --ckpt <checkpoint> --multi`

See [MULTIRES_README.md](MULTIRES_README.md) for complete documentation.
