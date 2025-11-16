# Cluster Setup Instructions

You're trying to run on a cluster at `/pscratch/sd/d/dpark1/`. The repository needs to be cloned there first.

## Quick Setup on Cluster

### Option 1: Clone from GitHub (Recommended)

```bash
# SSH to your cluster
ssh your-cluster

# Navigate to your scratch directory
cd /pscratch/sd/d/dpark1/Claude/

# Remove old directory if exists
rm -rf PixNerd

# Clone your fork with CIFAR-10 support
git clone https://github.com/colpark/pixnerdme.git PixNerd

# Enter directory
cd PixNerd

# Verify files are present
ls configs_c2i/cifar_repa_v1.yaml
ls src/data/dataset/cifar10.py
```

### Option 2: Copy from Local Machine

If you can't access GitHub from the cluster:

```bash
# From your LOCAL machine (Mac)
cd /Users/davidpark/Documents/Claude/

# Use rsync to copy to cluster
rsync -avz --progress PixNerd/ your-cluster:/pscratch/sd/d/dpark1/Claude/PixNerd/

# Or use scp
scp -r PixNerd/ your-cluster:/pscratch/sd/d/dpark1/Claude/PixNerd/
```

## Install Dependencies on Cluster

```bash
# SSH to cluster
ssh your-cluster
cd /pscratch/sd/d/dpark1/Claude/PixNerd

# Load necessary modules (adjust for your cluster)
module load python/3.10
module load cuda/11.8  # or appropriate CUDA version

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (adjust for your cluster's CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

## Verify Setup

```bash
# Test that config file exists
ls -la configs_c2i/cifar_repa_v1.yaml

# Test Python imports
python -c "from src.data.dataset.cifar10 import PixCIFAR10; print('✓ CIFAR-10 dataset importable')"

# Run validation script
python test_cifar10.py
```

## Training on Cluster

### Interactive Training (for testing)

```bash
cd /pscratch/sd/d/dpark1/Claude/PixNerd
source venv/bin/activate

# Single GPU
python main.py fit -c configs_c2i/cifar_repa_v1.yaml

# Multi-GPU (specify devices)
python main.py fit -c configs_c2i/cifar_repa_v1.yaml --trainer.devices=4
```

### Batch Job Training (recommended)

Create a SLURM job script `train_cifar10.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pixnerd_cifar10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/cifar10_%j.out
#SBATCH --error=logs/cifar10_%j.err

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
cd /pscratch/sd/d/dpark1/Claude/PixNerd
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Run training
python main.py fit -c configs_c2i/cifar_repa_v1.yaml \
    --trainer.devices=4 \
    --trainer.default_root_dir=/pscratch/sd/d/dpark1/Claude/PixNerd/workdirs

echo "Training completed at $(date)"
```

Submit the job:

```bash
sbatch train_cifar10.sh
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch training logs
tail -f logs/cifar10_*.out

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Common Issues

### Issue: Config file not found

**Solution**: Make sure you're in the correct directory:
```bash
cd /pscratch/sd/d/dpark1/Claude/PixNerd
ls configs_c2i/cifar_repa_v1.yaml  # Should exist
```

### Issue: Module not found (torch, lightning, etc.)

**Solution**: Make sure virtual environment is activated:
```bash
source venv/bin/activate
pip list | grep torch  # Should show PyTorch
```

### Issue: Out of memory

**Solution**: Reduce batch size in config:
```yaml
data:
  train_batch_size: 64  # Reduce from 128
```

### Issue: CUDA version mismatch

**Solution**: Install PyTorch matching your cluster's CUDA:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Cluster-Specific Configurations

You may want to modify the config for cluster environment:

```bash
# Edit the config to use cluster paths
nano configs_c2i/cifar_repa_v1.yaml
```

Update these paths:
```yaml
trainer:
  default_root_dir: /pscratch/sd/d/dpark1/Claude/PixNerd/workdirs  # Cluster workdir

data:
  train_dataset:
    init_args:
      root: /pscratch/sd/d/dpark1/Claude/PixNerd/data/cifar10  # Cluster data dir
```

## Quick Diagnostic Commands

```bash
# Verify you're in the right place
pwd  # Should show: /pscratch/sd/d/dpark1/Claude/PixNerd

# Check all CIFAR-10 files exist
ls -la configs_c2i/cifar_repa_v1.yaml
ls -la src/data/dataset/cifar10.py
ls -la test_cifar10.py
ls -la CIFAR10_README.md

# Test Python environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test config file parsing
python -c "import yaml; yaml.safe_load(open('configs_c2i/cifar_repa_v1.yaml')); print('✓ Config valid')"
```
