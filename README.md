# PixNerd: Pixel Neural Field Diffusion
<div style="text-align: center;">
  <a href="http://arxiv.org/abs/2507.23268"><img src="https://img.shields.io/badge/arXiv-2507.23268-b31b1b.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/spaces/MCG-NJU/PixNerd"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-green" alt="arXiv"></a>  
</div>

![](./figs/arch.png)

## Introduction
We propose PixNerd, a powerful and efficient **pixel-space** diffusion transformer for image generation (without VAE). Different from conventional pixel diffusion models, we employ the neural field to improve the high frequercy modeling .

* We achieve **1.93 FID** on ImageNet256x256 Benchmark with PixNerd-XL/16 (1600k training steps).
* We achieve **2.84 FID** on ImageNet512x512 Benchmark with PixNerd-XL/16.
* We achieve **0.73 overall score** on GenEval Benchmark with PixNerd-XXL/16.
* We achieve **80.9 avergae score** on DPG Benchmark with PixNerd-XXL/16.

## Visualizations
![](./figs/pixelnerd_teaser.png)
![](./figs/pixnerd_multires.png)
## Checkpoints

| Dataset       | Model         | Params | FID   | HuggingFace                           |
|---------------|---------------|--------|-------|---------------------------------------|
| ImageNet256   | PixNerd-XL/16 | 700M   | 1.93  | [🤗](https://huggingface.co/MCG-NJU/PixNerd-XL-P16-C2I) |
| ImageNet512   | PixNerd-XL/16 | 700M   | 2.84  | [🤗](https://huggingface.co/MCG-NJU/PixNerd-XL-P16-C2I) |

| Dataset       | Model         | Params | GenEval | DPG  | HuggingFace                                              |
|---------------|---------------|--------|------|------|----------------------------------------------------------|
| Text-to-Image | PixNerd-XXL/16| 1.2B | 0.73 | 80.9 | [🤗](https://huggingface.co/MCG-NJU/PixNerd-XXL-P16-T2I) |
## Online Demos
![](./figs/demo.png)
We provide online demos for PixNerd-XXL/16(text-to-image) on HuggingFace Spaces.

强烈建议本地部署玩玩，线上的模型推理速度会慢一些。以及因为这个我把任意分辨率和动画都关了。

HF spaces: [https://huggingface.co/spaces/MCG-NJU/PixNerd](https://huggingface.co/spaces/MCG-NJU/PixNerd)

To host the local gradio demo, run the following command:
```bash
# for text-to-image applications
python app.py --config configs_t2i/inference_heavydecoder.yaml  --ckpt_path=XXX.ckpt
```

## Usages
For C2i(ImageNet), We use ADM evaluation suite to report FID.
```bash
# for installation
pip install -r requirements.txt
```

```bash
# for inference
python main.py predict -c configs_c2i/pix256std1_repa_pixnerd_xl.yaml --ckpt_path=XXX.ckpt
# # or specify the GPU(s) to use with as :
CUDA_VISIBLE_DEVICES=0,1, python main.py predict -c configs_c2i/pix256std1_repa_pixnerd_xl.yaml --ckpt_path=XXX.ckpt
```

```bash
# for training
# train
python main.py fit -c configs_c2i/pix256std1_repa_pixnerd_xl.yaml
```
For T2i, we use GenEval and DPG to collect metrics.

## CIFAR-10 Support

This fork includes CIFAR-10 dataset support with optimized model architecture for 32×32 resolution.

**Quick Start:**
```bash
# Clean any previous runs (if needed)
bash cleanup_workdir.sh

# Train on CIFAR-10 (dataset auto-downloads)
python main.py fit -c configs_c2i/cifar_basic_v1.yaml

# Test setup
python test_cifar10.py
```

**Model Configuration:**
- Resolution: 32×32 (vs 256×256 for ImageNet)
- Patch Size: 4 (vs 16 for ImageNet)
- Hidden Size: 384 (vs 1152 for ImageNet-XL)
- Blocks: 12 (vs 30 for ImageNet-XL)
- Classes: 10 (vs 1000 for ImageNet)

**Documentation:** See [CIFAR10_README.md](CIFAR10_README.md) for detailed instructions.

### Multi-Resolution Generation (Following Paper Approach)

**PixNerd's Key Feature**: Train ONE model, generate at MULTIPLE resolutions (32×32, 64×64, 128×128+).

#### Option 1: Use Existing 32×32 Checkpoint (Quick Test)

```bash
# Use your existing 32×32 checkpoint to generate at 128×128
python generate_simple.py \
    --ckpt ./workdirs/exp_cifar10_basic_pixnerd_base/last.ckpt \
    --resolution 128 \
    --num-images 10000
```

Test this first! The neural field should enable higher resolution generation from your 32×32 trained model.

#### Option 2: Train New Model at 128×128 (Best Quality)

```bash
# Train multi-resolution model at 128×128 for best quality
python main.py fit -c configs_c2i/cifar_multires_128.yaml

# Generate at multiple resolutions with SAME checkpoint
python generate_multires.py --ckpt <checkpoint> --multi

# Generates:
# - cifar10_32x32.npz
# - cifar10_64x64.npz
# - cifar10_128x128.npz
```

**How It Works**: Neural field positional encoding with normalized [0,1] coordinates enables resolution flexibility - train once, generate at any resolution!

**Documentation**:
- [MULTIRES_README.md](MULTIRES_README.md) - Complete documentation
- [QUICKSTART_MULTIRES.md](QUICKSTART_MULTIRES.md) - Quick reference

**Alternative**: For cascaded super-resolution (2-stage pipeline), see [CIFAR10_SR_README.md](CIFAR10_SR_README.md)

## Reference
```bibtex
@article{2507.23268,
Author = {Shuai Wang and Ziteng Gao and Chenhui Zhu and Weilin Huang and Limin Wang},
Title = {PixNerd: Pixel Neural Field Diffusion},
Year = {2025},
Eprint = {arXiv:2507.23268},
}
```

## Acknowledgement
The code is mainly built upon [FlowDCN](https://github.com/MCG-NJU/FlowDCN) and [DDT](https://github.com/MCG-NJU/DDT).
