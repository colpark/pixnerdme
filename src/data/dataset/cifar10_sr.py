"""
CIFAR-10 Super-Resolution Dataset
Generates pairs of 32×32 low-resolution and 128×128 high-resolution images
"""
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import to_tensor
from PIL import Image


class CIFAR10SuperResDataset(CIFAR10):
    """
    CIFAR-10 dataset for super-resolution training.

    Returns pairs of:
    - Low-res: 32×32 CIFAR-10 image (original)
    - High-res: 128×128 upsampled target image

    Args:
        root: Root directory where dataset exists or will be downloaded
        train: If True, creates dataset from training set, otherwise test set
        download: If True, downloads the dataset from the internet
        lr_size: Low resolution size (default: 32)
        hr_size: High resolution size (default: 128)
        upscale_method: Method for creating HR targets ('bicubic' or 'nearest')
    """

    def __init__(self, root, train=True, download=True,
                 lr_size=32, hr_size=128, upscale_method='bicubic'):
        super().__init__(root, train=train, download=download)

        self.lr_size = lr_size
        self.hr_size = hr_size
        self.upscale_method = upscale_method

        # Normalize to [-1, 1] range
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx: int):
        # Get original CIFAR-10 image (32×32) and label
        image, target = super().__getitem__(idx)  # PIL Image

        # Convert to tensor
        lr_image = to_tensor(image)  # [3, 32, 32] in [0, 1]

        # Create high-resolution target by upsampling
        # For CIFAR-10, we artificially create HR targets since originals are 32×32
        hr_image = F.interpolate(
            lr_image.unsqueeze(0),
            size=(self.hr_size, self.hr_size),
            mode=self.upscale_method,
            align_corners=False if self.upscale_method == 'bicubic' else None
        ).squeeze(0)  # [3, 128, 128]

        # Normalize both to [-1, 1]
        lr_normalized = self.normalize(lr_image)
        hr_normalized = self.normalize(hr_image)

        # For conditioning, we need upsampled LR at HR resolution
        lr_upsampled = F.interpolate(
            lr_image.unsqueeze(0),
            size=(self.hr_size, self.hr_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # [3, 128, 128]
        lr_upsampled_normalized = self.normalize(lr_upsampled)

        # Metadata
        metadata = {
            "raw_lr": lr_image,  # Original 32×32
            "raw_hr": hr_image,  # Target 128×128
            "lr_upsampled": lr_upsampled_normalized,  # Conditioning input (128×128)
            "class": target,
        }

        # Return HR image as target, class label, and metadata
        return hr_normalized, target, metadata


class CIFAR10SuperResEvalDataset(torch.utils.data.Dataset):
    """
    Evaluation dataset for super-resolution.
    Uses real CIFAR-10 test images (no augmentation).

    Args:
        root: Root directory where dataset exists
        pretrained_ckpt: Path to pretrained 32×32 CIFAR-10 model (optional)
        hr_size: High resolution target size (default: 128)
        max_num_instances: Maximum number of test instances (default: 10000)
    """

    def __init__(self, root, pretrained_ckpt=None, hr_size=128, max_num_instances=10000):
        # Load CIFAR-10 test set
        self.cifar10_test = CIFAR10(root, train=False, download=True)
        self.hr_size = hr_size
        self.max_num_instances = min(max_num_instances, len(self.cifar10_test))
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx):
        # Get CIFAR-10 test image
        image, target = self.cifar10_test[idx]

        # Convert to tensor
        lr_image = to_tensor(image)  # [3, 32, 32]

        # Create HR target by upsampling
        hr_image = F.interpolate(
            lr_image.unsqueeze(0),
            size=(self.hr_size, self.hr_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        # Normalize
        lr_normalized = self.normalize(lr_image)
        hr_normalized = self.normalize(hr_image)

        # Upsampled LR for conditioning
        lr_upsampled = F.interpolate(
            lr_image.unsqueeze(0),
            size=(self.hr_size, self.hr_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        lr_upsampled_normalized = self.normalize(lr_upsampled)

        metadata = {
            "raw_lr": lr_image,
            "raw_hr": hr_image,
            "lr_upsampled": lr_upsampled_normalized,
            "class": target,
            "idx": idx,
        }

        return hr_normalized, target, metadata


class CIFAR10GeneratedSuperResDataset(torch.utils.data.Dataset):
    """
    Super-resolution dataset using generated 32×32 images from pretrained model.

    This is for the full pipeline:
    1. Generate 32×32 images with pretrained CIFAR-10 model
    2. Super-resolve to 128×128 with SR model

    Args:
        generated_lr_path: Path to .npz file with generated 32×32 images
        hr_size: High resolution target size (default: 128)
    """

    def __init__(self, generated_lr_path, hr_size=128):
        import numpy as np

        # Load generated 32×32 images
        data = np.load(generated_lr_path)
        self.lr_images = torch.from_numpy(data['arr_0'])  # [N, H, W, C] uint8

        # Convert to [N, C, H, W] float [0, 1]
        self.lr_images = self.lr_images.permute(0, 3, 1, 2).float() / 255.0

        self.hr_size = hr_size
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = self.lr_images[idx]  # [3, 32, 32]

        # Normalize
        lr_normalized = self.normalize(lr_image)

        # Upsampled LR for conditioning (at HR resolution)
        lr_upsampled = F.interpolate(
            lr_image.unsqueeze(0),
            size=(self.hr_size, self.hr_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        lr_upsampled_normalized = self.normalize(lr_upsampled)

        # Generate random noise at HR resolution for generation
        noise = torch.randn(3, self.hr_size, self.hr_size)

        # Class label (cycle through 10 classes if not provided)
        class_label = idx % 10

        metadata = {
            "raw_lr": lr_image,
            "lr_upsampled": lr_upsampled_normalized,
            "class": class_label,
        }

        # Return noise for generation
        return noise, class_label, metadata
