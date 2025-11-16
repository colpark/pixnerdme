import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from torchvision.transforms.functional import to_tensor


class PixCIFAR10(CIFAR10):
    """
    CIFAR-10 dataset for PixNerd training.

    CIFAR-10 consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images
    and 10000 test images.

    Args:
        root: Root directory where dataset exists or will be downloaded
        train: If True, creates dataset from training set, otherwise test set
        download: If True, downloads the dataset from the internet
        random_crop: If True, applies random crop and horizontal flip augmentation
    """

    def __init__(self, root, train=True, download=True, random_crop=False):
        super().__init__(root, train=train, download=download)

        # CIFAR-10 images are already 32x32, no resizing needed
        if random_crop:
            # Training augmentation: random crop with padding and horizontal flip
            self.transform = Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
            ])
        else:
            # No transformation needed for center crop (already 32x32)
            self.transform = None

        # Normalize to [-1, 1] range (mean=0.5, std=0.5 per channel)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx: int):
        # Get image and target from parent CIFAR10 dataset
        image, target = super().__getitem__(idx)

        # Apply augmentation if specified
        if self.transform is not None:
            image = self.transform(image)

        # Convert PIL image to tensor
        raw_image = to_tensor(image)

        # Normalize to [-1, 1]
        normalized_image = self.normalize(raw_image)

        # Metadata for compatibility with PixNerd pipeline
        metadata = {
            "raw_image": raw_image,
            "class": target,
        }

        return normalized_image, target, metadata


class PixCIFAR10NoCrop(CIFAR10):
    """
    CIFAR-10 dataset without augmentation for clean evaluation.

    Use this for validation to see non-augmented image generation.

    Args:
        root: Root directory where dataset exists or will be downloaded
        train: If True, creates dataset from training set, otherwise test set
        download: If True, downloads the dataset from the internet
    """

    def __init__(self, root, train=True, download=True):
        super().__init__(root, train=train, download=download)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)
        raw_image = to_tensor(image)
        normalized_image = self.normalize(raw_image)

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }

        return normalized_image, target, metadata


class CIFAR10RandomNDataset(torch.utils.data.Dataset):
    """
    Random noise dataset for CIFAR-10 evaluation/prediction.
    Generates random noise for each class to test generation capability.

    Args:
        num_classes: Number of classes (10 for CIFAR-10)
        max_num_instances: Maximum number of instances to generate
        latent_shape: Shape of latent noise [C, H, W]
    """

    def __init__(self, num_classes=10, max_num_instances=10000, latent_shape=(3, 32, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.max_num_instances = max_num_instances
        self.latent_shape = latent_shape

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx):
        # Generate random noise
        latent = torch.randn(*self.latent_shape)

        # Cycle through classes
        class_label = idx % self.num_classes

        # Metadata
        metadata = {
            "class": class_label,
        }

        return latent, class_label, metadata
