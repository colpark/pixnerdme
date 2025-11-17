"""
Image-based conditioning for super-resolution
Conditions on low-resolution images upsampled to target resolution
"""
import torch
import torch.nn as nn
from src.models.conditioner.base import BaseConditioner


class ImageConditioner(BaseConditioner):
    """
    Conditioner that uses upsampled low-resolution images as conditioning.

    For super-resolution, we condition on the bicubic-upsampled LR image.
    This is concatenated with the noisy image during diffusion.

    Args:
        image_key: Key in metadata dict containing the conditioning image (default: 'lr_upsampled')
        use_concat: If True, returns image for concatenation. If False, returns embedding (default: True)
    """

    def __init__(self, image_key='lr_upsampled', use_concat=True):
        super().__init__()
        self.image_key = image_key
        self.use_concat = use_concat

    def _impl_condition(self, y, metadata):
        """
        Extract conditioning image from metadata.

        Args:
            y: Class labels (not used for image conditioning)
            metadata: Dict containing image_key with conditioning images

        Returns:
            Conditioning tensor [B, C, H, W]
        """
        if self.image_key not in metadata:
            raise KeyError(f"Metadata must contain '{self.image_key}' for image conditioning")

        cond_image = metadata[self.image_key]  # [B, C, H, W]

        # Ensure it's on the right device
        if not cond_image.is_cuda:
            cond_image = cond_image.cuda()

        return cond_image

    def _impl_uncondition(self, y, metadata):
        """
        For unconditional generation (CFG), return zeros.

        Returns:
            Zero tensor with same shape as conditioning image
        """
        cond_image = metadata[self.image_key]
        uncond = torch.zeros_like(cond_image).cuda()
        return uncond


class ImageClassConditioner(BaseConditioner):
    """
    Combined image and class label conditioning for super-resolution.

    Conditions on both:
    1. Upsampled low-resolution image (concatenated with noisy input)
    2. Class label (embedded for class-conditional generation)

    Args:
        num_classes: Number of classes (10 for CIFAR-10)
        image_key: Key in metadata dict containing the conditioning image
        class_embed_dim: Dimension for class embedding (default: 512)
    """

    def __init__(self, num_classes=10, image_key='lr_upsampled', class_embed_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.null_class = num_classes
        self.image_key = image_key

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes + 1, class_embed_dim)  # +1 for null class

    def _extract_image_cond(self, metadata):
        """Extract and prepare image conditioning"""
        if self.image_key not in metadata:
            raise KeyError(f"Metadata must contain '{self.image_key}' for image conditioning")

        cond_image = metadata[self.image_key]
        if not cond_image.is_cuda:
            cond_image = cond_image.cuda()
        return cond_image

    def _impl_condition(self, y, metadata):
        """
        Return dict with both image and class conditioning.

        Args:
            y: Class labels [B]
            metadata: Dict containing conditioning image

        Returns:
            Dict with 'image': [B, C, H, W] and 'class': [B, embed_dim]
        """
        # Image conditioning
        image_cond = self._extract_image_cond(metadata)

        # Class conditioning
        class_labels = torch.tensor(y).long().cuda()
        class_cond = self.class_embedding(class_labels)

        return {
            'image': image_cond,
            'class': class_cond
        }

    def _impl_uncondition(self, y, metadata):
        """
        Return unconditional inputs for classifier-free guidance.

        Returns:
            Dict with zero image and null class embedding
        """
        # Zero image
        image_cond = self._extract_image_cond(metadata)
        zero_image = torch.zeros_like(image_cond).cuda()

        # Null class
        batch_size = len(y)
        null_class_labels = torch.full((batch_size,), self.null_class, dtype=torch.long).cuda()
        null_class_cond = self.class_embedding(null_class_labels)

        return {
            'image': zero_image,
            'class': null_class_cond
        }


class ConcatImageConditioner(BaseConditioner):
    """
    Simple conditioner that returns conditioning image for channel-wise concatenation.

    This is used when the denoiser expects conditioning via input concatenation.
    The conditioning image is concatenated with the noisy image along the channel dimension.

    For PixNerd SR: input becomes [B, 6, H, W] = concat([noisy_hr, lr_upsampled], dim=1)

    Args:
        image_key: Key in metadata containing conditioning image (default: 'lr_upsampled')
    """

    def __init__(self, image_key='lr_upsampled'):
        super().__init__()
        self.image_key = image_key

    def _impl_condition(self, y, metadata):
        """Return conditioning image from metadata"""
        if self.image_key not in metadata:
            raise KeyError(f"Metadata must contain '{self.image_key}' for image conditioning")

        cond_image = metadata[self.image_key]
        if not cond_image.is_cuda:
            cond_image = cond_image.cuda()

        return cond_image

    def _impl_uncondition(self, y, metadata):
        """Return zero image for unconditional generation"""
        cond_image = metadata[self.image_key]
        return torch.zeros_like(cond_image).cuda()
