"""
Super-Resolution Lightning Model
Extends LightningModel to handle image-based conditioning via input concatenation
"""
import torch
from src.lightning_model import LightningModel


class SuperResolutionLightningModel(LightningModel):
    """
    Lightning model for image-conditioned super-resolution.

    Key differences from base model:
    - Concatenates upsampled LR image with noisy HR image before passing to denoiser
    - Input to denoiser: [B, 6, H, W] = concat([noisy_hr, lr_upsampled], dim=1)
    - Denoiser must have in_channels=6 to handle concatenated input
    """

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        # x is HR target image [B, 3, H, W]
        # metadata['lr_upsampled'] is conditioning image [B, 3, H, W] at HR resolution

        with torch.no_grad():
            # VAE encode (identity for pixel-space)
            x = self.vae.encode(x)

            # Get conditioning and unconditioning
            condition, uncondition = self.conditioner(y, metadata)

        # Concatenate LR conditioning with HR target for training
        # The diffusion trainer will add noise to x and concatenate with lr_upsampled
        loss = self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser,
            self.diffusion_sampler,
            x,  # HR target
            condition,  # Class labels
            uncondition,  # Unconditioned class labels
            metadata  # Contains 'lr_upsampled' for image conditioning
        )

        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        # xT is noise at HR resolution [B, 3, H, W]
        # metadata['lr_upsampled'] is conditioning image [B, 3, H, W]

        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        # Sample SR images (conditioned on lr_upsampled via concatenation)
        samples = self.diffusion_sampler(
            self.ema_denoiser if not self.eval_original_model else self.denoiser,
            xT,
            condition,
            uncondition,
            metadata  # Contains 'lr_upsampled'
        )

        samples = self.vae.decode(samples)

        # Convert to uint8 [0, 255]
        from src.models.autoencoder.base import fp2uint8
        samples = fp2uint8(samples)

        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples
