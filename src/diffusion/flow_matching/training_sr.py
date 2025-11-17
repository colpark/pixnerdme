"""
Flow Matching Trainer for Super-Resolution
Handles concatenation of conditioning images with noisy inputs
"""
import torch
import torch.nn as nn
from src.diffusion.flow_matching.training import FlowMatchingTrainer


class SuperResolutionFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Flow matching trainer adapted for image-conditioned super-resolution.

    Key modification:
    - Concatenates upsampled LR image with noisy HR image before denoising
    - Input to denoiser: concat([x_t, lr_upsampled], dim=1) → [B, 6, H, W]

    Args:
        lognorm_t: Use log-normal time sampling
        timeshift: Time shift parameter
        scheduler: Flow scheduler class
        loss_weight_fn: Loss weighting function
        image_cond_key: Key in metadata for conditioning image (default: 'lr_upsampled')
    """

    def __init__(self, lognorm_t=True, timeshift=1.0,
                 scheduler=None, loss_weight_fn=None,
                 image_cond_key='lr_upsampled'):
        super().__init__(lognorm_t=lognorm_t, timeshift=timeshift,
                        scheduler=scheduler, loss_weight_fn=loss_weight_fn)
        self.image_cond_key = image_cond_key

    def forward(self, net, ema_net, sampler, x, condition, uncondition, metadata):
        """
        Training step with image conditioning.

        Args:
            net: Denoiser network (requires in_channels=6)
            ema_net: EMA denoiser
            sampler: Sampler (not used during training)
            x: HR target images [B, 3, H, W]
            condition: Class labels [B] or embeddings
            uncondition: Unconditional class labels
            metadata: Dict containing image_cond_key with LR conditioning images

        Returns:
            Dict with 'loss' key
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample time steps
        if self.lognorm_t:
            u = torch.randn((batch_size,), device=device)
            t = torch.sigmoid(u)
        else:
            t = torch.rand((batch_size,), device=device)

        # Time shifting
        t = t * self.timeshift

        # Sample noise
        noise = torch.randn_like(x)

        # Flow matching: x_t = (1-t)*noise + t*x
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * x

        # Get conditioning image
        if self.image_cond_key not in metadata:
            raise KeyError(f"Metadata must contain '{self.image_cond_key}' for SR training")

        lr_upsampled = metadata[self.image_cond_key]  # [B, 3, H, W]
        if not lr_upsampled.is_cuda:
            lr_upsampled = lr_upsampled.cuda()

        # Concatenate noisy HR with upsampled LR conditioning
        x_t_conditioned = torch.cat([x_t, lr_upsampled], dim=1)  # [B, 6, H, W]

        # Classifier-free guidance: randomly use unconditional
        if uncondition is not None:
            cfg_prob = 0.1  # 10% unconditional
            mask = torch.rand(batch_size, device=device) < cfg_prob
            y_input = torch.where(mask, uncondition, condition)
        else:
            y_input = condition

        # Predict velocity
        v_pred = net(x_t_conditioned, t, y_input)  # [B, 3, H, W]

        # Target velocity: v = x - noise
        v_target = x - noise

        # Compute loss
        loss_weight = self.loss_weight_fn(t) if self.loss_weight_fn else torch.ones_like(t)
        loss_weight = loss_weight.view(-1, 1, 1, 1)

        loss = torch.mean(loss_weight * (v_pred - v_target) ** 2)

        return {"loss": loss}
