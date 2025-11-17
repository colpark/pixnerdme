"""
Flow Matching Sampler for Super-Resolution
Handles image conditioning during sampling
"""
import torch
from src.diffusion.flow_matching.sampling import EulerSampler


class SuperResolutionEulerSampler(EulerSampler):
    """
    Euler ODE sampler adapted for image-conditioned super-resolution.

    Concatenates upsampled LR image with noisy samples during denoising.

    Args:
        num_steps: Number of sampling steps
        guidance: Classifier-free guidance scale
        guidance_interval_min: Min time for guidance
        guidance_interval_max: Max time for guidance
        scheduler: Flow scheduler
        w_scheduler: Weight scheduler (optional)
        guidance_fn: Guidance function
        step_fn: ODE step function
        image_cond_key: Key in metadata for conditioning image
    """

    def __init__(self, num_steps=50, guidance=2.0,
                 guidance_interval_min=0.0, guidance_interval_max=1.0,
                 scheduler=None, w_scheduler=None,
                 guidance_fn=None, step_fn=None,
                 image_cond_key='lr_upsampled'):
        super().__init__(num_steps=num_steps, guidance=guidance,
                        guidance_interval_min=guidance_interval_min,
                        guidance_interval_max=guidance_interval_max,
                        scheduler=scheduler, w_scheduler=w_scheduler,
                        guidance_fn=guidance_fn, step_fn=step_fn)
        self.image_cond_key = image_cond_key

    def forward(self, net, x, condition, uncondition=None, metadata=None):
        """
        Sample SR images with image conditioning.

        Args:
            net: Denoiser network (requires in_channels=6)
            x: Initial noise [B, 3, H, W] at HR resolution
            condition: Class conditioning
            uncondition: Unconditional class labels (for CFG)
            metadata: Dict containing image_cond_key with LR conditioning

        Returns:
            Super-resolved images [B, 3, H, W]
        """
        if metadata is None or self.image_cond_key not in metadata:
            raise ValueError(f"Metadata must contain '{self.image_cond_key}' for SR sampling")

        # Get conditioning image
        lr_upsampled = metadata[self.image_cond_key]  # [B, 3, H, W]
        if not lr_upsampled.is_cuda:
            lr_upsampled = lr_upsampled.cuda()

        batch_size = x.shape[0]
        device = x.device

        # Initialize scheduler
        timesteps = self.scheduler.get_timesteps(self.num_steps, device=device)

        # Start from noise
        x_t = x

        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device)

            # Concatenate with conditioning image
            x_t_conditioned = torch.cat([x_t, lr_upsampled], dim=1)  # [B, 6, H, W]

            # Predict velocity with conditioning
            if uncondition is not None and self.guidance > 1.0:
                # Classifier-free guidance
                if self.guidance_interval_min <= t <= self.guidance_interval_max:
                    # Conditional prediction
                    v_cond = net(x_t_conditioned, t_batch, condition)

                    # Unconditional prediction
                    v_uncond = net(x_t_conditioned, t_batch, uncondition)

                    # Apply guidance
                    v_pred = v_uncond + self.guidance * (v_cond - v_uncond)
                else:
                    v_pred = net(x_t_conditioned, t_batch, condition)
            else:
                v_pred = net(x_t_conditioned, t_batch, condition)

            # Euler ODE step
            if i < len(timesteps) - 1:
                dt = timesteps[i + 1] - t
                x_t = x_t + v_pred * dt
            else:
                # Last step
                x_t = x_t + v_pred * (1.0 - t)

        return x_t
