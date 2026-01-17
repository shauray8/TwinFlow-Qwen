import torch
import torch.nn as nn
from typing import Optional, Literal
import os

class RewardModelWrapper(nn.Module):
    """Unified wrapper for multiple reward models"""
    
    def __init__(
        self,
        reward_type: Literal["hpsv2", "pickscore", "imagereward"] = "hpsv2",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.reward_type = reward_type
        self.device = device
        
        # Load reward model
        if reward_type == "hpsv2":
            from transformers import CLIPModel, CLIPProcessor
            model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            self.model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
            self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            
        elif reward_type == "pickscore":
            from transformers import AutoProcessor, AutoModel
            model_id = "yuvalkirstain/PickScore_v1"
            self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            
        elif reward_type == "imagereward":
            import ImageReward as RM
            self.model = RM.load("ImageReward-v1.0", download_root=cache_dir)
            self.processor = None  # ImageReward has built-in preprocessing
        
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
    
    @torch.no_grad()
    def compute_reward(
        self,
        images: torch.Tensor,  # [-1, 1] range, (B, C, H, W)
        prompts: list[str],
    ) -> torch.Tensor:
        """
        Args:
            images: Generated images in [-1, 1] range
            prompts: List of text prompts
        Returns:
            rewards: (B,) tensor of reward scores
        """
        # Convert [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        
        if self.reward_type in ["hpsv2", "pickscore"]:
            # Convert to PIL for CLIP processing
            from torchvision.transforms.functional import to_pil_image
            pil_images = [to_pil_image(img.cpu()) for img in images]
            
            # Process inputs
            inputs = self.processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Compute similarity as reward
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            rewards = (image_embeds * text_embeds).sum(dim=-1)
            
            # Scale to [0, 1] range approximately
            if self.reward_type == "hpsv2":
                rewards = (rewards + 1.0) / 2.0  # cosine sim is in [-1, 1]
            
        elif self.reward_type == "imagereward":
            # ImageReward expects PIL images
            from torchvision.transforms.functional import to_pil_image
            pil_images = [to_pil_image(img.cpu()) for img in images]
            
            rewards = []
            for img, prompt in zip(pil_images, prompts):
                reward = self.model.score(prompt, img)
                rewards.append(reward)
            rewards = torch.tensor(rewards, device=self.device)
        
        return rewards

def compute_reward_gradients(
    reward_model: RewardModelWrapper,
    fake_samples: torch.Tensor,  # Generated samples in [-1, 1]
    prompts: list[str],
) -> torch.Tensor:
    """
    Compute ReFL-style gradients through frozen reward model.
    Uses finite differences as CLIP doesn't support pixel-space gradients.
    """
    device = fake_samples.device
    dtype = fake_samples.dtype
    
    # Compute reward for current samples
    with torch.no_grad():
        base_rewards = reward_model.compute_reward(fake_samples, prompts)
    
    # Compute approximate gradients using finite differences
    # This is the practical approach used in ReFL for frozen reward models
    epsilon = 0.01
    grad_samples = torch.zeros_like(fake_samples)
    
    # Sample random directions for gradient estimation
    num_samples = 4  # Number of random directions
    
    for _ in range(num_samples):
        # Random perturbation
        noise = torch.randn_like(fake_samples) * epsilon
        perturbed = fake_samples + noise
        
        # Compute reward for perturbed samples
        with torch.no_grad():
            perturbed_rewards = reward_model.compute_reward(perturbed, prompts)
        
        # Estimate gradient via finite difference
        reward_diff = (perturbed_rewards - base_rewards).view(-1, 1, 1, 1)
        grad_samples += noise * reward_diff / (epsilon ** 2)
    
    grad_samples = grad_samples / num_samples
    
    return grad_samples.to(dtype)
