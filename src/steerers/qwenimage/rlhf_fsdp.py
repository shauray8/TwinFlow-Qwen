import os
import sys
import time
from absl import app, flags
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import random
from torch.utils.data import DataLoader
from torch.amp import autocast as torch_autocast

from omegaconf import OmegaConf
from copy import deepcopy
from torchvision.utils import save_image, make_grid

from data import Text2ImageDataset, Text2ImageParquetDataset
from data.bucketed_dataset_sharegpt import BucketedShareGPTDataset, BucketBatchSampler
from services.tools import create_logger
from networks import MODELS
from methodes import METHODES


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=lock_rank)
    torch.cuda.set_device(lock_rank)
    _ = torch.tensor([1], device=f"cuda:{lock_rank}")


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()

def bucketed_collate_fn(batch):
    """
    Custom collate for variable-sized images from buckets.
    All images in batch MUST be same size (enforced by BucketBatchSampler).
    """
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    bucket_keys = [item['bucket_key'] for item in batch]
    heights = [item['height'] for item in batch]
    widths = [item['width'] for item in batch]
    zs = [item['z'] for item in batch]
    
    # Stack z tensors
    if zs[0].numel() > 0:
        z = torch.stack(zs)
    else:
        z = torch.tensor([])
    
    return {
        'image': images,
        'text': texts,
        'bucket_key': bucket_keys,
        'height': heights[0],
        'width': widths[0],
        'z': z,
    }

def save_ckpt(
    ckpt_root_dir,
    model_to_save,
    optimizer,
    global_step,
):
    """
    Save checkpoint with FSDP using CPU offload and proper synchronization.
    All ranks participate but only rank 0 writes to disk.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    import torch.distributed as dist
    
    rank = dist.get_rank()
    
    # Clear cache before gathering state dict
    torch.cuda.empty_cache()
    
    # ==================== CRITICAL: All ranks must participate ====================
    model_dir = os.path.join(ckpt_root_dir, f"global_step_{global_step}", "model")
    opt_dir = os.path.join(ckpt_root_dir, f"global_step_{global_step}", "optimizer")
    
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
    
    # Synchronize before state dict gathering
    dist.barrier()
    
    try:
        # Configure CPU offload (critical for avoiding OOM)
        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        
        # All ranks participate in gathering state dict
        with FSDP.state_dict_type(
            model_to_save.transformer,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            # This call happens on ALL ranks, but only rank 0 gets the full state_dict
            state_dict = model_to_save.transformer.state_dict()
            
            # Only rank 0 saves to disk
            if rank == 0:
                # Move to CPU if not already
                cpu_state = {
                    k: v.cpu() if v.is_cuda else v 
                    for k, v in state_dict.items()
                }
                
                # Save model using transformers save_pretrained
                # Note: This assumes transformer.module has a save_pretrained method
                torch.save(cpu_state, os.path.join(model_dir, "pytorch_model.bin"))
                
                # Save optimizer
                torch.save(
                    optimizer.state_dict(), 
                    os.path.join(opt_dir, "optimizer.pt")
                )
                
                print(f"[Rank 0] Saved checkpoint to {ckpt_root_dir}/global_step_{global_step}")
        
        # Synchronize after saving
        dist.barrier()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[Rank {rank}] OOM during checkpoint save, clearing cache and retrying...")
            torch.cuda.empty_cache()
            # Don't retry, just continue - checkpoint will be incomplete but training continues
            dist.barrier()  # Still need to sync to prevent deadlock
        else:
            raise e
    
    # Clear cache after checkpoint
    torch.cuda.empty_cache()
    # ===========================================================================

class PrecomputedBatchSampler:
    def __init__(self, batches, shuffle=True):
        self.batches = batches
        self.shuffle = shuffle
    
    def __iter__(self):
        batches = self.batches.copy()
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

def main(_):
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    # ==================== CHANGE 1: Extract method_type BEFORE passing to METHODES ====================
    method_type = config["method"].pop("method_type")
    
    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    parent_path = config_path.split("/")
    exp_name = os.path.join(parent_path[-2], parent_path[-1].split(".")[0])
    config["train"]["output_dir"] = os.path.join(
        config["train"]["output_dir"], exp_name
    )
    config["train"]["save_checkpoint_path"] = os.path.join(
        config["train"]["output_dir"], "checkpoints"
    )
    if is_main_process(rank):
        os.makedirs(config["train"]["output_dir"], exist_ok=True)
        os.makedirs(config["train"]["save_checkpoint_path"], exist_ok=True)

    logger = create_logger(__name__, config["train"]["output_dir"])
    if is_main_process(rank) and config["train"].get("use_wandb", False):
        import wandb
        
        # Auto-generate run name if not provided
        run_name = config["train"].get("wandb_run_name")
        if run_name is None:
            run_name = f"twinflow_{exp_name}_{config['train']['seed']}"
        
        # Initialize wandb
        wandb.init(
            project=config["train"].get("wandb_project", "twinflow-qwen"),
            entity=config["train"].get("wandb_entity"),
            name=run_name,
            tags=config["train"].get("wandb_tags", []),
            config=config,  # Log entire config
            resume="allow",  # Allow resuming if run_id exists
        )
        
        logger.info("=" * 70)
        logger.info("WANDB INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"  Project: {wandb.run.project}")
        logger.info(f"  Run Name: {wandb.run.name}")
        logger.info(f"  Run URL: {wandb.run.url}")
        logger.info("=" * 70)

    demoimages_dir = os.path.join(config["train"]["output_dir"], "demoimages")
    if is_main_process(rank):
        os.makedirs(demoimages_dir, exist_ok=True)
    demo_c = [
        "a photo of a bench",
        "a photo of a cow",
    ]

    set_seed(config["train"]["seed"], rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None

    # --- Load pipeline and models ---
    wrapped_model = MODELS[config["model"]["model_name"]](
        model_id=config["model"]["model_path"],
        aux_time_embed=config["model"]["aux_time_embed"],
        text_dtype=mixed_precision_dtype,
        imgs_dtype=mixed_precision_dtype,
    )
    # wrapped_model.transformer.to(device)
    wrapped_model.transformer.requires_grad_(True)
    wrapped_model.transformer.enable_gradient_checkpointing()

    # NOTE: a little bit hard to directly add an EMA on 8GPU, you could modify to LoRA
    # training, as shown in `src/steerers/stable_diffusion_3/sft_ddp.py`
    
    # wrapped_model.add_module("ema_transformer", deepcopy(wrapped_model.transformer))
    # wrapped_model.ema_transformer.requires_grad_(False)

    no_split_modules = wrapped_model.get_no_split_modules()
    wrapped_model.transformer.float()
    wrapped_model.transformer = FSDP(
        wrapped_model.transformer,
        device_id=local_rank,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda module: module.__class__.__name__ in no_split_modules
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=False,
    )
    # wrapped_model.ema_transformer = FSDP(
    #     wrapped_model.ema_transformer,
    #     device_id=local_rank,
    #     auto_wrap_policy=partial(
    #         lambda_auto_wrap_policy,
    #         lambda_fn=lambda module: module.__class__.__name__ in no_split_modules
    #     ),
    #     mixed_precision=MixedPrecision(
    #         param_dtype=torch.bfloat16,
    #         reduce_dtype=torch.float32,
    #         buffer_dtype=torch.float32,
    #     ),
    #     backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    #     forward_prefetch=True,
    #     use_orig_params=False,
    # )

    # ==================== CHANGE 2: Initialize method AFTER model loading ====================
    # This allows passing VAE reference if needed for RL
    method_config = config["method"].copy()
    
    # OPTIONAL: Pass VAE for RL pixel-space decoding (if your model has one)
    if hasattr(wrapped_model, 'vae'):
        method_config['vae_for_rl'] = wrapped_model.vae
    
    method = METHODES[method_type](**method_config)
    
    # ==================== CHANGE 3: Log RL configuration ====================
    if is_main_process(rank):
        if hasattr(method, 'use_rl') and method.use_rl:
            logger.info("=" * 70)
            logger.info("RL CONFIGURATION")
            logger.info("=" * 70)
            logger.info(f"  RL Enabled: {method.use_rl}")
            logger.info(f"  RL Warmup Steps: {method.rl_warmup_steps}")
            logger.info(f"  RL Weight: {method.rl_weight}")
            logger.info(f"  Reward Model: {method.reward_model_type}")
            logger.info("=" * 70)
        
        if hasattr(method, 'use_dynamic_renoise') and method.use_dynamic_renoise:
            logger.info("=" * 70)
            logger.info("DYNAMIC RENOISE CONFIGURATION")
            logger.info("=" * 70)
            logger.info(f"  Dynamic Renoise: {method.use_dynamic_renoise}")
            logger.info(f"  Renoise Schedule: {method.renoise_schedule}")
            logger.info("=" * 70)

    transformer_ddp = wrapped_model.transformer
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.train()

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        # transformer_ddp.module.parameters(),
        lr=config["train"]["lr"],
        betas=config["train"]["betas"],
        weight_decay=config["train"]["weight_decay"],
        foreach=True,
        # eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    full_dataset = BucketedShareGPTDataset(
        metadata_path=config["data"]["bucket_metadata_path"],
        dataset_name=config["data"]["dataset_name"],
        split="train",
        image_dir=config["data"]["image_dir"],
    )
    
    if is_main_process(rank):
        logger.info(f"Total dataset size: {len(full_dataset)}")
    
    # ==================== NEW APPROACH: Bucket-aware distributed sampling ====================
    # Group samples by bucket
    bucket_to_indices = {}
    for idx, sample in enumerate(full_dataset.samples):
        bucket_key = sample['bucket_key']
        if bucket_key not in bucket_to_indices:
            bucket_to_indices[bucket_key] = []
        bucket_to_indices[bucket_key].append(idx)
    
    # Distribute buckets across ranks
    all_batches = []
    for bucket_key, indices in bucket_to_indices.items():
        # Shuffle indices within bucket
        random.Random(config["train"]["seed"]).shuffle(indices)
        
        # Create batches for this bucket
        batch_size = config["train"]["micro_batch_size"]
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if len(batch) == batch_size:  # Only full batches
                all_batches.append(batch)
    
    # Shuffle all batches
    random.Random(config["train"]["seed"]).shuffle(all_batches)
    
    # Distribute batches across ranks
    local_batches = [all_batches[i] for i in range(rank, len(all_batches), world_size)]
    
    if is_main_process(rank):
        logger.info(f"Total batches: {len(all_batches)}")
        logger.info(f"Batches per rank: {len(local_batches)}")
        logger.info(f"Bucket distribution:")
        for bucket_key, indices in bucket_to_indices.items():
            num_batches = len(indices) // batch_size
            logger.info(f"  {bucket_key}: {len(indices)} samples, ~{num_batches} batches")
    
    # Create custom sampler that yields batches
    class PrecomputedBatchSampler:
        def __init__(self, batches, shuffle=True):
            self.batches = batches
            self.shuffle = shuffle
        
        def __iter__(self):
            batches = self.batches.copy()
            if self.shuffle:
                random.shuffle(batches)
            for batch in batches:
                yield batch
        
        def __len__(self):
            return len(self.batches)
    
    batch_sampler = PrecomputedBatchSampler(local_batches, shuffle=True)
    
    train_dataloader = DataLoader(
        full_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=bucketed_collate_fn,
    )
    # ==================== END Fixed Dataset Loading ====================
    
    # Calculate training steps
    total_train_batch_size = (
        config["train"]["micro_batch_size"]
        * world_size
        * config["train"]["grad_accumulation_steps"]
    )
    effective_grad_accum_steps = config["train"]["grad_accumulation_steps"]
    total_train_steps = len(local_batches)  # Number of batches this rank will process
    
    if is_main_process(rank):
        logger.info(f"Training configuration:")
        logger.info(f"  Micro batch size: {config['train']['micro_batch_size']}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Grad accumulation steps: {effective_grad_accum_steps}")
        logger.info(f"  Total train steps per epoch: {total_train_steps}")
        
    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config["train"]["load_checkpoint_path"] != "":
        load_path = config["train"]["load_checkpoint_path"]
        logger.info(f"Resuming from checkpoint: {load_path}")

        wrapped_model.transformer.module.transformer = (
            wrapped_model.transformer.module.transformer.from_pretrained(
                os.path.join(load_path, "model"), torch_dtype=torch.float32
            ).to(device)
        )
        wrapped_model.ema_transformer.transformer = (
            wrapped_model.ema_transformer.transformer.from_pretrained(
                os.path.join(load_path, "ema"), torch_dtype=torch.float32
            ).to(device)
        )
        opt_state = torch.load(
            os.path.join(load_path, "optimizer", "optimizer.pt"), map_location=device
        )
        optimizer.load_state_dict(opt_state)

        global_step_str = os.path.basename(os.path.normpath(load_path))
        if global_step_str.startswith("global_step_"):
            global_step = int(global_step_str.replace("global_step_", ""))
            first_epoch = global_step // total_train_steps  # approximate
        else:
            global_step = 0
            first_epoch = 0

    # if is_main_process(rank):
    with (
        torch.no_grad(),
        torch_autocast(
            enabled=enable_amp, dtype=mixed_precision_dtype, device_type="cuda"
        ),
    ):
        demox = wrapped_model.sample(
            demo_c,
            sampler=method.sampling_loop,
            height=1024, #config["data"]["height"],
            width=1024, #config["data"]["width"],
            seed=42,
            cfg_scale=0.0,
            return_traj=True,
        )
        demoimages_path = f"{demoimages_dir}/{global_step:07d}.png"
        demox = (
            demox.view(-1, len(demo_c), *demox.shape[-3:])
            .permute(1, 0, 2, 3, 4)
            .reshape(-1, *demox.shape[-3:])
        )
        save_image(
            (demox + 1) / 2,
            os.path.join(demoimages_path),
            nrow=len(demox) // len(demo_c),
        )
        logger.info(f"Saved demoimages to {demoimages_path}")
        del demox

    logger.info("***** Running training *****")
    optimizer.zero_grad()

    for epoch in range(first_epoch, config["train"]["num_train_epochs"]):
        if hasattr(train_dataloader, "set_epoch"):
            train_sampler.set_epoch(epoch)

        train_iter = iter(train_dataloader)
        optimizer_step = 0
        for _ in range(total_train_steps):
            global_step += 1
            is_sync_step = global_step % effective_grad_accum_steps == 0
            if not is_sync_step:
                sync_context = transformer_ddp.no_sync()
            else:
                from contextlib import nullcontext

                sync_context = nullcontext()

            torch.cuda.synchronize()
            
            batch = next(train_iter)#[0]
            
            # NOTE: for editing training, just return additional ref image in batch
            text, image, z = batch["text"], batch["image"].cuda(), batch["z"]
            
            # ==================== CHANGE 4: Extract prompts for RL ====================
            prompts = text  # Store prompts as list[str] for reward model

            # logger.info(text[:2])
            start_time = time.time()
            if z.numel() != 0:
                z = z.cuda()
            else:
                z = None

            with sync_context:
                with torch.no_grad():
                    (
                        prompt_embeds,
                        # pooled_prompt_embeds,
                        prompt_embeds_mask,
                        uncond_prompt_embeds,
                        # pooled_uncond_prompt_embeds,
                        uncond_prompt_embeds_mask,
                    ) = wrapped_model.encode_prompt(text, do_cfg=True)
                    prompt_embeds = prompt_embeds.to(torch.float32)
                    uncond_prompt_embeds = uncond_prompt_embeds.to(torch.float32)

                    latents = wrapped_model.pixels_to_latents(image)
                    latents = latents.to(torch.float32)

                # if is_main_process(rank):
                #     logger.info(f"\033[92mcond size: {prompt_embeds.shape}\033[0m")
                #     logger.info(f"\033[92muncond size: {uncond_prompt_embeds.shape}\033[0m")
                #     logger.info(f"\033[92mlatents size: {latents.shape}\033[0m")

                with torch_autocast(
                    enabled=enable_amp,
                    dtype=mixed_precision_dtype,
                    device_type="cuda",
                    cache_enabled=False,
                ):
                    # ==================== CHANGE 5: Pass prompts to training_step ====================
                    loss = method.training_step(
                        wrapped_model,
                        latents,
                        c=[prompt_embeds,
                        # pooled_prompt_embeds,
                        prompt_embeds_mask],
                        e=[uncond_prompt_embeds,
                        # pooled_uncond_prompt_embeds,
                        uncond_prompt_embeds_mask],
                        step=(global_step - 1),  # Step for dynamic scheduling
                        v=z,
                        prompts=prompts,  # NEW: Pass prompts for RL
                    )

                scaled_loss = loss / effective_grad_accum_steps
                scaled_loss.backward()

            grad_norm = 0.0
            if is_sync_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    transformer_ddp.module.parameters(),
                    config["train"]["max_grad_norm"],
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            actual_loss = scaled_loss * effective_grad_accum_steps
            loss_tensor = actual_loss.detach().clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            delta_time = time.time() - start_time

            # ==================== CHANGE 6: Enhanced logging with RL indicators ====================
            if is_main_process(rank):
                log_msg = (
                    f"step=({global_step:08d}): "
                    f"loss: {avg_loss:.4f}, "
                    f"grad_norm: {grad_norm:.2f}, "
                    f"step_time: {delta_time:.2f}s"
                )
                rl_active = False 
                renoise_bias = 0.0
                
                # Add RL status indicator
                if hasattr(method, 'use_rl') and method.use_rl:
                    if global_step >= method.rl_warmup_steps:
                        log_msg += " [RL:ACTIVE]"
                        rl_active = True
                    elif global_step == method.rl_warmup_steps - 1:
                        log_msg += " [RL:WARMUP-ENDING]"
                    else:
                        warm_pct = (global_step / method.rl_warmup_steps) * 100
                        log_msg += f" [RL:WARMUP {warm_pct:.0f}%]"
                
                # Add dynamic renoise indicator
                if hasattr(method, 'use_dynamic_renoise') and method.use_dynamic_renoise:
                    renoise_bias = method.get_dynamic_renoise_bias(global_step)
                    log_msg += f" [Renoise bias: {renoise_bias:.2f}]"
                
                logger.info(log_msg)

                # ==================== NEW: WandB Logging ====================
                if config["train"].get("use_wandb", False):
                    log_interval = config["train"].get("wandb_log_interval", 10)
                    
                    if global_step % log_interval == 0:
                        wandb_logs = {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/step_time": delta_time,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/bucket": bucket_key,
                        }
                        
                        # Add RL metrics if active
                        if hasattr(method, 'use_rl') and method.use_rl:
                            wandb_logs["rl/active"] = int(rl_active)
                            wandb_logs["rl/warmup_progress"] = min(global_step / method.rl_warmup_steps, 1.0)
                        
                        # Add dynamic renoise metrics
                        if hasattr(method, 'use_dynamic_renoise') and method.use_dynamic_renoise:
                            wandb_logs["renoise/bias"] = renoise_bias
                        
                        wandb.log(wandb_logs, step=global_step)
                # ==================== END WandB Logging ====================
                
                
                # ==================== CHANGE 7: Log RL activation milestone ====================
                if hasattr(method, 'use_rl') and method.use_rl:
                    if global_step == method.rl_warmup_steps:
                        logger.info("=" * 70)
                        logger.info("🚀 RL ACTIVATED - Reward model gradients now active")
                        logger.info("=" * 70)

            if global_step % 100 == 0:
                with (
                    torch.no_grad(),
                    torch_autocast(
                        enabled=enable_amp,
                        dtype=mixed_precision_dtype,
                        device_type="cuda",
                    ),
                ):
                    all_demo_images = []
                    demo_captions = []
                    demo_resolutions = [(1328, 1328)]
                    
                    for height, width in demo_resolutions:
                        demox = wrapped_model.sample(
                            demo_c,
                            sampler=method.sampling_loop,
                            height=height,
                            width=width,
                            seed=42,
                            cfg_scale=0.0,
                            return_traj=False,
                        )
                        all_demo_images.append(demox)
                        # Create captions for each resolution
                        for prompt in demo_c:
                            demo_captions.append(f"{prompt} ({width}x{height})")
                    
                    demox = torch.cat(all_demo_images, dim=0)
                    demoimages_path = f"{demoimages_dir}/{global_step:07d}.png"
                    save_image(
                        (demox + 1) / 2,
                        demoimages_path,
                        nrow=len(demo_c),
                    )
                    logger.info(f"Saved demoimages to {demoimages_path}")
                    
                    # ==================== NEW: Log to WandB ====================
                    
                    if is_main_process(rank) and config["train"].get("use_wandb", False):
                        # Convert tensors to wandb images
                        demo_images_list = []
                        for i, (img, caption) in enumerate(zip(demox, demo_captions)):
                            #img_np = ((img + 1) / 2).cpu().numpy().transpose(1, 2, 0)
                            img_np = ((img + 1) / 2).float().cpu().numpy().transpose(1, 2, 0)
                            img_np = (img_np * 255).astype(np.uint8)
                            demo_images_list.append(wandb.Image(img_np, caption=demo_captions[i]))
                        
                        wandb.log({
                            "samples": demo_images_list,
                        }, step=global_step)
                    # ==================== END WandB Logging ====================
                    
                    del demox, all_demo_images
            if global_step % 1000 == 0:
                with (
                    torch.no_grad(),
                    torch_autocast(
                        enabled=enable_amp,
                        dtype=mixed_precision_dtype,
                        device_type="cuda",
                    ),
                ):
                    save_ckpt(
                        config["train"]["save_checkpoint_path"],
                        wrapped_model,
                        optimizer,
                        global_step,
                    )
                    logger.info(
                        f"Saved ckpt to {config['train']['save_checkpoint_path']}"
                    )
            
            dist.barrier()
            del batch

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)