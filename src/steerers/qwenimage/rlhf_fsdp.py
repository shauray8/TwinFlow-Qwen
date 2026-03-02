import os
import sys
import time
import shutil
import threading
from queue import Queue, Empty
from absl import app, flags
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'backend:native'

class AsyncLogger:
    def __init__(self):
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def log(self, data, step):
        self.queue.put((data, step))
    
    def _worker(self):
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                data, step = item
                try:
                    wandb.log(data, step=step)
                except Exception as e:
                    print(f"logging error: {e}")
                self.queue.task_done()
            except Empty:
                continue
    
    def shutdown(self):
        self.running = False
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Empty:
                break
        self.thread.join(timeout=5.0)

class HFCheckpointUploader: # storage is expensive on verda :(
    def __init__(self, repo_id, token):
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def upload(self, ckpt_dir, step):
        self.queue.put((ckpt_dir, step))

    def _worker(self):
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                ckpt_dir, step = item
                try:
                    path_in_repo = f"checkpoints/global_step_{step}"
                    print(f"[HF Upload] Uploading {ckpt_dir} - {self.repo_id}/{path_in_repo}")
                    self.api.upload_folder(
                        folder_path=ckpt_dir,
                        repo_id=self.repo_id,
                        path_in_repo=path_in_repo,
                        repo_type="model",
                    )
                    print(f"[HF Upload] Upload complete for step {step}")
                    shutil.rmtree(ckpt_dir)
                    print(f"[HF Upload] Deleted local checkpoint: {ckpt_dir}")
                except Exception as e:
                    print(f"[HF Upload Error] step={step}: {e}")
                self.queue.task_done()
            except Empty:
                continue

    def shutdown(self):
        print("[HF Upload] Waiting for uploads to complete...")
        self.queue.join()
        self.running = False
        self.thread.join(timeout=120.0)
        print("[HF Upload] All uploads complete.")


def setup_distributed(rank, local_rank, world_size): #Experimental
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    os.environ["NCCL_ALGO"] = "Ring"  
    os.environ["NCCL_NET_GDR_LEVEL"] = "PHB" 
    os.environ["NCCL_MIN_NCHANNELS"] = "4" 
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["NCCL_BUFFSIZE"] = str(8 * 1024 * 1024) 
    os.environ["NCCL_IB_DISABLE"] = "0" 
    
    if world_size <= 8:
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        os.environ["NCCL_SHM_DISABLE"] = "0"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    _ = torch.tensor([1], device=f"cuda:{local_rank}")
    
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    bucket_keys = [item['bucket_key'] for item in batch]
    heights = [item['height'] for item in batch]
    widths = [item['width'] for item in batch]
    zs = [item['z'] for item in batch]
    
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

def apply_fsdp2_to_model(model, no_split_modules, device_mesh, mp_policy, use_activation_checkpointing=True):
    transformer = model.transformer
    sharded_modules = []
    
    def apply_fsdp_to_blocks(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
           
            if child.__class__.__name__ in no_split_modules:
                fully_shard(
                    child,
                    mesh=device_mesh,
                    reshard_after_forward=True,
                    mp_policy=mp_policy,
                    #communication_dtype=torch.float16,
                )
                sharded_modules.append(child)
            else:
                apply_fsdp_to_blocks(child, full_name)
    
    apply_fsdp_to_blocks(transformer)
    
    if sharded_modules:
        sharded_modules[-1].set_reshard_after_forward(False)
    
    fully_shard(
        transformer,
        mesh=device_mesh,
        reshard_after_forward=True,
        mp_policy=mp_policy,
        #communication_dtype=torch.float16,
    )
    
    return sharded_modules

def setup_explicit_prefetching(model, sharded_modules, num_forward_prefetch=2, num_backward_prefetch=2):
    if not sharded_modules or len(sharded_modules) < 3:
        return
    
    for i, module in enumerate(sharded_modules):
        if i >= len(sharded_modules) - num_forward_prefetch:
            break
        layers_to_prefetch = [
            sharded_modules[i + j] 
            for j in range(1, min(num_forward_prefetch + 1, len(sharded_modules) - i))
        ]
        if hasattr(module, 'set_modules_to_forward_prefetch'):
            module.set_modules_to_forward_prefetch(layers_to_prefetch)
    
    for i, module in enumerate(sharded_modules):
        if i < num_backward_prefetch:
            continue
        layers_to_prefetch = [
            sharded_modules[i - j] 
            for j in range(1, min(num_backward_prefetch + 1, i + 1))
        ]
        if hasattr(module, 'set_modules_to_backward_prefetch'):
            module.set_modules_to_backward_prefetch(layers_to_prefetch)

def save_ckpt_fsdp2(
    ckpt_root_dir,
    model_to_save,
    optimizer,
    scheduler,
    global_step,
    use_dcp=True,
):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
    )
    rank = dist.get_rank()
    torch.cuda.empty_cache()
    ckpt_dir = os.path.join(ckpt_root_dir, f"global_step_{global_step}")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()
    try:
        if use_dcp:
            # DCP Sharded Checkpoint
            state_dict = {
                "model": get_model_state_dict(model_to_save.transformer),
                "optimizer": get_optimizer_state_dict(
                    model_to_save.transformer,
                    optimizer,
                ),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
            }
            
            dcp.save(state_dict, checkpoint_id=ckpt_dir)
            if rank == 0:
                print(f"[Rank 0] Saved DCP checkpoint to {ckpt_dir}")
        else:
            # Full State Dict
            model_dir = os.path.join(ckpt_dir, "model")
            opt_dir = os.path.join(ckpt_dir, "optimizer")
            
            if rank == 0:
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(opt_dir, exist_ok=True)
            
            dist.barrier()
           
            sharded_sd = model_to_save.transformer.state_dict()
            
            cpu_state_dict = {}
            for param_name, sharded_param in sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if rank == 0:
                    cpu_state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
            
            if rank == 0:
                torch.save(cpu_state_dict, os.path.join(model_dir, "pytorch_model.bin"))
                torch.save(optimizer.state_dict(), os.path.join(opt_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(opt_dir, "scheduler.pt"))
                print(f"[Rank 0] Saved full checkpoint to {ckpt_dir}")
        
        dist.barrier()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[Rank {rank}] OOM during checkpoint save")
            torch.cuda.empty_cache()
            dist.barrier()
        else:
            raise e
    
    torch.cuda.empty_cache()

def save_checkpoint_sync(ckpt_root_dir, model_to_save, optimizer, scheduler, global_step):
    """Synchronous checkpoint saving."""
    save_ckpt_fsdp2(ckpt_root_dir, model_to_save, optimizer, scheduler, global_step, use_dcp=True)

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

def run_validation(val_dataloader, wrapped_model, method, device, mp_dtype, global_step, rank, world_size, total_val_batches_global):
    dist.barrier()
    transformer = wrapped_model.transformer
    transformer.eval()
    
    # FIX: text encoder is on the correct device?
    if hasattr(wrapped_model, 'text_encoder'):
        wrapped_model.text_encoder = wrapped_model.text_encoder.to(device)
    
    total_val_loss = 0.0
    num_val_batches_processed = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            text, image, z = batch["text"], batch["image"].to(device, non_blocking=True), batch["z"]
            prompts = text
            (
                prompt_embeds,
                prompt_embeds_mask,
                uncond_prompt_embeds,
                uncond_prompt_embeds_mask,
            ) = wrapped_model.encode_prompt(text, do_cfg=True)
            prompt_embeds = prompt_embeds.to(torch.float32)
            uncond_prompt_embeds = uncond_prompt_embeds.to(torch.float32)

            latents = wrapped_model.pixels_to_latents(image)
            latents = latents.to(torch.float32)

            with torch_autocast(
                enabled=True,
                dtype=mp_dtype,
                device_type="cuda",
                cache_enabled=False,
            ):
                loss = method.training_step(
                    wrapped_model,
                    latents,
                    c=[prompt_embeds, prompt_embeds_mask],
                    e=[uncond_prompt_embeds, uncond_prompt_embeds_mask],
                    step=global_step,
                    v=z,
                    prompts=prompts,
                )
            
            total_val_loss += loss.item()
            num_val_batches_processed += 1

    total_val_loss_tensor = torch.tensor(total_val_loss, device=device)
    num_batches_tensor = torch.tensor(num_val_batches_processed, device=device)
    
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
    
    avg_val_loss = total_val_loss_tensor.item() / (num_batches_tensor.item() + 1e-8)
    transformer.train()
    dist.barrier()

    return {
        "avg_val_loss": avg_val_loss,
        "batches_processed": num_batches_tensor.item(),
        "total_batches": total_val_batches_global
    }

def main(_):
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    method_type = config["method"].pop("method_type")
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )
    
    if is_main_process(rank):
        print(f"Initialized DeviceMesh: {device_mesh}")

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
    
    hf_uploader = None
    hf_cfg = config["train"].get("hf_hub", {})
    if is_main_process(rank) and hf_cfg.get("use_hf_hub", False):
        hf_uploader = HFCheckpointUploader(
            repo_id=hf_cfg["repo_id"],
            token=hf_cfg["token"],
        )
        logger.info(f"HF checkpoint uploader - {hf_cfg['repo_id']}")

    async_logger = None
    if is_main_process(rank) and config["train"].get("use_wandb", False):
        run_name = config["train"].get("wandb_run_name")
        if run_name is None:
            run_name = f"twinflow_{exp_name}_{config['train']['seed']}"
        
        wandb.init(
            project=config["train"].get("wandb_project", "twinflow-qwen"),
            entity=config["train"].get("wandb_entity"),
            name=run_name,
            tags=config["train"].get("wandb_tags", []) + [""],
            config=config,
            resume="allow",
        )
        
        async_logger = AsyncLogger()
        
        logger.info(f"  Project: {wandb.run.project}")
        logger.info(f"  Run Name: {wandb.run.name}")
        logger.info(f"  Run URL: {wandb.run.url}")

    demoimages_dir = os.path.join(config["train"]["output_dir"], "demoimages")
    if is_main_process(rank):
        os.makedirs(demoimages_dir, exist_ok=True)
    demo_c = [
        "The girl is sitting on steps in front of a building. She is relaxed and bored. She looks up slightly, surveying the scene around her. Her posture is casual. A sense of calmness and loneliness is present despite the hustle from the city lights around her. Wide shot.",
        "Horror-themed (extreme close shot of eyes :1.3) of nordic woman, (war face paint:1.2), mohawk blonde haircut wit thin braids, runes tattoos, sweat, (detailed dirty skin:1.3) shiny, (epic battleground backgroun :1.2), analog, haze, (lens blur :1.3), hard light, sharp focus on eyes, low saturation, by ilya kuvshinov and flora bosil, eerie, unsettling, dark, spooky, suspenseful, grim, highly detailed",
        "a photo of a cow"
    ]

    set_seed(config["train"]["seed"], rank)

    mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    wrapped_model = MODELS[config["model"]["model_name"]](
        model_id=config["model"]["model_path"],
        aux_time_embed=config["model"]["aux_time_embed"],
        text_dtype=mixed_precision_dtype,
        imgs_dtype=mixed_precision_dtype,
    )
    
    wrapped_model.transformer.requires_grad_(True)
    wrapped_model.transformer.enable_gradient_checkpointing()

    no_split_modules = wrapped_model.get_no_split_modules()
    
    if config["train"].get("use_torch_compile", False):
        wrapped_model.transformer = torch.compile(
            wrapped_model.transformer,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False
        )
        if is_main_process(rank):
            logger.info("compile activated")

    if is_main_process(rank):
        logger.info(f"No-split modules: {no_split_modules}")
        logger.info(f"Mixed precision: param_dtype={mp_policy.param_dtype}, reduce_dtype={mp_policy.reduce_dtype}")
    
    sharded_modules = apply_fsdp2_to_model(
        wrapped_model,
        no_split_modules,
        device_mesh,
        mp_policy,
        use_activation_checkpointing=True,
    )
    
    setup_explicit_prefetching(
        wrapped_model.transformer,
        sharded_modules,
        num_forward_prefetch=2,
        num_backward_prefetch=2,
    )
    
    if is_main_process(rank):
        logger.info(f"Applied FSDP2 to {len(sharded_modules)} modules with explicit prefetching")
        logger.info(f"Transformer type after FSDP2: {type(wrapped_model.transformer)}")

    method_config = config["method"].copy()
    # RL gradients use model.latents_to_pixels / pixels_to_latents directly,
    # so no vae_for_rl needed.
    method = METHODES[method_type](**method_config)
    
    if is_main_process(rank):
        if hasattr(method, 'use_rl') and method.use_rl:
            logger.info(f"RL Enabled: {method.use_rl}")
            logger.info(f"RL Warmup Steps: {method.rl_warmup_steps}")
            logger.info(f"RL Weight: {method.rl_weight}")
            logger.info(f"Reward Model: {method.reward_model_type}")
        
        if hasattr(method, 'use_dynamic_renoise') and method.use_dynamic_renoise:
            logger.info(f"Dynamic Renoise: {method.use_dynamic_renoise}")
            logger.info(f"Renoise Schedule: {method.renoise_schedule}")

    transformer_ddp = wrapped_model.transformer
    
    transformer_trainable_parameters = [
        p for p in transformer_ddp.parameters() if p.requires_grad
    ]
    
    transformer_ddp.train()

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config["train"]["lr"],
        betas=config["train"]["betas"],
        weight_decay=config["train"]["weight_decay"],
        fused=True if torch.cuda.is_available() else False,
    )

    full_dataset = BucketedShareGPTDataset(
        metadata_path=config["data"]["bucket_metadata_path"],
        dataset_name=config["data"]["dataset_name"],
        split="train",
        image_dir=config["data"]["image_dir"],
    )
    
    if is_main_process(rank):
        logger.info(f"Total dataset size: {len(full_dataset)}")
    
    bucket_to_indices = {}
    for idx, sample in enumerate(full_dataset.samples):
        bucket_key = sample['bucket_key']
        if bucket_key not in bucket_to_indices:
            bucket_to_indices[bucket_key] = []
        bucket_to_indices[bucket_key].append(idx)
    
    train_bucket_to_indices = {}
    val_bucket_to_indices = {}
    for bucket_key, indices in bucket_to_indices.items():
        rng = random.Random(config["train"]["seed"])
        rng.shuffle(indices)
        
        n_val = max(1, int(0.1 * len(indices))) 
        val_bucket_to_indices[bucket_key] = indices[:n_val]
        train_bucket_to_indices[bucket_key] = indices[n_val:]
    
    all_train_batches = []
    for bucket_key, indices in train_bucket_to_indices.items():
        batch_size = config["train"]["micro_batch_size"]
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if len(batch) == batch_size:  # Only full batches
                all_train_batches.append(batch)
    
    random.Random(config["train"]["seed"]).shuffle(all_train_batches)
    
    all_val_batches = []
    for bucket_key, indices in val_bucket_to_indices.items():
        batch_size = config["train"]["micro_batch_size"]
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if len(batch) == batch_size:  # Only full batches
                all_val_batches.append(batch)
    
    # Distribute batches across ranks.
    # Truncate to a multiple of world_size BEFORE distributing so every rank
    # gets the same number of batches.  An uneven split would cause FSDP2
    # all_gather collectives to desync ranks that finish early exit the loop
    # while others are still mid-forward-pass, causing a 10-min NCCL timeout!
    even_train = (len(all_train_batches) // world_size) * world_size
    even_val   = (len(all_val_batches)   // world_size) * world_size
    local_train_batches = [all_train_batches[i] for i in range(rank, even_train, world_size)]
    local_val_batches   = [all_val_batches[i]   for i in range(rank, even_val,   world_size)]

    total_val_batches_global = even_val
    
    if is_main_process(rank):
        logger.info(f"Data split summary:")
        logger.info(f"Total training samples: {sum(len(indices) for indices in train_bucket_to_indices.values())}")
        logger.info(f"Total validation samples: {sum(len(indices) for indices in val_bucket_to_indices.values())}")
        logger.info(f"Total training batches: {len(all_train_batches)}")
        logger.info(f"Total validation batches: {len(all_val_batches)}")
        logger.info(f"Batches per rank (train): {len(local_train_batches)}")
        logger.info(f"Batches per rank (val): {len(local_val_batches)}")
        logger.info(f"Bucket distribution:")
        for bucket_key, indices in train_bucket_to_indices.items():
            num_train_batches = len(indices) // config["train"]["micro_batch_size"]
            num_val_batches = len(val_bucket_to_indices[bucket_key]) // config["train"]["micro_batch_size"]
            logger.info(f"Bucket {bucket_key}:")
            logger.info(f"Train samples: {len(indices)}, batches: {num_train_batches}")
            logger.info(f"Val samples: {len(val_bucket_to_indices[bucket_key])}, batches: {num_val_batches}")
    
    train_batch_sampler = PrecomputedBatchSampler(local_train_batches, shuffle=True)
    val_batch_sampler = PrecomputedBatchSampler(local_val_batches, shuffle=False)
    
    num_workers = config["train"].get("num_workers", 8)
    prefetch_factor = config["train"].get("prefetch_factor", 4)
    
    train_dataloader = DataLoader(
        full_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=bucketed_collate_fn,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory_device=f"cuda:{local_rank}"
    )
    
    cached_val_batches = []
    if config["train"].get("cache_validation_data", True) and len(local_val_batches) > 0:
        logger.info(f"Caching validation data to GPU memory on rank {rank}")
        for batch in DataLoader(
            full_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=bucketed_collate_fn,
            persistent_workers=True,
            prefetch_factor=2,
        ):
            cached_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    cached_batch[k] = v.to(device, non_blocking=True)
                else:
                    cached_batch[k] = v
            cached_val_batches.append(cached_batch)
        val_dataloader = cached_val_batches
    else:
        val_dataloader = DataLoader(
            full_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=bucketed_collate_fn,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory_device=f"cuda:{local_rank}"
        )
    
    if isinstance(val_dataloader, list):
        logger.info(f"Using cached validation data with {len(val_dataloader)} batches on rank {rank}")
    else:
        logger.info(f"Using standard validation dataloader on rank {rank}")
    
    total_train_batch_size = (
        config["train"]["micro_batch_size"]
        * world_size
        * config["train"]["grad_accumulation_steps"]
    )
    effective_grad_accum_steps = config["train"]["grad_accumulation_steps"]
    total_train_steps = len(local_train_batches)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["train"].get("lr_reduction_factor", 0.1),
        patience=config["train"].get("lr_patience", 3),
        min_lr=config["train"].get("min_lr", 1e-7),
        threshold=config["train"].get("lr_threshold", 1e-4),
    )
    
    if is_main_process(rank):
        logger.info(f"Scheduler type: ReduceLROnPlateau")
        logger.info(f"Mode: min (reduce when validation loss plateaus)")
        logger.info(f"Reduction factor: {scheduler.factor}")
        logger.info(f"Patience: {scheduler.patience} epochs")
        logger.info(f"Min LR: {scheduler.min_lrs[0]}")
        logger.info(f"Threshold: {scheduler.threshold}")

    first_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = config["train"].get("early_stop_patience", 0)
    early_stop = early_stop_patience > 0
    
    if config["train"]["load_checkpoint_path"] != "":
        load_path = config["train"]["load_checkpoint_path"]
        logger.info(f"Resuming from checkpoint: {load_path}")
        # TODO: Implement FSDP2-compatible checkpoint loading with DCP
        # Should load scheduler state and best_val_loss here

    with (
        torch.no_grad(),
        torch_autocast(
            enabled=enable_amp, dtype=mixed_precision_dtype, device_type="cuda"
        ),
    ):
        demox = wrapped_model.sample(
            demo_c,
            sampler=method.sampling_loop,
            height=1024,
            width=1024,
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

    logger.info("***** Starting training *****")
    optimizer.zero_grad()

    validation_results = {}

    for epoch in range(first_epoch, config["train"]["num_train_epochs"]):
        train_iter = iter(train_dataloader)
        optimizer_step = 0
        
        for _ in range(total_train_steps):
            global_step += 1
            is_sync_step = global_step % effective_grad_accum_steps == 0
            
            if not is_sync_step:
                transformer_ddp.set_requires_gradient_sync(False)
            else:
                transformer_ddp.set_requires_gradient_sync(True)

            torch.cuda.synchronize()
            
            batch = next(train_iter)
            text, image, z = batch["text"], batch["image"].cuda(), batch["z"]
            prompts = text

            start_time = time.time()
            if z.numel() != 0:
                z = z.cuda()
            else:
                z = None

            if hasattr(transformer_ddp, 'unshard'):
                transformer_ddp.unshard()

            with torch.no_grad():
                (
                    prompt_embeds,
                    prompt_embeds_mask,
                    uncond_prompt_embeds,
                    uncond_prompt_embeds_mask,
                ) = wrapped_model.encode_prompt(text, do_cfg=True)
                prompt_embeds = prompt_embeds.to(torch.float32)
                uncond_prompt_embeds = uncond_prompt_embeds.to(torch.float32)

                latents = wrapped_model.pixels_to_latents(image)
                latents = latents.to(torch.float32)

            with torch_autocast(
                enabled=enable_amp,
                dtype=mixed_precision_dtype,
                device_type="cuda",
                cache_enabled=False,
            ):
                loss = method.training_step(
                    wrapped_model,
                    latents,
                    c=[prompt_embeds, prompt_embeds_mask],
                    e=[uncond_prompt_embeds, uncond_prompt_embeds_mask],
                    step=(global_step - 1),
                    v=z,
                    prompts=prompts,
                )

            scaled_loss = loss / effective_grad_accum_steps

            # NaN detection: skip this step entirely if loss is NaN/Inf
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                if is_main_process(rank):
                    logger.warning(f"Step {global_step}: NaN/Inf loss detected, skipping update")
                optimizer.zero_grad(set_to_none=True)
                dist.barrier()
                del batch
                continue

            scaled_loss.backward()

            grad_norm = 0.0
            if is_sync_step:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    transformer_ddp.parameters(),
                    config["train"]["max_grad_norm"],
                )
                grad_norm = grad_norm_tensor.full_tensor().item() if hasattr(grad_norm_tensor, 'full_tensor') else float(grad_norm_tensor)

                # Skip optimizer step if gradients contain NaN
                if not (torch.isnan(grad_norm_tensor) or torch.isinf(grad_norm_tensor)):
                    optimizer.step()
                elif is_main_process(rank):
                    logger.warning(f"Step {global_step}: NaN/Inf grad_norm, skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            actual_loss = scaled_loss * effective_grad_accum_steps
            loss_tensor = actual_loss.detach().clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            delta_time = time.time() - start_time

            if is_main_process(rank):
                log_msg = (
                    f"step=({global_step:08d}): "
                    f"loss: {avg_loss:.4f}, "
                    f"grad_norm: {grad_norm:.2f}, "
                    f"step_time: {delta_time:.2f}s"
                )
                rl_active = False 
                renoise_bias = 0.0
                
                if hasattr(method, 'use_rl') and method.use_rl:
                    if global_step >= method.rl_warmup_steps:
                        log_msg += " [RL:ACTIVE]"
                        rl_active = True
                    elif global_step == method.rl_warmup_steps - 1:
                        log_msg += " [RL:WARMUP-ENDING]"
                    else:
                        warm_pct = (global_step / method.rl_warmup_steps) * 100
                        log_msg += f" [RL:WARMUP {warm_pct:.0f}%]"
                
                if hasattr(method, 'use_dynamic_renoise') and method.use_dynamic_renoise:
                    renoise_bias = method.get_dynamic_renoise_bias(global_step)
                    log_msg += f" [Renoise bias: {renoise_bias:.2f}]"
                
                logger.info(log_msg)

                if config["train"].get("use_wandb", False):
                    log_interval = config["train"].get("wandb_log_interval", 10)
                    
                    if global_step % log_interval == 0:
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        
                        wandb_logs = {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/step_time": delta_time,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/bucket": batch["bucket_key"][0] if batch["bucket_key"] else "unknown",
                            "system/gpu_memory_allocated_gb": memory_allocated,
                            "system/gpu_memory_reserved_gb": memory_reserved,
                            "system/epoch": epoch,
                            "system/global_step": global_step,
                        }
                        
                        if hasattr(method, 'use_rl') and method.use_rl:
                            wandb_logs["rl/active"] = int(rl_active)
                            wandb_logs["rl/warmup_progress"] = min(global_step / method.rl_warmup_steps, 1.0)
                        
                        if hasattr(method, 'use_dynamic_renoise') and method.use_dynamic_renoise:
                            wandb_logs["renoise/bias"] = renoise_bias
                        
                        async_logger.log(wandb_logs, step=global_step)
                
                if hasattr(method, 'use_rl') and method.use_rl:
                    if global_step == method.rl_warmup_steps:
                        logger.info("RL ACTIVATED")

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
                        for prompt in demo_c:
                            demo_captions.append(f"{prompt} ({width}x{height})")
                    
                    demox = torch.cat(all_demo_images, dim=0)
                    demoimages_path = f"{demoimages_dir}/{global_step:07d}.png"
                    save_image(
                        (demox + 1) / 2,
                        demoimages_path,
                        nrow=len(demo_c),
                    )
                    logger.info(f"Saved images to {demoimages_path}")
                    
                    if is_main_process(rank) and config["train"].get("use_wandb", False):
                        demo_images_list = []
                        for i, (img, caption) in enumerate(zip(demox, demo_captions)):
                            img_np = ((img + 1) / 2).float().cpu().numpy().transpose(1, 2, 0)
                            img_np = (img_np * 255).astype(np.uint8)
                            demo_images_list.append(wandb.Image(img_np, caption=demo_captions[i]))
                        
                        async_logger.log({
                            "samples": demo_images_list,
                        }, step=global_step)
                    
                    del demox, all_demo_images
                    
            if global_step % 1000 == 0:
                save_checkpoint_sync(
                    config["train"]["save_checkpoint_path"],
                    wrapped_model,
                    optimizer,
                    scheduler,
                    global_step
                )
                if is_main_process(rank):
                    logger.info(f"Saved checkpoint at step {global_step}")
                    if hf_uploader is not None:
                        ckpt_dir = os.path.join(
                            config["train"]["save_checkpoint_path"],
                            f"global_step_{global_step}"
                        )
                        hf_uploader.upload(ckpt_dir, global_step)
            
            dist.barrier()
            del batch
        
        # Distributed collectives (dist.all_reduce) must run on the main thread
        if is_main_process(rank):
            logger.info(f"Running validation for epoch {epoch}")

        validation_results = run_validation(
            val_dataloader,
            wrapped_model,
            method,
            device,
            mixed_precision_dtype,
            global_step,
            rank,
            world_size,
            total_val_batches_global
        )

        avg_val_loss = validation_results.get("avg_val_loss", float('inf'))
        batches_processed = validation_results.get("batches_processed", 0)
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if is_main_process(rank):
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Log validation results
        if is_main_process(rank):
            logger.info(f"Epoch {epoch} validation results:")
            logger.info(f"  Validation loss: {avg_val_loss:.4f}")
            logger.info(f"  Batches processed: {batches_processed} / {total_val_batches_global}")
            
            lr_msg = f"Learning rate updated: {prev_lr:.2e} -> {current_lr:.2e}"
            if prev_lr != current_lr:
                logger.info(f"{lr_msg}")
            else:
                logger.info(f"LR unchanged: {prev_lr:.2e}")
            
            if config["train"].get("use_wandb", False):
                val_logs = {
                    "val/loss": avg_val_loss,
                    "val/batches_processed": batches_processed,
                    "val/total_batches": total_val_batches_global,
                    "val/best_loss": best_val_loss,
                    "val/epochs_without_improvement": epochs_without_improvement,
                    "train/lr": current_lr,
                    "epoch": epoch,
                }
                async_logger.log(val_logs, step=global_step)
        
        # Early stopping check
        if early_stop and epochs_without_improvement >= early_stop_patience:
            if is_main_process(rank):
                logger.info(f"Early stopping triggered after {epochs_without_improvement}")
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
            dist.barrier()
            break
        
        if global_step % 10 != 0:  # Avoid duplicate save if already saved at this step
            save_checkpoint_sync(
                config["train"]["save_checkpoint_path"],
                wrapped_model,
                optimizer,
                scheduler,
                global_step
            )
            if is_main_process(rank):
                logger.info(f"Saved checkpoint after epoch {epoch}")
                if hf_uploader is not None:
                    ckpt_dir = os.path.join(
                        config["train"]["save_checkpoint_path"],
                        f"global_step_{global_step}"
                    )
                    hf_uploader.upload(ckpt_dir, global_step)

    if is_main_process(rank):
        logger.info("Saving final checkpoint...")

    save_ckpt_fsdp2(
        config["train"]["save_checkpoint_path"],
        wrapped_model,
        optimizer,
        scheduler,
        global_step,
        use_dcp=True,
    )

    if is_main_process(rank) and hf_uploader is not None:
        ckpt_dir = os.path.join(
            config["train"]["save_checkpoint_path"],
            f"global_step_{global_step}"
        )
        hf_uploader.upload(ckpt_dir, global_step)
        hf_uploader.shutdown()

    if async_logger is not None:
        async_logger.shutdown()

    if is_main_process(rank):
        wandb.finish()
    
    cleanup_distributed()
    logger.info("Training completed!")

if __name__ == "__main__":
    app.run(main)
