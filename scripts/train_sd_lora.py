from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import load_config, resolve_path
from ccds.utils import ensure_dir, seed_everything


DEFAULT_DATA_ROOT = Path("/root/gpufree-data/clip_diffusion_fewshot_ccds_lora")


class LoraManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, tokenizer, resolution: int):
        self.df = pd.read_csv(manifest_csv)
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.transform(image)
        tokenized = self.tokenizer(
            str(row["caption"]),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small SD1.5 UNet LoRA on a seed-specific train-only manifest.")
    parser.add_argument("--config", required=True, help="Path to YAML config with generation/lora sections.")
    parser.add_argument("--seed", type=int, required=True, help="Few-shot split seed. Used to format lora paths.")
    parser.add_argument("--manifest", default=None, help="Override train manifest CSV.")
    parser.add_argument("--output-dir", default=None, help="Override LoRA output directory.")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", default=None, choices=["no", "fp16", "bf16"], help="Override LoRA mixed precision.")
    args = parser.parse_args()

    try:
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires peft. Install dependencies with `pip install -r requirements.txt` "
            "or `pip install peft`."
        ) from exc

    cfg = load_config(args.config)
    gen_cfg = cfg["generation"]
    lora_cfg = cfg.get("lora", {})
    seed_everything(int(args.seed))

    output_dir = _format_output_dir(cfg, lora_cfg, args.seed, args.output_dir)
    ensure_dir(output_dir)
    manifest_csv = Path(args.manifest) if args.manifest else output_dir / "train_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"LoRA manifest not found: {manifest_csv}. Run scripts/prepare_lora_dataset.py first."
        )

    model_id = str(gen_cfg["model_id"])
    mixed_precision = str(args.mixed_precision or lora_cfg.get("mixed_precision", "fp16"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mixed_precision == "fp16" and device.type == "cuda":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16" and device.type == "cuda":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=weight_dtype, safety_checker=None)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device, dtype=weight_dtype)
    vae = pipe.vae.to(device, dtype=weight_dtype)
    unet = pipe.unet.to(device, dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if bool(lora_cfg.get("gradient_checkpointing", True)) and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    rank = int(args.rank or lora_cfg.get("rank", 8))
    adapter_name = str(lora_cfg.get("adapter_name", "domain_lora"))
    lora_alpha = int(lora_cfg.get("lora_alpha", rank))
    target_modules = lora_cfg.get("target_modules") or ["to_q", "to_k", "to_v", "to_out.0"]
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights=str(lora_cfg.get("init_lora_weights", "gaussian")),
            target_modules=target_modules,
        ),
        adapter_name=adapter_name,
    )

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found on UNet.")

    resolution = int(args.resolution or lora_cfg.get("resolution", 512))
    batch_size = int(args.train_batch_size or lora_cfg.get("train_batch_size", 1))
    grad_accum = int(args.gradient_accumulation_steps or lora_cfg.get("gradient_accumulation_steps", 4))
    max_steps = int(args.max_train_steps or lora_cfg.get("max_train_steps", 800))
    lr = float(args.learning_rate or lora_cfg.get("learning_rate", 1e-4))

    dataset = LoraManifestDataset(manifest_csv, tokenizer, resolution)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(lora_cfg.get("num_workers", 0)))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=float(lora_cfg.get("weight_decay", 0.0)))

    unet.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=max_steps, desc="train sd lora")
    while global_step < max_steps:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean") / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite LoRA loss at step {global_step}: {loss.item()}. "
                    "Try --mixed-precision bf16 or no, lower --learning-rate, or lower --rank."
                )
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(lora_cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
            if global_step >= max_steps:
                break
    progress.close()

    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name=adapter_name))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
        weight_name=str(lora_cfg.get("weight_name", "pytorch_lora_weights.safetensors")),
    )
    print(f"Wrote LoRA weights to {output_dir}")


def _format_output_dir(cfg: dict, lora_cfg: dict, seed: int, override: str | None) -> Path:
    if override:
        return resolve_path(str(override).format(seed=seed))
    configured = lora_cfg.get("output_dir")
    if configured:
        return resolve_path(str(configured).format(seed=seed))
    return DEFAULT_DATA_ROOT / str(cfg.get("project_name", "default_experiment")) / "lora" / f"seed{seed}"


if __name__ == "__main__":
    main()
