from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from .prompts import build_prompts
from .utils import ensure_dir, get_device, seed_everything


def generate_candidates(
    class_map_csv: str | Path,
    output_dir: str | Path,
    metadata_csv: str | Path,
    model_id: str,
    prompt_templates: list[str],
    negative_prompt: str,
    num_candidates_per_class: int,
    image_size: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 0,
    dtype: str = "float16",
    resume: bool = True,
    allow_untracked_existing: bool = False,
    lora_path: str | Path | None = None,
    lora_scale: float = 1.0,
    lora_adapter_name: str = "domain_lora",
) -> None:
    device = get_device()
    torch_dtype = torch.float16 if dtype == "float16" and device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to(device)
    if lora_path:
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        if lora_path.is_file():
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name=lora_adapter_name)
        else:
            pipe.load_lora_weights(str(lora_path), adapter_name=lora_adapter_name)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([lora_adapter_name], adapter_weights=[float(lora_scale)])
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    class_map = pd.read_csv(class_map_csv)
    output_dir = ensure_dir(output_dir)
    metadata_csv = Path(metadata_csv)
    previous_records = _load_previous_records(metadata_csv) if resume else {}
    records = []
    fingerprint = _config_fingerprint(
        model_id,
        prompt_templates,
        negative_prompt,
        image_size,
        num_inference_steps,
        guidance_scale,
        seed,
        dtype,
        str(lora_path) if lora_path else "",
        lora_scale,
    )
    generator = torch.Generator(device=device).manual_seed(seed) if device.type == "cuda" else torch.Generator().manual_seed(seed)

    for _, row in tqdm(class_map.iterrows(), total=len(class_map), desc="Generating classes"):
        class_name = row["class_name"]
        label = int(row["label"])
        class_dir = ensure_dir(output_dir / f"label_{label:03d}_{_slug(class_name)}")
        prompts = build_prompts(class_name, prompt_templates)
        for i in range(num_candidates_per_class):
            prompt = prompts[i % len(prompts)]
            image_path = class_dir / f"{i:04d}.png"
            if resume and image_path.exists():
                old_record = previous_records.get(str(image_path.resolve()))
                if old_record is None and not allow_untracked_existing:
                    raise FileExistsError(
                        f"Existing image has no metadata record: {image_path}. "
                        "Use --no-resume to regenerate or remove stale files."
                    )
                if old_record is not None:
                    old_record = dict(old_record)
                    old_record["reused_existing"] = True
                    records.append(old_record)
                    continue
                records.append(
                    _record(
                        image_path,
                        class_name,
                        label,
                        prompt,
                        seed,
                        model_id,
                        guidance_scale,
                        num_inference_steps,
                        fingerprint,
                        reused_existing=True,
                        lora_path=str(lora_path) if lora_path else "",
                        lora_scale=lora_scale,
                    )
                )
                continue
            seed_everything(seed + label * 100000 + i)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=image_size,
                height=image_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            image.save(image_path)
            records.append(
                _record(
                    image_path,
                    class_name,
                    label,
                    prompt,
                    seed,
                    model_id,
                    guidance_scale,
                    num_inference_steps,
                    fingerprint,
                    reused_existing=False,
                    lora_path=str(lora_path) if lora_path else "",
                    lora_scale=lora_scale,
                )
            )

    ensure_dir(metadata_csv.parent)
    pd.DataFrame(records).to_csv(metadata_csv, index=False)


def _record(
    image_path: Path,
    class_name: str,
    label: int,
    prompt: str,
    seed: int,
    model_id: str,
    guidance: float,
    steps: int,
    config_fingerprint: str,
    reused_existing: bool,
    lora_path: str = "",
    lora_scale: float = 1.0,
) -> dict:
    return {
        "image_path": str(image_path.resolve()),
        "class_name": class_name,
        "label": label,
        "prompt": prompt,
        "seed": seed,
        "model": model_id,
        "lora_path": lora_path,
        "lora_scale": lora_scale,
        "guidance_scale": guidance,
        "num_steps": steps,
        "config_fingerprint": config_fingerprint,
        "reused_existing": reused_existing,
    }


def _load_previous_records(metadata_csv: Path) -> dict[str, dict]:
    if not metadata_csv.exists():
        return {}
    df = pd.read_csv(metadata_csv)
    if "image_path" not in df.columns:
        return {}
    return {str(Path(row["image_path"]).resolve()): row.to_dict() for _, row in df.iterrows()}


def _config_fingerprint(
    model_id: str,
    prompt_templates: list[str],
    negative_prompt: str,
    image_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    dtype: str,
    lora_path: str = "",
    lora_scale: float = 1.0,
) -> str:
    text = "|".join(
        [
            model_id,
            repr(prompt_templates),
            negative_prompt,
            str(image_size),
            str(num_inference_steps),
            str(guidance_scale),
            str(seed),
            dtype,
            lora_path,
            str(lora_scale),
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")
