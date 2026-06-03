from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import generation_metadata_path, load_config, resolve_path
from ccds.datasets import class_map_path
from ccds.diffusion import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diffusion candidate images.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Override candidates per class for quick tests.")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate even when image files already exist.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    gen = cfg["generation"]
    class_map_csv = resolve_path(class_map_path(dataset))
    metadata_csv = Path(str(generation_metadata_path(cfg)).format(seed=args.seed))
    output_dir = resolve_path(str(gen["output_dir"]).format(seed=args.seed))
    num_candidates = args.limit_per_class or int(gen["num_candidates_per_class"])
    lora_path = gen.get("lora_path")
    lora_path = resolve_path(str(lora_path).format(seed=args.seed)) if lora_path else None

    generate_candidates(
        class_map_csv=class_map_csv,
        output_dir=output_dir,
        metadata_csv=metadata_csv,
        model_id=gen["model_id"],
        prompt_templates=gen["prompts"],
        negative_prompt=gen["negative_prompt"],
        num_candidates_per_class=num_candidates,
        image_size=int(gen["image_size"]),
        num_inference_steps=int(gen["num_inference_steps"]),
        guidance_scale=float(gen["guidance_scale"]),
        seed=args.seed,
        dtype=gen.get("dtype", "float16"),
        resume=not args.no_resume,
        lora_path=lora_path,
        lora_scale=float(gen.get("lora_scale", 1.0)),
        lora_adapter_name=str(gen.get("lora_adapter_name", "domain_lora")),
    )
    print(f"Wrote metadata to {metadata_csv}")


if __name__ == "__main__":
    main()
