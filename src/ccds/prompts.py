from __future__ import annotations


def build_prompts(class_name: str, templates: list[str]) -> list[str]:
    """Format prompt templates for one class name."""
    return [template.format(class_name=class_name) for template in templates]
