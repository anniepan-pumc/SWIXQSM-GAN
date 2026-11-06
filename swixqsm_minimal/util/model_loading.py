"""Architecture-aware loading, with explicit legacy UNetSmall compatibility."""
import hashlib
import io
from pathlib import Path

import torch

from ..model.generator import UNetSmall
from ..model.training_models import MODEL_NAMES, create_model


def load_checkpoint_snapshot(path):
    """Load and hash the same bytes even if live best.pt is atomically replaced."""
    payload = Path(path).read_bytes()
    return (torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True),
            hashlib.sha256(payload).hexdigest())


def generator_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    top = checkpoint.get("model_spec")
    configured = config.get("model_spec")
    if top is not None and configured is not None and top != configured:
        raise ValueError("Checkpoint and configuration model specifications conflict")
    spec = top if top is not None else configured
    if spec is None:
        if checkpoint.get("schema") == "swixqsm.comparison_checkpoint" or config.get("model_name") is not None:
            raise ValueError("A declared comparison checkpoint cannot omit its model specification")
        generator = UNetSmall(normalize=False)
    else:
        if (not isinstance(spec, dict) or spec.get("schema") != "swixqsm.prospective_training_model"
                or spec.get("version") != 1 or spec.get("model_name") not in MODEL_NAMES
                or config.get("model_name") != spec["model_name"]):
            raise ValueError("Unsupported or inconsistent comparison model specification")
        generator, _, expected = create_model(spec["model_name"], image_size=config["size"])
        for key in ("image_size", "generator", "discriminator", "parameter_counts", "input", "input_padding"):
            if spec.get(key) != expected[key]:
                raise ValueError("Declared model architecture differs from implementation: " + key)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    return generator
