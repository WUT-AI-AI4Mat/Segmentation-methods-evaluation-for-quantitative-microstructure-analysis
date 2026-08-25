import torch


def load_incremental_checkpoint(model, checkpoint_path, map_location="cpu"):
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint_path}")

    model_state = model.state_dict()
    unexpected = [key for key in state_dict if key not in model_state]
    mismatched = [
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    if unexpected or mismatched:
        details = []
        if unexpected:
            details.append(f"unexpected keys: {unexpected[:5]}")
        if mismatched:
            details.append(f"shape mismatches: {mismatched[:5]}")
        raise RuntimeError("Checkpoint is incompatible; " + "; ".join(details))

    model.load_state_dict(state_dict, strict=False)
    if not any("lora_" in key for key in state_dict):
        raise RuntimeError("Checkpoint does not contain LoRA parameters.")
    if not any("mask_decoder" in key for key in state_dict):
        raise RuntimeError("Checkpoint does not contain mask-decoder parameters.")
    return len(state_dict)
