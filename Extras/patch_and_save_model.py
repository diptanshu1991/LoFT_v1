import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Step 1: Patch attention layers to avoid using inplace bitwise ops
class PatchedAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn

    def forward(self, *args, **kwargs):
        output = self.original_attn(*args, **kwargs)

        if isinstance(output, tuple):
            result = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    item = item | torch.zeros_like(item)  # force non-inplace
                result.append(item)
            return tuple(result)
        elif isinstance(output, torch.Tensor):
            return output | torch.zeros_like(output)
        return output

# Step 2: Replace attention modules in the model
def patch_model(model):
    for name, module in model.named_modules():
        if "attn" in name.lower() and isinstance(module, nn.Module):
            parts = name.split('.')
            obj = model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], PatchedAttention(module))
    return model

# Step 3: Load original, patch, and save
def patch_and_save(model_dir: str, patched_dir: str):
    os.makedirs(patched_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    patched_model = patch_model(model)
    patched_model.save_pretrained(patched_dir)
    print(f"✅ Patched model saved to: {patched_dir}")

# Example usage
if __name__ == "__main__":
    original_model_path = "loft-cli/models/finetuned_model/final_model"
    patched_model_path = "loft-cli/models/patched_model"
    patch_and_save(original_model_path, patched_model_path)
