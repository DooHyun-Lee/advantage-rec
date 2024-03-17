import torch
from typing import Dict, Union

def move_batch_to_device(batch: Dict, microbatch_idx: int, gradient_accum_steps:int, device: str) -> Dict:
    chunk_size = len(list(batch.values())[0]) // gradient_accum_steps
    start_idx = chunk_size * microbatch_idx
    end_idx = chunk_size * (microbatch_idx + 1)
    sliced = {k: v[start_idx:end_idx] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device

def formatted_dict(d: Dict) -> Dict:
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}