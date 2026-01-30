import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def mask_features(
    features_batch: Dict[str, torch.Tensor],
    maskable_features: List[str],
    mask_ratio: float = 0.2,
    mask_token_id_map: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[Dict[str, torch.Tensor], torch.BoolTensor]:

    if not 0 <= mask_ratio <= 1:
        raise ValueError

    if not maskable_features:
        batch_size, seq_len = next(iter(features_batch.values())).shape[:2]
        device = next(iter(features_batch.values())).device
        masked_positions = torch.zeros((batch_size, seq_len),
                                       dtype=torch.bool,
                                       device=device)
        masked_batch = {
            name: values.clone() for name, values in features_batch.items()
            }
        return masked_batch, masked_positions

    batch_size, seq_len = next(iter(features_batch.values())).shape[:2]
    device = next(iter(features_batch.values())).device

    num_mask_per_seq = int(seq_len * mask_ratio)
    if num_mask_per_seq == 0 and mask_ratio > 0:
        num_mask_per_seq = 1

    if num_mask_per_seq > 0:
        random_indices = torch.argsort(torch.rand(batch_size,
                                                  seq_len,
                                                  device=device),
                                       dim=1)[:, :num_mask_per_seq]
        masked_positions = torch.zeros((batch_size, seq_len),
                                       dtype=torch.bool,
                                       device=device)
        row_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        masked_positions[row_indices, random_indices] = True
    else:
        masked_positions = torch.zeros((batch_size, seq_len),
                                       dtype=torch.bool,
                                       device=device)

    masked_batch = {}
    mask_token_id_map = mask_token_id_map or {}

    for name, values in features_batch.items():
        if name in maskable_features:
            mask_val = mask_token_id_map.get(name, 0)
            masked_values = values.clone()

            mask_shape = [1] * len(values.shape)
            mask_shape[0] = batch_size
            mask_shape[1] = seq_len
            expanded_mask = masked_positions.view(mask_shape).expand_as(values)

            if isinstance(mask_val, float) and np.isnan(mask_val):
                fill_val = torch.tensor(float('nan'),
                                        device=values.device,
                                        dtype=values.dtype)
            else:
                fill_val = torch.tensor(mask_val,
                                        device=values.device,
                                        dtype=values.dtype)

            masked_values = torch.where(expanded_mask, fill_val, masked_values)
            masked_batch[name] = masked_values
        else:
            masked_batch[name] = values.clone()

    return masked_batch, masked_positions
