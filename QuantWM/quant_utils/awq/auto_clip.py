import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc
import os
__all__ = ["auto_clip_block"]



@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape



    if isinstance(q_config, dict):

        group_size = q_config.get("q_group_size", int(os.environ['W_GROUP_SIZE']) if 'W_GROUP_SIZE' in os.environ else -1)
        clip_qcfg = q_config
    elif q_config is None:

        group_size = -1
        clip_qcfg = {}
    else:

        group_size = -1
        clip_qcfg = {}


    group_size = group_size if group_size > 0 else w.shape[1]
    if w.shape[1] % group_size != 0:
        group_size=w.shape[1]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: max(1, input_feat.shape[1] // n_sample_token)]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    import math


    oc_batch_size = 256 if w.shape[0] >= 256 else 64
    w_all = w
    best_max_val_all = []

    num_chunks = math.ceil(w_all.shape[0] / oc_batch_size)
    for i_b in range(num_chunks):
        start = i_b * oc_batch_size
        end = min(start + oc_batch_size, w_all.shape[0])
        w = w_all[start:end] 

        org_max_val = w.abs().amax(dim=-1, keepdim=True) 

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1) 

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)

            org_shape = cur_w.shape
            cur_w_2d = cur_w.reshape(-1, org_shape[-1])
            q_w_2d = pseudo_quantize_tensor(cur_w_2d, n_bit=n_bit, **clip_qcfg)
            q_w = q_w_2d.view(org_shape)

            cur_out = (input_feat * q_w).sum(dim=-1)

            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]

        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)


    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name, linear in named_linears.items():
        if any(_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]):
            continue
        max_val = auto_clip_layer(
            linear.weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        clip_list.append((name, max_val))

    return clip_list



@torch.no_grad()
def apply_clip(module, clip_list):
    from .module import get_op_by_name

    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)

        max_val = max_val.to(layer.weight.device).to(layer.weight.dtype)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)