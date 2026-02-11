import torch
import torch.nn as nn
from tqdm import tqdm
import functools
import logging
from models.ptq import QLayerNorm

@torch.no_grad()
def get_scale(fcs, act_scales, alpha=0.5):
    """
    fcs: 一个或一组 Linear（它们的输入维度相同）
    act_scales: 对应输入激活的 per-channel max_abs，shape = (C_in,)
    return: per-channel scale, shape = (C_in,)
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype

    act_scales = act_scales.to(device=device, dtype=dtype).clamp(min=1e-5)

    # 统计这些 Linear 的 weight per-channel 最大值
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs],
        dim=0
    )  # (len(fcs), C_in)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)  # (C_in,)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    return scales


@torch.no_grad()
def smooth_ln_fcs(ln: nn.LayerNorm, fcs, scales: torch.Tensor):
    """
    将 scale 从激活侧“搬一部分”到 LN gamma/bias 和后面的 Linear weight 上：
      ln: 这一层的 LayerNorm（对最后一维）
      fcs: 一个或一组以 LN 输出为输入的 Linear
      scales: per-channel 缩放 (C,)
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scales = scales.to(device).to(dtype)

    # LN 的 gamma / bias 除以 scale
    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    # 后面的 Linear weight 乘以 scale
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    # 简单 NaN 检查
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    """
    把 scale 从 fc1 output 侧往 fc2 input 侧搬：
      fc1: 上一层 Linear
      fc2: 下一层 Linear
      scales: 对 fc2 输入维度（也就是 fc1 输出维度）的 per-channel scale (C_mid,)
    """
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    device, dtype = fc2.weight.device, fc2.weight.dtype
    scales = scales.to(device).to(dtype)
    # fc1 对输出做除 scale
    fc1.weight.div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # fc2 对输入做乘 scale
    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def smooth_wm_predictor(
    world_model,
    act_scales: dict,
    alpha: float = 0.5,
    logger=print,
    use_tqdm: bool = True,
):
    predictor = getattr(world_model, "predictor", None)
    if predictor is None:
        logger("[smooth_wm_predictor] no predictor, skip")
        return

    layers = predictor.transformer.layers
    module_to_name = {m: n for n, m in world_model.named_modules()}

    LN_TYPES = (nn.LayerNorm, QLayerNorm)

    layer_iter = enumerate(layers)
    if use_tqdm:
        layer_iter = tqdm(layer_iter, total=len(layers), desc="Smooth vit_q predictor")

    for lid, layer in layer_iter:
        if not hasattr(layer, "attn") or not hasattr(layer, "ff"):
            logger(f"[smooth_wm_predictor] layer {lid}: not QTransformerBlock, skip")
            continue

        attn = layer.attn
        ff   = layer.ff

        # ===== Attention =====
        attn_ln = getattr(attn, "norm", None)
        qkv_fc  = getattr(attn, "to_qkv", None)

        if isinstance(attn_ln, LN_TYPES) and isinstance(qkv_fc, nn.Linear):
            qkv_name = module_to_name.get(qkv_fc)
            if qkv_name in act_scales:
                sm_scales = get_scale(qkv_fc, act_scales[qkv_name], alpha)
                smooth_ln_fcs(attn_ln, qkv_fc, sm_scales)
                logger(f"[smooth] layer {lid} attn {qkv_name}")

        # ===== FFN =====
        if hasattr(ff, "net") and len(ff.net) >= 5:
            ff_ln = ff.net[0]
            fc1   = ff.net[1]

            if isinstance(ff_ln, LN_TYPES) and isinstance(fc1, nn.Linear):
                fc1_name = module_to_name.get(fc1)
                if fc1_name in act_scales:
                    sm_scales = get_scale(fc1, act_scales[fc1_name], alpha)
                    smooth_ln_fcs(ff_ln, fc1, sm_scales)
                    logger(f"[smooth] layer {lid} ffn {fc1_name}")


def get_act_scales_wm(model, name_filter=None):
  
    act_scales = {}   
    hooks = []

    def stat_tensor(name: str, x: torch.Tensor):
        if name_filter is not None and not name_filter(name):
            return

        if x.numel() == 0:
            return

        hidden_dim = x.shape[-1]
        x = x.view(-1, hidden_dim).abs().detach()  # (N, C)
        coming_max = torch.max(x, dim=0)[0].float().cpu()  # (C,)

        if torch.any(torch.isnan(coming_max)):
            print(f"[get_act_scales_wm] NaN found in {name}, skip this batch")
            return

        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], coming_max)
        else:
            act_scales[name] = coming_max

    def stat_input_hook(module, inputs, output, name: str):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(x, torch.Tensor):
            stat_tensor(name, x)

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            h = m.register_forward_hook(
                functools.partial(stat_input_hook, name=name)
            )
            hooks.append(h)

    return act_scales, hooks
@torch.no_grad()
def smooth_wm_encoder(world_model, act_scales: dict, alpha: float = 0.5, logger=print, use_tqdm: bool = True):
    encoder = getattr(world_model, "encoder", None)
    if encoder is None:
        logger("[smooth_wm_encoder] world_model no encoder")
        return

    vit = getattr(encoder, "base_model", None)
    if vit is None or not hasattr(vit, "blocks"):
        logger("[smooth_wm_encoder] encoder.base_model.blocks are not exist")
        return

    blocks = vit.blocks 
    module_to_name = {m: n for n, m in world_model.named_modules()}

    block_iter = enumerate(blocks)
    if use_tqdm:
        block_iter = tqdm(block_iter, total=len(blocks), desc="Smooth encoder blocks")

    for bid, block in block_iter:
        ln1 = getattr(block, "norm1", None)
        attn = getattr(block, "attn", None)
        qkv = getattr(attn, "qkv", None) if attn is not None else None

        if isinstance(ln1, nn.LayerNorm) and isinstance(qkv, nn.Linear):
            qkv_name = module_to_name.get(qkv, None)
            if qkv_name is not None and qkv_name in act_scales:
                inp_scales = act_scales[qkv_name]  
                sm_scales = get_scale(qkv, inp_scales, alpha)
                smooth_ln_fcs(ln1, qkv, sm_scales)
                logger(f"[smooth_wm_encoder] block {bid} attn: smooth {qkv_name}")
            else:
                logger(f"[smooth_wm_encoder] block {bid} attn: act_scales[{qkv_name}] are not exist")
        else:
            logger(f"[smooth_wm_encoder] block {bid} attn: norm1 或 attn.qkv are not exist")

        ln2 = getattr(block, "norm2", None)
        mlp = getattr(block, "mlp", None)

        fc1 = getattr(mlp, "fc1", None) if mlp is not None else None
        fc2 = getattr(mlp, "fc2", None) if mlp is not None else None

        if isinstance(ln2, nn.LayerNorm) and isinstance(fc1, nn.Linear):
            fc1_name = module_to_name.get(fc1, None)
            if fc1_name is not None and fc1_name in act_scales:
                fc1_inp_scales = act_scales[fc1_name]  # (C_in,)
                sm_scales = get_scale(fc1, fc1_inp_scales, alpha)
                smooth_ln_fcs(ln2, fc1, sm_scales)
                logger(f"[smooth_wm_encoder] block {bid} ffn: smooth {fc1_name}")
            else:
                logger(f"[smooth_wm_encoder] block {bid} ffn: act_scales[{fc1_name}] 不存在，跳过 LN2+fc1")
        else:
            logger(f"[smooth_wm_encoder] block {bid} ffn: 未找到 norm2 或 mlp.fc1，跳过 LN2+fc1")



@torch.no_grad()
def get_act_per_channel_scales(model, dataloader, num_samples=128):

    logging.info("Start get_act_per_channel_scales, num_samples: {}".format(num_samples))
    model.eval()
    device = next(model.parameters()).device
    act_per_channel_max = {}  
    act_per_channel_min = {}  


    def stat_tensor(name, tensor):

        target_keywords = {"q_proj", "o_proj", "up_proj", "down_proj", 
                           "mlp.gate", "block_sparse_moe.gate", "w2", "w3"}
        if any(kw in name for kw in target_keywords):
            if tensor.numel() == 0:
                logging.warning(f"Empty tensor for layer {name}, skip stat")
                return

            tensor = tensor.detach().float().to(device) 
            hidden_dim = tensor.shape[-1]
            tensor_reshaped = tensor.reshape(-1, hidden_dim)

            curr_max = torch.max(tensor_reshaped, dim=0)[0].cpu() 
            if name in act_per_channel_max:
                act_per_channel_max[name] = act_per_channel_max[name] * 0.9 + curr_max * 0.1
            else:
                act_per_channel_max[name] = curr_max
            

            curr_min = torch.min(tensor_reshaped, dim=0)[0].cpu()
            if name in act_per_channel_min:
                act_per_channel_min[name] = act_per_channel_min[name] * 0.9 + curr_min * 0.1
            else:
                act_per_channel_min[name] = curr_min


    def stat_input_hook(m, x, y, name):

        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            logging.warning(f"Non-tensor input for layer {name}, skip stat")
            return
        stat_tensor(name, x)


    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hook = m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            hooks.append(hook)
            logging.debug(f"Registered hook for Linear layer: {name}")

    try:
        for i in tqdm(range(num_samples), desc="Stat act scales"):
            try:
                batch_input = dataloader[i][0].to(device, non_blocking=True)
                model(batch_input)
            except Exception as e:
                logging.error(f"Failed to process sample {i}: {str(e)}")
                continue
    finally:
        for h in hooks:
            h.remove()
        logging.debug("All hooks removed")

    for name in act_per_channel_max:
        act_per_channel_max[name] = act_per_channel_max[name].clamp(min=1e-5)
        act_per_channel_min[name] = act_per_channel_min[name].clamp(min=1e-5)
        logging.info(f"Layer {name}: max scale range [{act_per_channel_max[name].min():.4f}, {act_per_channel_max[name].max():.4f}]")
        logging.info(f"Layer {name}: min scale range [{act_per_channel_min[name].min():.4f}, {act_per_channel_min[name].max():.4f}]")

    logging.info("Finish get_act_per_channel_scales")
    return act_per_channel_max, act_per_channel_min