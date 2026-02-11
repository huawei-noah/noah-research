import gc
import torch
import torch.nn as nn


from .qmodule import ScaledActivation
from .module import get_op_by_name, get_op_name, set_op_by_name

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0



@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor, *, tail_only: bool = True):
 
    assert isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)
    assert scales.dim() == 1, f"scales must be 1D, got {scales.shape}"

    device = fc1.weight.device
    dtype = fc1.weight.dtype
    s = scales.to(device=device, dtype=dtype)

    k = s.numel()
    out1 = fc1.weight.size(0)      
    in2  = fc2.weight.size(1)    

    if tail_only:
        assert out1 >= k and in2 >= k, f"tail_only requires out1,in2 >= k. got out1={out1}, in2={in2}, k={k}"
        idx1 = slice(out1 - k, out1)  
        idx2 = slice(in2 - k, in2)     
    else:

        assert out1 == k and in2 == k, f"full scaling requires out1==in2==k. got out1={out1}, in2={in2}, k={k}"
        idx1 = slice(None)
        idx2 = slice(None)


    fc1.weight.data[idx1, :].div_(s.view(k, 1))


    if fc1.bias is not None:
        fc1.bias.data[idx1].div_(s)


    fc2.weight.data[:, idx2].mul_(s.view(1, k))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, ))
    from models.ptq.layers import QLinear 
    assert isinstance(fc, (nn.Linear, QLinear))

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat):
    from .quantizer import pseudo_quantize_tensor

    if w_bit is not None:

        if isinstance(q_config, dict):
            cfg_kwargs = q_config
        elif q_config is None:
            cfg_kwargs = {}
        else:
            cfg_kwargs = {}

        def w_quantize_func(p):
            return pseudo_quantize_tensor(
                p,
                n_bit=w_bit,
                **cfg_kwargs,
            ).detach()

    else:

        def w_quantize_func(p):
            return p


    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)
        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):

        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []  #
  
    # =====================NestedTensorBlock  =====================
    if module.__class__.__name__ == "NestedTensorBlock":


        # scales_list.append(
        #     _auto_get_scale(
        #         prev_op=module.norm1,
        #         layers=[module.attn.qkv],
        #         inp=input_feat["attn.qkv"],
        #         module2inspect=module.attn,
        #         kwargs=module_kwargs,
        #     )
        # )

        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm2,
                layers=[module.mlp.fc1],
                inp=input_feat["mlp.fc1"],
                module2inspect=module.mlp,
            )
        )

        # scales_list.append(
        #     _auto_get_scale(
        #         prev_op=module.mlp.act,
        #         layers=[module.mlp.fc2],
        #         inp=input_feat["mlp.fc2"],
        #     )
        # )
    # ===================== Predictor QTransformerBlock）=====================
    elif module.__class__.__name__ == "QTransformerBlock":

        # scales_list.append(
        #     _auto_get_scale(
        #         prev_op=module.attn.norm,
        #         layers=[module.attn.to_qkv],
        #         inp=input_feat["attn.to_qkv"],
        #         module2inspect=module.attn,
        #         kwargs={},  
        #     )
        # )


        scales_list.append(
            _auto_get_scale(
                prev_op=module.ff.net[0],
                layers=[module.ff.net[1]],
                inp=input_feat["ff.net.1"],
                module2inspect=module.ff,
                kwargs={},
            )
        )


        # scales_list.append(
        #     _auto_get_scale(
        #         prev_op=module.ff.net[2],       # GELU
        #         layers=[module.ff.net[4]],      # fc2
        #         inp=input_feat["ff.net.4"],
        #         # module2inspect=None 
        #         kwargs={},
        #     )
        # )




    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]


        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm)):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, nn.SiLU)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))
