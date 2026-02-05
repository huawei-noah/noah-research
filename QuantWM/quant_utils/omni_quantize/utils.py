from collections import OrderedDict
from .int_linear import QuantLinear
import torch
import torch.nn as nn
from .int_matmul import QuantMatMul

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)

def smooth_ln_fcs_temporary(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts
        else:
            fc.temp_bias = fc.weight@shifts
        fc.temp_weight = fc.weight * scales.view(1,-1)


def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = fc1.temp_bias - shifts
        fc1.temp_bias = fc1.temp_bias/scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/scales.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/scales.view(-1)
        fc1.temp_weight = fc1.weight/scales.view(-1,1)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@shifts
    else:
        fc2.temp_bias = fc2.weight@shifts
    fc2.temp_weight = fc2.weight * scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)
    k_proj.temp_weight = k_proj.temp_weight*scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*scales.view(-1)



def smooth_ln_fcs_inplace(ln, fcs, scales, shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]

    # --- Align dtype/device for LN path ---
    # Prefer LN weight dtype (LN parameters are real Parameters)
    ln_dtype = ln.weight.dtype
    ln_device = ln.weight.device
    scales_ln = scales.to(device=ln_device, dtype=ln_dtype)
    shifts_ln = shifts.to(device=ln_device, dtype=ln_dtype)

    # LN: (bias - shift) / scale ; weight / scale
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts_ln)
        ln.bias.div_(scales_ln)
    else:
        # register bias buffer with correct dtype/device
        if hasattr(ln, "bias"):
            del ln.bias
        ln.register_buffer('bias', (-1 * shifts_ln) / scales_ln)

    ln.weight.div_(scales_ln)

    # --- FC path ---
    for fc in fcs:
        fc.use_temporary_parameter = False

        # Align to FC weight dtype/device (QuantLinear weight may be buffer)
        w = fc.weight
        fc_dtype = w.dtype
        fc_device = w.device

        scales_fc = scales.to(device=fc_device, dtype=fc_dtype)
        shifts_fc = shifts.to(device=fc_device, dtype=fc_dtype)

        # bias += W @ shift
        delta_b = w @ shifts_fc

        if hasattr(fc, 'bias') and fc.bias is not None:
            # IMPORTANT: ensure fc.bias dtype matches delta_b
            if fc.bias.dtype != delta_b.dtype:
                # if bias is Parameter/buffer, safest is cast delta to bias dtype
                delta_b = delta_b.to(dtype=fc.bias.dtype)
            fc.bias.add_(delta_b)
        else:
            if hasattr(fc, "bias"):
                del fc.bias
            fc.register_buffer('bias', delta_b)

        # weight *= scale
        fc.weight.mul_(scales_fc.view(1, -1))


def smooth_fc_fc_inplace(fc1, fc2, scales, shifts=None):
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False

    # align dtype/device to fc1 weight
    w1 = fc1.weight
    scales1 = scales.to(device=w1.device, dtype=w1.dtype)
    shifts1 = shifts.to(device=w1.device, dtype=w1.dtype)

    fc1.bias.sub_(shifts1.to(dtype=fc1.bias.dtype))
    fc1.bias.div_(scales1.view(-1).to(dtype=fc1.bias.dtype))
    fc1.weight.div_(scales1.view(-1, 1))

    # align dtype/device to fc2 weight
    w2 = fc2.weight
    shifts2 = shifts.to(device=w2.device, dtype=w2.dtype)
    delta_b = w2 @ shifts2

    if hasattr(fc2, 'bias') and fc2.bias is not None:
        if fc2.bias.dtype != delta_b.dtype:
            delta_b = delta_b.to(dtype=fc2.bias.dtype)
        fc2.bias.add_(delta_b)
    else:
        if hasattr(fc2, "bias"):
            del fc2.bias
        fc2.register_buffer('bias', delta_b)

    fc2.weight.mul_(scales.to(device=w2.device, dtype=w2.dtype).view(1, -1))

def smooth_q_k_inplace(q_proj, k_proj, scales,):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    q_proj.weight.div_(scales.view(-1,1))
    q_proj.bias.div_(scales.view(-1))
    k_proj.weight.mul_(scales.view(-1,1))
    k_proj.bias.mul_(scales.view(-1))




def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():

        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, is_llama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if is_llama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.quantizer(module.temp_weight)
            else:
                module.temp_weight = module.quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, is_llama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if is_llama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    # self.use_weight_quant = weight_quant
    # self.use_act_quant = act_quant
    from models.ptq.layers import QAct,QLinear
    for m in self.modules():
        if isinstance(m, (QLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
        if isinstance(m, QAct):
            m.set_quant_state(weight_quant, act_quant)
            
def set_lwc_state(self, lwc):
    from models.ptq.layers import QLinear
    for m in self.modules():
        if isinstance(m, QLinear):
            m.set_lwc_state(lwc)

@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

import torch
import torch.nn as nn

from .int_linear import QuantLinear  


def _is_quant_linear(m: nn.Module) -> bool:

    return isinstance(m, QuantLinear) or (
        isinstance(m, nn.Linear) and hasattr(m, "quantizer")
    )



def register_dino_let_params(qlayer: nn.Module, dtype, device):

    import torch
    import torch.nn as nn
    from quantize.int_linear import QuantLinear

    modules = list(qlayer.named_modules())

    for ln_name, ln_mod in modules:
        if not isinstance(ln_mod, nn.LayerNorm):
            continue

        ln_dim = ln_mod.normalized_shape[0]
        ln_tag = ln_name.replace(".", "_")

        for lin_name, lin_mod in modules:
            if not isinstance(lin_mod, QuantLinear):
                continue

            # LN 输出 = Linear 输入
            if lin_mod.in_features != ln_dim:
                continue

            lin_tag = lin_name.replace(".", "_")
            base = f"{ln_tag}__{lin_tag}"

            scale = torch.ones(lin_mod.in_features, dtype=dtype, device=device)
            shift = torch.zeros(lin_mod.in_features, dtype=dtype, device=device)

            qlayer.register_parameter(
                f"{base}_smooth_scale", nn.Parameter(scale)
            )
            qlayer.register_parameter(
                f"{base}_smooth_shift", nn.Parameter(shift)
            )

@torch.no_grad()
def smooth_and_quant_temporary_dino(qlayer: nn.Module, args):

    import torch
    import torch.nn as nn
    from quantize.int_linear import QuantLinear

    suffix = "_smooth_scale"

    for pname, p in qlayer.named_parameters():
        if pname.endswith(suffix):
            p.data = torch.clamp(p.data, min=1e-5)


    for name, module in qlayer.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue

        ln_tag = name.replace(".", "_")

        for pname, _ in qlayer.named_parameters():
            if not (pname.startswith(ln_tag) and pname.endswith(suffix)):
                continue

            base = pname[:-len(suffix)]
            scale = getattr(qlayer, base + "_smooth_scale")
            shift = getattr(qlayer, base + "_smooth_shift")

            module.use_temporary_parameter = True
            module.temp_weight = module.weight / scale

            if module.bias is None:
                module.temp_bias = (-shift) / scale
            else:
                module.temp_bias = (module.bias - shift) / scale

    for name, module in qlayer.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        lin_tag = name.replace(".", "_")

        has_let = False
        cur_weight = module.weight
        cur_bias = module.bias

        for pname, _ in qlayer.named_parameters():
            if not (pname.endswith(suffix) and lin_tag in pname):
                continue

            base = pname[:-len(suffix)]
            scale = getattr(qlayer, base + "_smooth_scale")
            shift = getattr(qlayer, base + "_smooth_shift")

            has_let = True
            cur_weight = cur_weight * scale.view(1, -1)

        w = module.weight
        shift_ = shift.to(device=w.device, dtype=w.dtype)

        if cur_bias is not None:
            cur_bias = cur_bias.to(device=w.device, dtype=w.dtype)
            cur_bias = cur_bias + w @ shift_
        else:
            cur_bias = w @ shift_

        if has_let:
            module.use_temporary_parameter = True
            module.temp_weight = cur_weight
            module.temp_bias = cur_bias

    for m in qlayer.modules():
        if not isinstance(m, QuantLinear):
            continue

        if hasattr(m, "temp_weight"):
            m.temp_weight = m.quantizer(m.temp_weight)
        else:
            m.temp_weight = m.quantizer(m.weight)

        if not hasattr(m, "temp_bias"):
            m.temp_bias = m.bias

        m.use_temporary_parameter = True
