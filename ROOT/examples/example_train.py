# 2025.11.27-Changed for main script for ROOT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,  
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from tqdm import tqdm

def simple_quantile(tensor: torch.Tensor, q: float) -> torch.Tensor:
    """
    A simple quantile implementation (unoptimized for GPU) for this toy script.
    """
    tensor = tensor.flatten()
    num_elements = tensor.numel()
    if num_elements == 0:
        return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
    index = torch.tensor(q * (num_elements - 1), device=tensor.device)

    lower_index = torch.floor(index).long()
    upper_index = torch.ceil(index).long()

    k_lower = lower_index + 1
    k_upper = upper_index + 1

    if k_lower == k_upper:
        return tensor.kthvalue(k_lower).values

    lower_value = tensor.kthvalue(k_lower).values
    upper_value = tensor.kthvalue(k_upper).values

    weight = index - lower_index
    return torch.lerp(lower_value, upper_value, weight)
    

    
# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def root5(G, steps):
    assert len(G.shape) == 2
    # Coefficients for ROOT used with 4 shapes used in this toy-train script
    COEFF_MAP_CANONICAL = {
        (2048, 2048): (3.4916, -4.8224, 2.1095),
        (2048, 16384): (3.1943, -3.8221, 1.5322),
        (2048, 3072): (3.4509, -4.5790, 1.9325),
        (2048, 8192): (2.9794, -3.6674, 1.6207)
    }
    rows, cols = G.size(0), G.size(1)

    shape_key = (min(rows, cols), max(rows, cols))
    try:
        a, b, c = COEFF_MAP_CANONICAL[shape_key]
    except KeyError:
        # Fallback to default Newton-Schulz coefficients for unknown shapes
        a, b, c = (3.4445, -4.7750, 2.0315)
        logger.warning(f"Shape {shape_key} not found. Using default coefficients.")
    X = G
    is_tall = rows > cols
    if is_tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if is_tall:
        X = X.T

    return X
    
# This code snippet is a modified version adapted from the following GitHub repositories:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# https://github.com/MoonshotAI/Moonlight/

class ROOT(torch.optim.Optimizer):
    """
    ROOT: Robust Orthogonalized Optimizer for Neural Network Training

    Paper: https://arxiv.org/abs/2511.20626
    Authors: Wei He, Kai Han, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao Chen, Yunhe Wang
    
    Arguments:
        root_params: The parameters to be optimized by ROOT.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        root_steps: The number of ROOT iterations. Unlike static Newton-Schulz steps, ROOT uses 
                    adaptive iterations tailored to matrix size and condition. 
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `root_params` which are
                      {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        root_params=None,
        momentum=0.95,
        nesterov=True,
        root_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            root_steps=root_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(root_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use ROOT, and those for which we will not
        for p in root_params:
            # Use ROOT for every parameter in root_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_root"] = True
        for p in adamw_params:
            # Do not use ROOT for parameters in adamw_params
            self.state[p]["use_root"] = False

    def adjust_lr_for_root(self, lr, param_shape):
        A, B = param_shape[:2]
        # Follow Moonlight(https://arxiv.org/abs/2502.16982), we adjust the learning rate and weight decay based on the size of the parameter matrix
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           ROOT           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_root"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                ## ROOT
                epsilon = simple_quantile(g.abs().float(), 0.9) + 1e-9
                o = torch.sign(g) * torch.nn.functional.relu(torch.abs(g) - epsilon)
                b = g - o
                u = root5(b, steps=group["root_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_root(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_root"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class Attention(nn.Module):
    def __init__(self, dim, n_head, n_kv, head_dim):
        super().__init__()
        self.n_head, self.n_kv, self.h_dim = n_head, n_kv, head_dim
        self.rep = n_head // n_kv
        
        self.qkv_proj = nn.Linear(dim, (n_head + 2 * n_kv) * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, self.n_kv, self.rep + 2, self.h_dim)
        
        q, k, v = qkv.split([self.rep, 1, 1], dim=3)
        
        q = q.reshape(B, T, self.n_head, self.h_dim).transpose(1, 2)
        k = k.squeeze(3).transpose(1, 2)
        v = v.squeeze(3).transpose(1, 2)

        q, k = self.apply_rope(q, k, freqs_cis)

        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))

    def apply_rope(self, q, k, freqs_cis):
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[None, None, :q.shape[2], :]
        return torch.view_as_real(q_ * freqs_cis).flatten(3).type_as(q), \
               torch.view_as_real(k_ * freqs_cis).flatten(3).type_as(k)

class MLP(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# This sample model definition is adapted from the Hugging Face transformers library.  https://github.com/huggingface/transformers
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conf = dict(vocab=128256, dim=2048, inter=8192, layers=16, 
                         n_head=32, n_kv=8, h_dim=64, rope_theta=500000.0)
        
        self.embed_tokens = nn.Embedding(self.conf['vocab'], self.conf['dim'])
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': RMSNorm(self.conf['dim']),
                'attn': Attention(self.conf['dim'], self.conf['n_head'], self.conf['n_kv'], self.conf['h_dim']),
                'norm2': RMSNorm(self.conf['dim']),
                'mlp': MLP(self.conf['dim'], self.conf['inter'])
            }) for _ in range(self.conf['layers'])
        ])
        self.norm = RMSNorm(self.conf['dim'])
        self.lm_head = nn.Linear(self.conf['dim'], self.conf['vocab'], bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self.register_buffer("freqs_cis", self.precompute_freqs_cis(self.conf['h_dim'], 131072))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02 # default
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def precompute_freqs_cis(self, dim, end, theta=500000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        for l in self.layers:
            h = l['norm1'](x)
            x = x + l['attn'](h, freqs_cis)
            h = l['norm2'](x)
            x = x + l['mlp'](h)
            
        logits = self.lm_head(self.norm(x))
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.conf['vocab']), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(loss=loss, logits=logits)

# Source: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py    
class ROOTDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


def get_model_and_dataloader(model_name, dataset_name):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    rain_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    if model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
                "Xenova/llama3-tokenizer", 
                trust_remote_code=True,
            )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = ROOTDataset(dataset_name, train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


    if model_name == "llama":
        print("Initializing SampleModel (ROOT-sample) with corrected initialization...")
        model = SampleModel()
    else:
        assert 0, f"model {model_name} not supported"
    
    return model, train_loader    
    

def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "root":
        root_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return ROOT(
            lr=lr,
            wd=wd,
            root_params=root_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--optimizer", type=str, default="root")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    args = parser.parse_args()
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")

    model, train_loader = get_model_and_dataloader(
        args.model, args.dataset)
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    num_epochs = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * num_epochs,
        num_cycles=0.5,
    )
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']} Training loss: {loss.item()}"
            )
