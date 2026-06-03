# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import gc
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from configure_logger import configure_logger
from utils import apply_chat_template, load_model_and_tokenizer, load_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
configure_logger(logger, "construct_predictors.log")


def parse_ranks_as_string(ranks_str: str):
    try:
        values = [int(x) for x in ranks_str.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError("Invalid ranks") from err
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument("--model_path", help="Model Path", type=str, required=True)
    parser.add_argument(
        "--calibration_prompts_path", help="Path to calibration prompts (json)", type=str, required=True
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rank", help="Predictor rank", type=int)
    group.add_argument(
        "--ranks", type=parse_ranks_as_string, help="Comma-separated list of ranks (e.g., 256,256,512,...,128)"
    )
    parser.add_argument("--s", help="Predicted sparsity hyperparam", required=True, type=float)
    parser.add_argument("--torch_dtype", help="Data type for LLM", required=True, type=str)
    parser.add_argument("--device_map", help="Device map for LLM", required=True, type=str)
    parser.add_argument("--predictors_output_path", help="Predictors Output Path", required=True)
    parser.add_argument(
        "--sparsity_plot_output_file", type=str, required=False, default="sparsity.pdf", help="Output filename (pdf)"
    )
    parser.add_argument(
        "--ablation_config_whitening", choices=["w/o", "cholesky"], required=False, default="cholesky", help="Ablation"
    )
    parser.add_argument(
        "--ablation_config_bias", choices=["w/o", "const", "full"], required=False, default="full", help="Ablation"
    )
    parser.add_argument(
        "--ablation_config_penalty",
        choices=["gate", "gate/up", "full"],
        required=False,
        default="full",
        help="Ablation",
    )
    args = parser.parse_args()
    return args


class Predictor(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, rank: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, rank, bias=False, dtype=torch.float16)
        self.fc2 = torch.nn.Linear(rank, intermediate_size, bias=True, dtype=torch.float16)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def save_predictor(predictor, layer_id: int, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    torch.save(predictor.state_dict(), os.path.join(output_path, f"model_{layer_id}.pt"))


# WARNING suppose batch_size=1
def collect_hidden_activations(model, tokenizer, prompts: list[str]) -> tuple:
    MAX_CONTEXT_LENGTH = 2048
    n_layers = len(model.model.layers)
    mlp_input_history = [[] for _ in range(n_layers)]
    gate_output_history = [[] for _ in range(n_layers)]
    down_input_history = [[] for _ in range(n_layers)]

    def gate_hook(module, x, output, layer_id):
        store_dtype = torch.float16
        mlp_input_history[layer_id].append(x[0][0].cpu().to(store_dtype))
        gate_output_history[layer_id].append(output[0].cpu().to(store_dtype))

    def down_hook(module, x, output, layer_id):
        store_dtype = torch.float16
        down_input_history[layer_id].append(x[0][0].cpu().to(store_dtype))

    hook_handles = []

    def activate_hooks():
        for layer_id, layer in enumerate(model.model.layers):
            hook_handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    lambda module, input, output, layer_id=layer_id: gate_hook(module, input, output, layer_id)
                )
            )
            hook_handles.append(
                layer.mlp.down_proj.register_forward_hook(
                    lambda module, input, output, layer_id=layer_id: down_hook(module, input, output, layer_id)
                )
            )

    def deactivate_hooks():
        for h in hook_handles:
            h.remove()
        hook_handles.clear()

    activate_hooks()
    for i in tqdm(range(len(prompts)), desc="Collecting hidden activations from calibration prompts"):
        inputs = tokenizer(
            [prompts[i]],
            truncation=True,
            return_tensors="pt",
            max_length=MAX_CONTEXT_LENGTH,
            add_special_tokens=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        _ = model(**inputs)
    deactivate_hooks()

    torch.cuda.empty_cache()
    logger.info("Aggregating the results...")

    mlp_inputs = [
        torch.concat(
            mlp_input_history[layer_id],
            dim=0,
        )
        for layer_id in range(n_layers)
    ]
    del mlp_input_history
    gc.collect()

    gate_outputs = [
        torch.concat(
            gate_output_history[layer_id],
            dim=0,
        )
        for layer_id in range(n_layers)
    ]
    del gate_output_history
    gc.collect()

    down_inputs = [
        torch.concat(
            down_input_history[layer_id],
            dim=0,
        )
        for layer_id in range(n_layers)
    ]
    del down_input_history
    gc.collect()

    return mlp_inputs, gate_outputs, down_inputs


def estimate_sparsity(gate_outputs, plot_output_path: str | None = None) -> np.array:
    from matplotlib import pyplot as plt

    threshold = 1e-8
    n_layers = len(gate_outputs)

    sparsities = [(a <= threshold).to(torch.float16).mean(dim=-1) for a in gate_outputs]  # (n_layers, n_tokens)
    sparsities_mean = np.array([a.mean().item() for a in sparsities]) * 100
    sparsities_p10 = np.array([np.quantile(a, q=0.1).item() for a in sparsities]) * 100
    sparsities_p90 = np.array([np.quantile(a, q=0.9).item() for a in sparsities]) * 100

    x = range(n_layers)

    plt.figure(figsize=(4, 4))

    plt.plot(sparsities_mean)
    plt.fill_between(
        x,
        sparsities_p10,
        sparsities_p90,
        color="blue",
        alpha=0.1,
    )
    plt.grid(True)
    plt.xlabel("Layer ID")
    plt.ylabel("Sparsity, %")

    plt.ylim([0, 100])

    # save plot
    if plot_output_path:
        plot_dir = os.path.dirname(plot_output_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_output_path, bbox_inches="tight", format="pdf", pad_inches=0.1)
    logger.info(f"Sparsity plot has been saved to {plot_output_path}")

    return sparsities_mean


def compute_predictor_weights(
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    r: int,
    mlp_input: torch.Tensor,
    gate_output: torch.Tensor,
    down_input: torch.Tensor,
    act_fn: torch.nn.Module,
    desired_sparsity: float,
    ablation_config: dict,
):
    compute_dtype = torch.float64
    compute_device = "cpu"
    n_neurons = w_gate.shape[0]
    n_tokens = mlp_input.shape[0] // 2  # half of the dataset for S construction and other for bias calibration

    ### AB construction
    X = mlp_input[:n_tokens].to(compute_device).to(compute_dtype)
    w = w_gate.to(compute_device).to(compute_dtype)

    if ablation_config["whitening"] == "cholesky":
        S = X.T @ X
        S = torch.linalg.cholesky(S, upper=False)
        u, s, v = torch.linalg.svd(w @ S)
        v = torch.linalg.solve(S.T, v.T).T
        del S
    else:
        u, s, v = torch.linalg.svd(w)

    down_proj = (v * (s**0.0).unsqueeze(1))[:r]
    up_proj = u[:, :r] * (s[:r] ** 1.0).unsqueeze(0)
    del u, s, v

    gc.collect()
    torch.cuda.empty_cache()

    ### Bias calibration

    # no bias
    if ablation_config["bias"] == "w/o":
        bias = torch.zeros(n_neurons, dtype=down_proj.dtype)
        return up_proj, down_proj, -bias

    X = mlp_input[n_tokens:].to(compute_device).to(compute_dtype)  # (n_tokens, d)
    predicted_values = (X @ (down_proj.T @ up_proj.T)).T  # (D, n_tokens)

    # same bias for all neurons
    if ablation_config["bias"] == "const":
        threshold = np.quantile(
            predicted_values.flatten().detach().cpu().to(torch.float32).numpy(), desired_sparsity
        ).item()
        bias = torch.ones(n_neurons, dtype=down_proj.dtype) * threshold
        return up_proj, down_proj, -bias

    match ablation_config["penalty"]:
        case "gate":
            neuron_importance = act_fn(gate_output[n_tokens:]).T.to(compute_device).to(compute_dtype)  # (D, n_tokens)
        case "gate/up":
            neuron_importance = down_input[n_tokens:].T.to(compute_device).to(compute_dtype)  # (D, n_tokens)
        case "full":
            down_norms = torch.linalg.norm(w_down.to(compute_device).to(compute_dtype), dim=0, keepdim=True).T  # (D, 1)
            neuron_importance = (
                down_input[n_tokens:].T.to(compute_device).to(compute_dtype) * down_norms
            )  # (D, n_tokens)
        case _:
            raise ValueError("Unknown penalty type value!")

    del X
    gc.collect()
    torch.cuda.empty_cache()

    sort_indices = torch.argsort(predicted_values, dim=-1)  # (D, n_tokens)
    sorted_neuron_importance = torch.gather(neuron_importance, dim=-1, index=sort_indices).to(
        torch.float32
    )  # (D, n_tokens)

    penalty = torch.cumsum(sorted_neuron_importance**2, dim=-1)  # (D, n_tokens)

    # todo: this may lead to non-uniform distribution with rare data on the edges
    # todo: average
    target_size = 256
    indices = torch.linspace(0, penalty.shape[-1] - 1, target_size).to(torch.int64)
    multiplier = penalty.shape[-1] // target_size
    penalty = penalty[:, indices]  # (D, target_size)
    delta_penalty = penalty[:, 1:] - penalty[:, :-1]
    delta_penalty = torch.concat([delta_penalty, torch.ones(n_neurons, 1) * float("inf")], dim=-1)

    # init thresholds
    thresholds = torch.clip(
        (penalty.cumsum(dim=-1) == 0).sum(dim=-1).to(torch.int64) - 1,
        0,
        int(0.95 * target_size) - 1,
    )

    sparsity = thresholds.to(torch.float16).mean().item() / target_size

    loss_increase = torch.gather(delta_penalty, dim=-1, index=thresholds.unsqueeze(1)).squeeze(1)
    n_turned_off_neurons = (thresholds == target_size - 1).sum()
    if n_turned_off_neurons > 0.01 * n_neurons:
        logger.warning("We turned off 1% of neurons, exit")
    elif sparsity > desired_sparsity:
        logger.info(f"Sparsity {100 * sparsity:.1f} exceeding desired sparsity {100 * desired_sparsity:.1f}")
    else:
        logger.info(f"Increasing sparsity from {100 * sparsity:.1f} to {100 * desired_sparsity:.1f}")
        n_steps = int((desired_sparsity - sparsity) / (1 / n_neurons / target_size))
        for _ in tqdm(range(n_steps)):  # desired sparsity
            best_neuron = torch.argmin(loss_increase)
            thresholds[best_neuron] += 1
            if thresholds[best_neuron] == target_size - 1:
                n_turned_off_neurons += 1
                if n_turned_off_neurons > 0.01 * n_neurons:
                    logger.warning("We turned off 1% of neurons, exit")
                    break
            loss_increase[best_neuron] = delta_penalty[best_neuron][thresholds[best_neuron]]

    sorted_predicted_values = torch.gather(predicted_values, dim=-1, index=sort_indices)
    bias = torch.gather(sorted_predicted_values, dim=-1, index=thresholds.unsqueeze(1) * multiplier).squeeze(1)
    torch.cuda.empty_cache()
    return up_proj, down_proj, -bias


def build_predictor(mlp, desired_sparsity, mlp_input, gate_output, down_input, rank, ablation_config):
    n_neurons = gate_output[0].shape[-1]
    hidden_size = mlp_input[0].shape[-1]
    predictor = Predictor(hidden_size=hidden_size, intermediate_size=n_neurons, rank=rank)
    up_proj, down_proj, bias = compute_predictor_weights(
        w_gate=mlp.gate_proj.weight,
        w_down=mlp.down_proj.weight,
        act_fn=mlp.act_fn,
        r=rank,
        mlp_input=mlp_input,
        gate_output=gate_output,
        down_input=down_input,
        desired_sparsity=desired_sparsity,
        ablation_config=ablation_config,
    )

    predictor.fc1.weight.data = down_proj.contiguous().to(torch.float16).to("cpu")
    predictor.fc2.weight.data = up_proj.contiguous().to(torch.float16).to("cpu")
    predictor.fc2.bias.data = bias.contiguous().to(torch.float16).to("cpu")
    return predictor


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, attn_implementation="sdpa", torch_dtype=args.torch_dtype, device_map=args.device_map
        )
    except ValueError:
        logger.info("sdpa attention is not supported, rollback to eager!")
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, attn_implementation="eager", torch_dtype=args.torch_dtype, device_map=args.device_map
        )

    prompts = load_prompts(args.calibration_prompts_path)
    logger.info(f"Collected {len(prompts)} prompts!")

    prompts = apply_chat_template(prompts, tokenizer)

    mlp_inputs, gate_outputs, down_inputs = collect_hidden_activations(model, tokenizer, prompts)
    logger.info(f"Calibration dataset consists of {mlp_inputs[0].shape[0]} tokens")

    model = model.to("cpu")

    sparsities = estimate_sparsity(gate_outputs, args.sparsity_plot_output_file)
    logger.info(f"Average sparsity -- {sparsities.mean().item():.1f}%")

    if args.rank:
        args.ranks = [args.rank for _ in range(len(mlp_inputs))]

    n_layers = len(model.model.layers)

    for layer_id in tqdm(range(n_layers), desc="Construct predictors"):
        desired_sparsity = (max(0, args.s - 0.1) if layer_id in (0, 1) else args.s) * sparsities[layer_id] / 100
        predictor = build_predictor(
            mlp=model.model.layers[layer_id].mlp,
            desired_sparsity=desired_sparsity,
            mlp_input=mlp_inputs[layer_id],
            gate_output=gate_outputs[layer_id],
            down_input=down_inputs[layer_id],
            rank=args.ranks[layer_id],
            ablation_config={
                "whitening": args.ablation_config_whitening,
                "bias": args.ablation_config_bias,
                "penalty": args.ablation_config_penalty,
            },
        )
        save_predictor(predictor, layer_id=layer_id, output_path=args.predictors_output_path)
        del predictor
        torch.cuda.empty_cache()

    logger.info("Done!")


if __name__ == "__main__":
    main()
