# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify a local Qwen3 BitBlend checkpoint through the native vLLM loader.

Runs TP=1 on CUDA, checks every loaded checkpoint tensor including fused shard
placement, and exercises all linear layers at every tier. This does not run
attention, token generation, or distributed multi-GPU execution.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import envs
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.quantization import resolve_quantization_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.kernels.linear.bitblend.reference import (
    decode_plane,
    quantize_activation,
)
from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
    database_provenance,
)
from vllm.model_executor.layers.quantization.bitblend.checkpoint import encode_weight
from vllm.model_executor.layers.quantization.bitblend.linear import BitBlendLinearMethod
from vllm.model_executor.layers.quantization.bitblend.runtime import (
    ExecutionPlan,
    execution_context,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@torch.inference_mode()
def validate(source: Path, checkpoint: Path, backend: str | None = None) -> dict:
    """Fail on incomplete loading, incorrect fusion, or reference disagreement."""
    manifest = json.loads((checkpoint / "bitblend_export.json").read_text())
    with (source / "model.safetensors").open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    if source_hash != manifest["source_sha256"]:
        raise ValueError("Source checkpoint differs from the export provenance")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    overrides = {}
    if backend:
        quant_config = json.loads((checkpoint / "config.json").read_text())[
            "quantization_config"
        ]
        overrides["quantization_config"] = {**quant_config, "backend": backend}
    model_config = ModelConfig(
        model=str(checkpoint),
        dtype="bfloat16",
        max_model_len=256,
        enforce_eager=True,
        hf_overrides=overrides,
    )
    model_config.quantization_config = resolve_quantization_config(
        model_config.quantization, model_config.quantization_config
    )
    config = VllmConfig(
        model_config=model_config,
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
    )
    with (
        tempfile.TemporaryDirectory(prefix="bitblend-dist-") as temp,
        set_current_vllm_config(config),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp}/store",
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        try:
            loader = DefaultModelLoader(config.load_config)
            model = loader.load_model(config, model_config)
            return check_model(model, source, checkpoint)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def check_model(model: torch.nn.Module, source: Path, checkpoint: Path) -> dict:
    """Compare disk bytes and original BF16 projections with loaded modules."""
    params = dict(model.named_parameters())
    config = model.quant_config
    profile = config.profile
    mapper = model.hf_to_vllm_mapper
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    matched, covered = 0, set()
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(checkpoint / filename, framework="pt") as reader:
            tensor_names = reader.keys()
            for name in tensor_names:
                mapped, shard = mapper._map_name_with_shard(name)
                if mapped == "lm_head.weight":
                    assert model.lm_head.weight is model.model.embed_tokens.weight
                    param = model.lm_head.weight
                else:
                    param = params[mapped]
                stored = reader.get_tensor(name)
                if shard is not None:
                    module = model.get_submodule(mapped.rsplit(".", 1)[0])
                    if isinstance(shard, str):
                        offset = module._get_shard_offset_mapping(shard)
                    else:
                        offset = sum(module.output_sizes[:shard])
                    param = param.narrow(0, offset, stored.shape[0])
                torch.testing.assert_close(param.cpu(), stored, rtol=0, atol=0)
                covered.add(mapped)
                matched += 1
    # lm_head is tied and excluded from named_parameters().
    missing = set(params) - covered
    if missing:
        raise AssertionError(f"Unverified model parameters: {sorted(missing)}")

    errors = {tier.name: [0.0, 0.0] for tier in profile.tiers}
    kernel_errors = {tier.name: 0.0 for tier in profile.tiers}
    layers = 0
    unquantized = 0
    with safe_open(source / "model.safetensors", framework="pt") as original:
        source_names_on_disk = original.keys()
        for name in source_names_on_disk:
            if name in params or name == "lm_head.weight":
                param = model.get_parameter(name)
                torch.testing.assert_close(
                    param.cpu(), original.get_tensor(name), rtol=0, atol=0
                )
                unquantized += 1
        for name, layer in model.named_modules():
            if not isinstance(
                getattr(layer, "quant_method", None), BitBlendLinearMethod
            ):
                continue
            if name.endswith("qkv_proj"):
                source_names = [
                    name.replace("qkv_proj", part + "_proj") for part in ("q", "k", "v")
                ]
            elif name.endswith("gate_up_proj"):
                source_names = [
                    name.replace("gate_up_proj", part + "_proj")
                    for part in ("gate", "up")
                ]
            else:
                source_names = [name]
            weight = (
                torch.cat(
                    [original.get_tensor(key + ".weight") for key in source_names]
                )
                .cuda()
                .float()
            )
            encoded = encode_weight(weight, profile)
            for key, tensor in encoded.items():
                torch.testing.assert_close(getattr(layer, key), tensor, rtol=0, atol=0)
            decoded = {
                plane.name: decode_plane(
                    encoded[f"weight_{plane.name}"],
                    encoded[f"weight_scale_{plane.name}"],
                    plane,
                )
                for plane in profile.planes
            }
            x = torch.randn(2, weight.shape[1], device="cuda", dtype=torch.bfloat16)
            baseline = x.float() @ weight.T
            for tier in profile.tiers:
                with execution_context(ExecutionPlan(profile.name, tier.name)):
                    actual = layer(x)[0]
                xq = quantize_activation(
                    x,
                    tier.activation,
                    fp4_scale_rule=(
                        "rounded-amax" if config.backend == "reference" else "ceil"
                    ),
                )
                expected = sum(xq @ decoded[p].T for p in tier.planes).bfloat16()
                if config.backend == "reference":
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                relative = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-30)
                )
                kernel_errors[tier.name] = max(kernel_errors[tier.name], relative)
                assert relative < 0.01, (name, tier.name, relative)
                assert bool(torch.isfinite(actual).all())
                errors[tier.name][0] += (
                    (actual.float() - baseline).square().sum().item()
                )
                errors[tier.name][1] += baseline.square().sum().item()
            layers += 1
            print(f"Verified {name} ({layers} linear layers)", flush=True)
    expected_layers = model.config.num_hidden_layers * 4
    assert layers == expected_layers, (layers, expected_layers)
    relative_errors = {tier: (a / b) ** 0.5 for tier, (a, b) in errors.items()}
    assert relative_errors["high"] < relative_errors["low"]
    return {
        "model": str(source),
        "checkpoint": str(checkpoint),
        "model_class": type(model).__name__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "backend": config.backend,
        "tactic_database": database_provenance(
            envs.VLLM_BITBLEND_SM120_TACTICS, use_default=config.backend != "reference"
        ),
        "tensor_parallel_size": 1,
        "checkpoint_tensors_verified": matched,
        "model_parameters_verified": len(params),
        "source_unquantized_tensors_verified": unquantized,
        "linear_layers": layers,
        "tier_checks": layers * len(profile.tiers),
        "relative_l2_vs_bf16_linear": relative_errors,
        "max_layer_relative_l2_vs_reference": kernel_errors,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "full_generation_tested": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", help="Registered BitBlend backend")
    args = parser.parse_args()
    report = validate(args.source.resolve(), args.checkpoint.resolve(), args.backend)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
