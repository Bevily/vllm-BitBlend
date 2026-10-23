# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline canonical BitBlend encoding and Qwen3 checkpoint export."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.bitblend.reference import (
    _FP4_VALUES,
    decode_plane,
)
from vllm.model_executor.layers.quantization.bitblend.profile import (
    MX_PROFILE,
    PlaneSpec,
    ProfileSpec,
    get_profile,
)


def encode_plane(
    weight: torch.Tensor, plane: PlaneSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a finite [N, K] matrix into unswizzled checkpoint bytes."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Weight must be a floating-point [N, K] matrix")
    n, k = weight.shape
    plane.shapes(n, k)
    values = weight.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Weight must contain only finite values")
    blocks = values.reshape(n, -1, plane.scale_block or k)
    amax = blocks.abs().amax(-1, keepdim=True)
    if plane.scale_block:
        if plane.bits == 4:
            rounded = ((amax.view(torch.int32) + 0x200000) & 0x7F800000).view(
                torch.float32
            )
            exponent = (
                torch.floor(
                    torch.log2(rounded.clamp_min(torch.finfo(torch.float32).tiny))
                )
                - 2
            )
        else:
            exponent = torch.ceil(torch.log2(amax / 448.0))
        exponent = exponent.clamp(-127, 127).int()
        scales = (exponent.squeeze(-1) + 127).to(torch.uint8)
        normalized = torch.ldexp(blocks, -exponent).reshape(n, k)
    else:
        scales = (amax.reshape(n, 1) / 448).clamp_min(torch.finfo(torch.float32).tiny)
        scales = torch.where(amax.reshape(n, 1) == 0, 1.0, scales)
        normalized = values / scales
    if plane.bits == 4:
        table = torch.tensor(_FP4_VALUES, device=values.device)
        # Bucketize midpoints with even-code ties, avoiding an [N,K,8] tensor.
        boundaries = (table[:-1] + table[1:]) / 2
        codes = torch.bucketize(normalized.abs().contiguous(), boundaries)
        for boundary_index in (1, 3, 5):
            codes = torch.where(
                normalized.abs() == boundaries[boundary_index],
                boundary_index + 1,
                codes,
            )
        codes = codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
        payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    else:
        payload = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    return payload.contiguous(), scales.contiguous()


def encode_weight(
    weight: torch.Tensor, profile: ProfileSpec = MX_PROFILE
) -> dict[str, torch.Tensor]:
    """Encode alternative residual planes from the same decoded base."""
    result = {}
    decoded = {}
    for plane in profile.planes:
        values = weight.float()
        if plane.residual_of:
            values = values - decoded[plane.residual_of]
        payload, scales = encode_plane(values, plane)
        result[f"weight_{plane.name}"] = payload
        result[f"weight_scale_{plane.name}"] = scales
        decoded[plane.name] = decode_plane(payload, scales, plane)
    return result


def export_qwen3(
    source: Path,
    destination: Path,
    profile: ProfileSpec = MX_PROFILE,
    device: str = "cpu",
) -> dict:
    """Export a single-file dense Qwen3 checkpoint without modifying the source.

    A temporary sibling directory is renamed only after all shards are saved.
    Existing destinations are rejected. All seven projection weights per layer
    must exist; embeddings, norms and the tied output head remain unquantized.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("quantization_config"):
        raise ValueError("Expected an unquantized dense Qwen3 checkpoint")
    projections = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.weight")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent)
    )
    try:
        weight_map, seen = {}, set()
        total_size = 0
        with safe_open(source / "model.safetensors", framework="pt") as reader:
            names_on_disk = reader.keys()
            if (
                config.get("tie_word_embeddings")
                and "lm_head.weight" in names_on_disk
                and not torch.equal(
                    reader.get_tensor("lm_head.weight"),
                    reader.get_tensor("model.embed_tokens.weight"),
                )
            ):
                raise ValueError("Tied embedding and lm_head weights differ")
            groups: dict[str, list[str]] = {}
            for name in names_on_disk:
                match = pattern.fullmatch(name)
                group = f"layer-{int(match[1]):02d}" if match else "unquantized"
                groups.setdefault(group, []).append(name)
            for group, names in sorted(groups.items()):
                tensors = {}
                for name in names:
                    weight = reader.get_tensor(name)
                    match = pattern.fullmatch(name)
                    if match and match[3] in projections:
                        seen.add((int(match[1]), match[3]))
                        encoded = encode_weight(weight.to(device), profile)
                        prefix = name.removesuffix("weight")
                        tensors.update(
                            {
                                prefix + key: value.cpu()
                                for key, value in encoded.items()
                            }
                        )
                    else:
                        if match:
                            raise ValueError(f"Unexpected projection: {name}")
                        tensors[name] = weight
                filename = f"model-{group}.safetensors"
                save_file(tensors, staging / filename, metadata={"format": "pt"})
                for name, value in tensors.items():
                    weight_map[name] = filename
                    total_size += value.numel() * value.element_size()
                print(f"Exported {group}: {len(tensors)} tensors", flush=True)
        expected = {
            (i, p) for i in range(config["num_hidden_layers"]) for p in projections
        }
        if seen != expected:
            raise ValueError(f"Projection coverage mismatch: {expected ^ seen}")
        quant_config = {
            "quant_method": "bitblend",
            "schema_version": 1,
            "profile": profile.name,
            "backend": "reference",
            "tier": "low",
        }
        config["quantization_config"] = quant_config
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "LICENSE",
            "README.md",
        ):
            if (source / name).is_file():
                shutil.copy2(source / name, staging / name)
        with (source / "model.safetensors").open("rb") as source_file:
            source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
        manifest = {
            "source": str(source),
            "profile": profile.name,
            "source_sha256": source_hash,
            "quantized_projections": len(seen),
            "output_tensors": len(weight_map),
            "tensor_bytes": total_size,
        }
        for filename, data in (
            ("config.json", config),
            ("bitblend_config.json", quant_config),
            ("bitblend_export.json", manifest),
            (
                "model.safetensors.index.json",
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            ),
        ):
            (staging / filename).write_text(json.dumps(data, indent=2) + "\n")
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", default=MX_PROFILE.name)
    args = parser.parse_args()
    print(
        json.dumps(
            export_qwen3(
                args.source, args.destination, get_profile(args.profile), args.device
            ),
            indent=2,
        )
    )
