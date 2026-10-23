# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build fused kernels against FlashInfer's own CUTLASS headers."""

import hashlib
from functools import cache
from pathlib import Path

_REPOSITORY_SOURCE = (
    Path(__file__).resolve().parents[6] / "csrc/quantization/bitblend/sm120"
)
_SOURCE = (
    _REPOSITORY_SOURCE
    if _REPOSITORY_SOURCE.is_dir()
    else Path(__file__).parent / "csrc"
)


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(_SOURCE.rglob("*")):
        if path.is_file() and path.suffix in (".cu", ".hpp", ".cuh"):
            digest.update(str(path.relative_to(_SOURCE)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@cache
def warp_module(
    tier: str,
    block_k=128,
    stages=2,
    token_warps=1,
    matrix_load=0,
    register_prefetch=0,
    vector_epilogue=0,
):
    """Compile a dual-plane specialization before inference or graph capture."""
    if tier not in ("mid", "high"):
        raise ValueError("Dual-plane compilation requires Mid or High")
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    extension = gen_jit_spec(
        f"vllm_bitblend_sm120_{tier}_warp_{source_digest()[:16]}"
        f"_k{block_k}s{stages}q{token_warps}l{matrix_load}"
        f"r{register_prefetch}e{vector_epilogue}",
        [_SOURCE / f"{tier}_warp.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[12]
        )
        + [
            f"-I{_SOURCE}",
            f"-DBITBLEND_BLOCK_K={block_k}",
            f"-DBITBLEND_STAGES={stages}",
            f"-DBITBLEND_TOKEN_WARPS={token_warps}",
            f"-DBITBLEND_MATRIX_LOAD={matrix_load}",
            f"-DBITBLEND_REGISTER_PREFETCH={register_prefetch}",
            f"-DBITBLEND_VECTOR_EPILOGUE={vector_epilogue}",
        ],
    ).build_and_load()
    import torch

    extension.initialize(torch.cuda.current_device())
    return extension


@cache
def grouped_module(variant):
    """Compile one compatible heterogeneous bundle, using CUTLASS SM120 MMA."""
    import torch
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    fields = (
        "TOKENS",
        "CHANNELS",
        "BLOCK_K",
        "STAGES",
        "TOKEN_WARPS",
        "MATRIX_LOAD",
        "REGISTER_PREFETCH",
        "VECTOR_EPILOGUE",
    )
    flags = [
        f"-DGROUPED_{tier}_{field}={value}"
        for tier, values in zip(("LOW", "MID", "HIGH"), variant)
        for field, value in zip(fields, values)
    ]
    key = hashlib.sha256(repr(variant).encode()).hexdigest()[:16]
    extension = gen_jit_spec(
        f"vllm_bitblend_sm120_grouped_{source_digest()[:16]}_{key}",
        [_SOURCE / "grouped.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[12]
        )
        + [f"-I{_SOURCE}", f"-DGROUPED_MIN_BLOCKS={variant[3]}", *flags],
    ).build_and_load()
    extension.initialize(torch.cuda.current_device())
    return extension
