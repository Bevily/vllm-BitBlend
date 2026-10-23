# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned shape results with strict environment and exact-shape matching."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from functools import cache
from pathlib import Path

from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
    GroupedConfig,
    HighWarpConfig,
    MidWarpConfig,
)
from vllm.model_executor.kernels.linear.bitblend.sm120.jit import source_digest

SCHEMA_VERSION = 1
BUNDLED_DATABASE = Path(__file__).with_name("configs") / "rtx5090.json"


@cache
def fingerprint() -> dict:
    import torch
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS

    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    digest = hashlib.sha256()
    root = CUTLASS_INCLUDE_DIRS[0]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "ops.py",
            "jit.py",
            "config.py",
            "backend.py",
            "database.py",
            "activation.py",
            "graph.py",
        )
    }

    def tool_version(command):
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()

    nvcc = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    return {
        "device": prop.name,
        "capability": [prop.major, prop.minor],
        "sm_count": prop.multi_processor_count,
        "shared_memory_limit": prop.shared_memory_per_block_optin,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "cudnn": torch.backends.cudnn.version(),
        "cutlass_headers_sha256": digest.hexdigest(),
        "kernel_sha256": source_digest(),
        "python_sources": sources,
        "execution": "eager",
        "activation": "flashinfer-cutedsl-ceil-fp4-fp8/mixed-one-launch-v1",
        "output": "bfloat16",
        "python": platform.python_version(),
        "driver": tool_version(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[torch.cuda.current_device()],
        "nvcc": tool_version([nvcc, "--version"]),
        "compiler": tool_version([compiler or "c++", "--version"]),
    }


def shape_key(tier: str, m: int, n: int, k: int) -> str:
    return f"{tier}:{m}:{n}:{k}"


def grouped_shape_key(groups, n: int, k: int) -> str:
    return f"grouped:{n}:{k}:" + ",".join(f"{tier}/{m}" for tier, m in groups)


def graph_shape_key(m: int, n: int, k: int, max_spans: int = 12) -> str:
    return f"graph:{max_spans}:{m}:{n}:{k}"


@cache
def resolve_database_path(path: str | None = None) -> str | None:
    """Prefer an explicit table; use the bundled table only on an exact match."""
    if path:
        return path
    if BUNDLED_DATABASE.is_file():
        document = json.loads(BUNDLED_DATABASE.read_text())
        if document.get("fingerprint") == fingerprint():
            return str(BUNDLED_DATABASE)
    return None


def database_provenance(path: str | None, *, use_default: bool = False) -> dict | None:
    """Record the exact optional tactic artifact used by a model evaluation."""
    if use_default:
        path = resolve_database_path(path)
    if not path:
        return None
    source = Path(path).resolve()
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


class TacticDatabase:
    def __init__(self, document: dict, environment: dict):
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported SM120 database schema")
        if document.get("fingerprint") != environment:
            raise ValueError("SM120 tactic database environment fingerprint mismatch")
        if document.get("status") != "verified":
            raise ValueError("Only independently verified databases can be deployed")
        self.entries = document.get("selected", {})
        self._kernels = {}
        for key, entry in self.entries.items():
            tier, m, n, k = key.split(":")
            if min(int(m), int(n), int(k)) <= 0:
                raise ValueError("Database shapes must be positive")
            if tier in ("low", "bf16"):
                continue
            if entry.get("kernel") in ("high_warp_v1", "mid_warp_v1"):
                if entry["kernel"] != f"{tier}_warp_v1" or entry.get("tactic", 0) != 0:
                    raise ValueError("Invalid dual-plane kernel tier or tactic")
                config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
                config = config_type(**entry["config"])
                config.validate_shape(int(m), int(n), int(k))
                self._kernels[key] = config, 0
                continue
            raise ValueError("Unknown or obsolete SM120 kernel family")
        self._grouped = {}
        for key, entry in document.get("grouped_selected", {}).items():
            family, n, k, spans = key.split(":")
            if family != "grouped" or entry.get("kernel") != "grouped_warp_v1":
                raise ValueError("Unknown grouped kernel family")
            config = GroupedConfig.from_dict(entry["config"])
            groups = [part.split("/") for part in spans.split(",")]
            for tier, m in groups:
                if tier not in ("low", "mid", "high"):
                    raise ValueError("Unknown grouped tier")
                getattr(config, tier).validate_shape(int(m), int(n), int(k))
            self._grouped[key] = config
        self._graph = {}
        for key, entry in document.get("graph_selected", {}).items():
            family, spans, m, n, k = key.split(":")
            if (
                family != "graph"
                or int(spans) != 12
                or entry.get("kernel") != "grouped_warp_v1"
            ):
                raise ValueError("Unknown mixed graph kernel or span capacity")
            config = GroupedConfig.from_dict(entry["config"])
            for tier_config in config.tiers:
                tier_config.validate_shape(int(m), int(n), int(k))
            self._graph[key] = config

    def lookup(self, tier: str, m: int, n: int, k: int):
        return self._kernels.get(shape_key(tier, m, n, k))

    def lookup_grouped(self, groups, n: int, k: int):
        return self._grouped.get(grouped_shape_key(groups, n, k))

    def lookup_graph(self, m: int, n: int, k: int, max_spans: int = 12):
        return self._graph.get(graph_shape_key(m, n, k, max_spans))

    def grouped_configurations(self, n: int, k: int):
        return {
            config
            for key, config in (*self._grouped.items(), *self._graph.items())
            if key.startswith(f"grouped:{n}:{k}:")
            or (key.startswith("graph:") and key.endswith(f":{n}:{k}"))
        }

    def configurations(self, tier: str, n: int, k: int):
        """Preload only kernels matching a layer's local matrix dimensions."""
        configurations = set()
        for key in self.entries:
            entry_tier, m, entry_n, entry_k = key.split(":")
            if (entry_tier, int(entry_n), int(entry_k)) == (tier, n, k):
                selected = self.lookup(tier, int(m), n, k)
                if selected is not None:
                    configurations.add(selected)
        return configurations


@cache
def load_database(path: str) -> TacticDatabase:
    return TacticDatabase(json.loads(Path(path).read_text()), fingerprint())
