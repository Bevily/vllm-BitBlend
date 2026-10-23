# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independently verify shape winners against default kernels in paired order."""

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import replace
from functools import partial
from pathlib import Path

from benchmark_bitblend import load_weights, write_json


def verify_warp(args):
    """Recheck exhaustive-search finalists with new inputs and paired timing."""
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        HighWarpConfig,
        MidWarpConfig,
        warp_candidates,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        SCHEMA_VERSION,
        TacticDatabase,
        fingerprint,
        shape_key,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.jit import warp_module
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import quantize

    tier = "mid" if args.mid_warp else "high"
    residual = "p1" if tier == "mid" else "p2"
    config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
    search_path = args.directory / f"{tier}-warp.json"
    document = json.loads(search_path.read_text())
    environment = fingerprint()
    if (
        document.get("status") != "measured"
        or document.get("fingerprint") != environment
        or document.get("kernel") != f"{tier}_warp_v1"
        or len(document["rows"]) != document["planned_shapes"]
    ):
        raise ValueError("Incomplete or incompatible unified dual-plane search")
    with (args.source / "model.safetensors").open("rb") as stream:
        if (
            hashlib.file_digest(stream, "sha256").hexdigest()
            != document["source_sha256"]
        ):
            raise ValueError("Source weights changed since search")
    for row in document["rows"]:
        expected_configs = set(warp_candidates(tier, *row["shape"]))
        for label in row.get("secondary_seeds", []):
            base = config_type(**row["configurations"][label])
            for register in (0, 1):
                for epilogue in (0, 1):
                    expected_configs.add(
                        replace(
                            base, register_prefetch=register, vector_epilogue=epilogue
                        )
                    )
        actual_configs = {config_type(**c) for c in row["configurations"].values()}
        for rejected in row.get("rejected", {}).values():
            cfg = config_type(**rejected["config"])
            info = warp_module(tier, *cfg.variant).resources(cfg.tokens, cfg.channels)
            if (info >> 8) & 255:
                raise ValueError("Rejected configuration has nonzero occupancy")
            actual_configs.add(cfg)
        if expected_configs != actual_configs or any(
            not 0 <= row["relative_l2"].get(label, float("inf")) < 0.001
            or not 0 < row["median_us"].get(label, float("inf")) < float("inf")
            for label in row["configurations"]
        ):
            raise ValueError("Missing or invalid dual-plane parameter measurements")

    torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False
    selected, results = {}, []
    for group, weight, weights, scales, decoded in load_weights(args.source):
        n, k = weight.shape
        scales = {key: value.flatten() for key, value in scales.items()}
        for row in document["rows"]:
            m, rn, rk = row["shape"]
            if (rn, rk) != (n, k):
                continue
            default = row["default_kernel"]
            finalists = sorted(
                row.get("refined", row["configurations"]), key=row["median_us"].get
            )[:3]
            labels = list(dict.fromkeys([default, *finalists]))
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            data, scale = quantize(x, tier)
            qa = quantize_activation(
                x,
                "mxfp4_block32" if tier == "mid" else "mxfp8_block32",
                fp4_scale_rule="ceil",
            )
            expected = (qa @ decoded["p0"].T + qa @ decoded[residual].T).bfloat16()
            scratch = torch.empty(k // 64 * m * n, device="cuda", dtype=torch.float32)
            graphs, errors = {}, {}
            output = torch.empty_like(expected)
            for label in labels:
                cfg = config_type(**row["configurations"][label])
                cfg.validate_shape(m, n, k)
                call = partial(
                    getattr(warp_module(tier, *cfg.variant), f"{tier}_warp"),
                    data,
                    scale,
                    weights["p0"],
                    scales["p0"],
                    weights[residual],
                    scales[residual],
                    output,
                    scratch,
                    cfg.splits,
                    cfg.tokens,
                    cfg.channels,
                )
                for _ in range(3):
                    call()
                error = (output.float() - expected.float()).norm()
                error /= expected.float().norm().clamp_min(1e-20)
                errors[label] = float(error)
                if not bool(torch.isfinite(output).all()) or errors[label] >= 0.001:
                    raise RuntimeError(
                        f"Dual-plane finalist failed correctness: {row['shape']}"
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(4):
                        call()
                graphs[label] = graph
            trials = {label: [] for label in labels}
            for repeat in range(6):
                order = labels if repeat % 2 == 0 else labels[::-1]
                for label in order:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.iterations):
                        graphs[label].replay()
                    end.record()
                    end.synchronize()
                    trials[label].append(
                        start.elapsed_time(end) * 1000 / (4 * args.iterations)
                    )
            medians = {label: statistics.median(t) for label, t in trials.items()}
            winner = min(labels, key=medians.get)
            speedups = [b / c for b, c in zip(trials[default], trials[winner])]
            stable = (
                medians[winner] <= medians[default] * 0.98
                and sum(s > 1 for s in speedups) >= 5
            )
            choice = winner if stable else default
            key = shape_key(tier, m, n, k)
            if key in selected:
                raise ValueError("Duplicate unified dual-plane shape")
            cfg = row["configurations"][choice]
            selected[key] = dict(
                kernel=f"{tier}_warp_v1",
                config=cfg,
                tactic=0,
                median_us=medians[choice],
                relative_l2=errors[choice],
                workspace_bytes=0 if cfg["splits"] == 1 else cfg["splits"] * m * n * 4,
                selection="verified-winner" if stable else "default-within-noise",
                search_best=row["best_kernel"],
            )
            results.append(
                dict(
                    group=group,
                    shape=[m, n, k],
                    selected=choice,
                    default=default,
                    median_us=medians,
                    trials_us=trials,
                    relative_l2=errors,
                    paired_speedups=speedups,
                )
            )
            write_json(args.directory / f"{tier}-verification.json", results)
            print(f"Verified {group} M={m}: {choice}", flush=True)
            del graphs
    if len(selected) != document["planned_shapes"]:
        raise ValueError("Verification did not cover all measured shapes")
    verified = dict(
        schema_version=SCHEMA_VERSION,
        fingerprint=environment,
        status="verified",
        kernel=f"{tier}_warp_v1",
        selected=selected,
        source_sha256=document["source_sha256"],
        search_sha256=hashlib.sha256(search_path.read_bytes()).hexdigest(),
        parameter_space=document["parameter_space"],
        verification=dict(
            seed=20261023,
            repeats=6,
            iterations=args.iterations,
            minimum_median_speedup=1 / 0.98,
            minimum_winning_pairs=5,
        ),
    )
    TacticDatabase(verified, environment)
    write_json(args.directory / "verified-database.json", verified)
    with (args.directory / "verified-shapes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "tier",
                "M",
                "N",
                "K",
                "tokens",
                "channels",
                "splits",
                "block_k",
                "stages",
                "token_warps",
                "matrix_load",
                "register_prefetch",
                "vector_epilogue",
                "median_us",
                "workspace_bytes",
                "selection",
            ]
        )
        for key, entry in selected.items():
            writer.writerow(
                [
                    *key.split(":"),
                    *entry["config"].values(),
                    entry["median_us"],
                    entry["workspace_bytes"],
                    entry["selection"],
                ]
            )
    print(f"Verified {len(selected)} unified dual-plane shapes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--high-warp", action="store_true")
    modes.add_argument("--mid-warp", action="store_true")
    parser.add_argument(
        "--source", type=Path, default=Path("/root/quant/models/Qwen3-0.6B")
    )
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    verify_warp(args)


if __name__ == "__main__":
    main()
