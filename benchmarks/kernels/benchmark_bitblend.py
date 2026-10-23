# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Search SM120 configurations with resource checks and resumable measurements."""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(document, indent=2) + "\n")
    temp.replace(path)


def load_weights(source):
    """Read and fuse real layer-zero Qwen projections in native vLLM order."""
    import torch
    from flashinfer.quantization import block_scale_interleave
    from safetensors import safe_open

    from vllm.model_executor.kernels.linear.bitblend.reference import decode_plane
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )
    from vllm.model_executor.layers.quantization.bitblend.profile import MX_PROFILE

    groups = (
        ("qkv", "self_attn", ("q", "k", "v")),
        ("o", "self_attn", ("o",)),
        ("gate_up", "mlp", ("gate", "up")),
        ("down", "mlp", ("down",)),
    )
    result = []
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
        for name, module, projections in groups:
            weight = (
                torch.cat(
                    [
                        f.get_tensor(f"model.layers.0.{module}.{p}_proj.weight")
                        for p in projections
                    ]
                )
                .cuda()
                .bfloat16()
            )
            encoded = encode_weight(weight, MX_PROFILE)
            weights, scales, decoded = {}, {}, {}
            for plane in MX_PROFILE.planes:
                p, s = (
                    encoded[f"weight_{plane.name}"],
                    encoded[f"weight_scale_{plane.name}"],
                )
                weights[plane.name] = (
                    p if plane.bits == 4 else p.view(torch.float8_e4m3fn)
                )
                scales[plane.name] = block_scale_interleave(s)
                decoded[plane.name] = decode_plane(p, s, plane)
            result.append((name, weight, weights, scales, decoded))
    return result


def time_cuda(fn, repeats, iterations):
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iterations)
    return times


def time_graph(fn, repeats, iterations):
    """Amortize Python/driver submission gaps with repeated work inside a graph."""
    import torch

    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    return [us / iterations for us in time_cuda(graph.replay, repeats, 1)]


def compare_activation(args):
    """Compare official per-span and fused quantization, with and without GEMM."""
    import torch
    from flashinfer.quantization.kernels.mxfp4_quantize import (
        _get_compiled_kernel_mxfp4,
    )
    from flashinfer.quantization.kernels.mxfp8_quantize import (
        _get_compiled_kernel_mxfp8_swizzled,
    )

    from vllm.model_executor.kernels.linear.bitblend.sm120.activation import (
        prepare_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import fingerprint
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import (
        prepare_grouped,
        quantize,
    )

    spans = [(tier, int(m)) for tier, m in (s.split("/") for s in args.grouped_spans)]
    torch.manual_seed(20261023)
    rows = []
    for name, weight, weights, scales, _ in load_weights(args.source):
        if args.grouped_layers and name not in args.grouped_layers:
            continue
        k = weight.shape[1]
        scales = {p: s.flatten() for p, s in scales.items()}
        x = torch.randn(
            sum(m for _, m in spans), k, device="cuda", dtype=torch.bfloat16
        )
        slices, start = [], 0
        for _, m in spans:
            slices.append(x[start : start + m])
            start += m

        def allocate_official(slices=slices):
            return tuple(
                (t, *quantize(part, t), m) for (t, m), part in zip(spans, slices)
            )

        baseline = allocate_official()
        launches = []
        target = torch.cuda.get_device_properties(x.device).multi_processor_count * 4
        for (tier, a, s, m), part in zip(baseline, slices):
            if tier == "high":
                kernel, unit = _get_compiled_kernel_mxfp8_swizzled(
                    "bf16", k, True, m * (k // 32) >= 65536
                )
            else:
                kernel, unit = _get_compiled_kernel_mxfp4(True, k, 0, True, False)
            padded = (m + 127) // 128 * 128
            launches.append(
                (
                    kernel,
                    (
                        part,
                        a.view(torch.uint8),
                        s.flatten(),
                        m,
                        padded,
                        min((padded + unit - 1) // unit, target),
                    ),
                )
            )

        def official(launches=launches):
            for kernel, arguments in launches:
                kernel(*arguments)

        fused = prepare_activation(x, spans)
        fused.run()
        for (_, a, s, _), (_, b, r, _) in zip(fused.groups, baseline):
            torch.testing.assert_close(
                a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(s.flatten(), r.flatten(), rtol=0, atol=0)
        gemm = prepare_grouped(fused.groups, weights, scales)
        baseline_gemm = prepare_grouped(baseline, weights, scales)

        def fused_linear(fused=fused, gemm=gemm):
            fused.run()
            return gemm.run()

        def official_linear(official=official, gemm=baseline_gemm):
            official()
            return gemm.run()

        torch.testing.assert_close(official_linear(), fused_linear(), rtol=0, atol=0)

        def fused_eager(x=x, weights=weights, scales=scales):
            return prepare_grouped(
                prepare_activation(x, spans).run(), weights, scales
            ).run()

        def official_eager(allocate=allocate_official, weights=weights, scales=scales):
            return prepare_grouped(allocate(), weights, scales).run()

        functions = dict(
            official_quantization=official,
            fused_quantization=fused.run,
            official_linear=official_linear,
            fused_linear=fused_linear,
        )
        graphs = {}
        for label, fn in functions.items():
            fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.iterations):
                    fn()
            graphs[label] = graph
        timings = {label: [] for label in graphs}
        for repeat in range(args.repeats):
            order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
            for label in order:
                value = time_cuda(graphs[label].replay, 1, 1)[0] / args.iterations
                timings[label].append(value)
        eager = {
            "official_linear": time_cuda(official_eager, args.repeats, args.iterations),
            "fused_linear": time_cuda(fused_eager, args.repeats, args.iterations),
        }
        row = dict(
            layer=name,
            shape=[x.shape[0], *weight.shape],
            graph_us={label: statistics.median(v) for label, v in timings.items()},
            graph_samples_us=timings,
            eager_including_preparation_us={
                label: statistics.median(v) for label, v in eager.items()
            },
            activation_tasks=fused.tasks,
            activation_blocks=fused.blocks,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_json(
            args.output / "activation.json",
            dict(
                fingerprint=fingerprint(),
                spans=spans,
                repeats=args.repeats,
                iterations=args.iterations,
                rows=rows,
                timing=(
                    "CUDA-event median; alternating graph order, preallocated outputs; "
                    "eager includes allocation and descriptor upload"
                ),
            ),
        )


def compare_grouped(args):
    """Measure mixed real-weight GEMMs; optionally coordinate-search full axes."""
    import time

    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.activation import (
        prepare_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        GroupedConfig,
        HighWarpConfig,
        LowWarpConfig,
        MidWarpConfig,
        grouped_coordinate_candidates,
        grouped_tier_candidates,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        SCHEMA_VERSION,
        fingerprint,
        grouped_shape_key,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import (
        gemm,
        prepare_grouped,
        quantize,
        warp_splits,
        warp_tile,
    )

    spans = []
    for value in args.grouped_spans:
        tier, m = value.split("/")
        if tier not in ("low", "mid", "high") or int(m) < 1:
            raise ValueError("Expected nonempty low/M, mid/M or high/M spans")
        spans.append((tier, int(m)))
    environment = fingerprint()
    source_hash = hashlib.file_digest(
        (args.source / "model.safetensors").open("rb"), "sha256"
    ).hexdigest()
    previous = None
    reference_configs = (
        json.loads(args.reference_configs.read_text())["selected"]
        if args.reference_configs
        else {}
    )
    if args.verify_grouped:
        previous = json.loads(args.verify_grouped.read_text())
        if (
            previous["fingerprint"] != environment
            or previous["source_sha256"] != source_hash
            or previous["status"] != "measured"
            or previous["spans"] != [list(s) for s in spans]
        ):
            raise ValueError("Grouped verification source/environment/spans mismatch")
    report = dict(
        schema_version=SCHEMA_VERSION,
        status="measuring",
        fingerprint=environment,
        source_sha256=source_hash,
        spans=spans,
        selected={},
        grouped_selected={},
        rows=[],
        search="multi-pass coordinate search; not a global optimum",
        candidate_limit_per_axis=args.grouped_search_limit,
        search_passes=args.grouped_search_passes,
        refined_thread_families=args.grouped_search_families,
        timing="iterations captured inside one graph; CUDA event time per GEMM",
        coordinate_minimum_improvement=0.01,
        reference_configs_sha256=(
            hashlib.sha256(args.reference_configs.read_bytes()).hexdigest()
            if args.reference_configs
            else None
        ),
        parameter_space=dict(
            threads=[128, 256, 512, 1024],
            tokens=[8, 16, 32, 64, 128],
            channels=[64, 128],
            block_k=[64, 128, 256],
            stages=list(range(1, 9)),
            token_warps=[1, 2, 4],
            high_matrix_load=[0, 1],
            register_prefetch=[0, 1],
            vector_epilogue=[0, 1],
            splits="every integer from 1 through K/block_k",
            waves=[0, 1, 2, 4],
            m_major=[False, True],
            heavy_first=[False, True],
            min_blocks=[1, 2, 3, 4, 6, 8],
            constraints="common threads, shared <=101376, compiled occupancy >0",
        ),
    )
    path = args.output / ("grouped-verified.json" if previous else "grouped.json")
    if previous:
        for field in (
            "candidate_limit_per_axis",
            "search_passes",
            "refined_thread_families",
            "reference_configs_sha256",
        ):
            report[field] = previous.get(field)
        report["verified_report_sha256"] = hashlib.sha256(
            args.verify_grouped.read_bytes()
        ).hexdigest()
    torch.manual_seed(20261024 if previous else 20261023)
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, weight, weights, scales, decoded in load_weights(args.source):
        if args.grouped_layers and name not in args.grouped_layers:
            continue
        n, k = weight.shape
        scales = {key: value.flatten() for key, value in scales.items()}
        key = grouped_shape_key(spans, n, k)
        groups, inputs, expected = [], [], []
        for tier, m in spans:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            a, s = quantize(x, tier)
            groups.append((tier, a, s, m))
            inputs.append(x)
            qa = quantize_activation(
                x,
                "mxfp8_block32" if tier == "high" else "mxfp4_block32",
                fp4_scale_rule="ceil",
            )
            value = qa @ decoded["p0"].T
            if tier != "low":
                value += qa @ decoded["p1" if tier == "mid" else "p2"].T
            expected.append(value.bfloat16())
        oracle = torch.cat(expected)
        row = dict(
            layer=name,
            shape=[sum(m for _, m in spans), n, k],
            candidates={},
            rejected={},
        )

        def measure(
            config,
            row=row,
            groups=groups,
            weights=weights,
            scales=scales,
            oracle=oracle,
        ):
            label = repr(config)
            if label in row["candidates"]:
                return row["candidates"][label]["median_us"]
            if label in row["rejected"]:
                return float("inf")
            call = prepare_grouped(groups, weights, scales, config=config)
            info = call.module.resources()
            if not (info >> 8) & 255:
                row["rejected"][label] = "zero compiled occupancy"
                return float("inf")
            actual = call.run()
            error = float(
                torch.linalg.vector_norm(actual.float() - oracle.float())
                / torch.linalg.vector_norm(oracle.float())
            )
            if not math.isfinite(error) or error >= 0.001:
                raise AssertionError(
                    f"Grouped candidate failed oracle: {error}, {config}"
                )
            # Kernel-only graph timing excludes descriptor construction and allocation.
            values = time_graph(call.run, args.repeats, args.iterations)
            median = statistics.median(values)
            row["candidates"][label] = dict(
                config=asdict(config),
                median_us=median,
                samples_us=values,
                relative_l2=error,
                registers=info & 255,
                active_blocks=(info >> 8) & 255,
                local_bytes=info >> 16,
            )
            return median

        winner = GroupedConfig()
        if previous:
            winner = GroupedConfig.from_dict(
                previous["grouped_selected"][key]["config"]
            )
        best = measure(winner)
        if args.search_grouped:
            counts = {
                tier: max((m for t, m in spans if t == tier), default=1)
                for tier in ("low", "mid", "high")
            }
            seeds = []
            for threads in (128, 256, 512, 1024):
                channels = 64 if threads <= 512 else 128
                q = threads // (2 * channels)
                seed = GroupedConfig(
                    **{
                        tier: kind(
                            max(8 * q, min(128, 1 << (counts[tier] - 1).bit_length())),
                            channels,
                            1,
                            token_warps=q,
                        )
                        for tier, kind in zip(
                            ("low", "mid", "high"),
                            (LowWarpConfig, MidWarpConfig, HighWarpConfig),
                        )
                    }
                )
                seeds.append((measure(seed), threads, seed))
                historical = {}
                for tier, kind in (("mid", MidWarpConfig), ("high", HighWarpConfig)):
                    entry = reference_configs.get(f"{tier}:{counts[tier]}:{n}:{k}")
                    if entry:
                        c = kind(**entry["config"])
                        if c.channels * 2 * c.token_warps == threads:
                            historical[tier] = c
                if historical:
                    low_splits = min(
                        max(1, historical.get("mid", seed.mid).splits // 2),
                        k // seed.low.block_k,
                    )
                    try:
                        inherited = replace(
                            seed, low=replace(seed.low, splits=low_splits), **historical
                        )
                    except ValueError:
                        pass
                    else:
                        inherited_score = measure(inherited)
                        if inherited_score < seeds[-1][0]:
                            seeds[-1] = (inherited_score, threads, inherited)
            seeds.sort(key=lambda item: item[0])
            for local_best, threads, seed in seeds[
                : args.grouped_search_families or None
            ]:
                local = seed
                for sweep in range(args.grouped_search_passes):
                    before = local
                    for tier in ("low", "mid", "high"):
                        if not any(t == tier for t, _ in spans):
                            continue
                        candidates = (
                            grouped_coordinate_candidates(
                                getattr(local, tier), tier, counts[tier], n, k
                            )
                            if args.grouped_search_limit
                            else grouped_tier_candidates(
                                tier, counts[tier], n, k, threads
                            )
                        )
                        for index, c in enumerate(candidates):
                            if (
                                args.grouped_search_limit
                                and index >= args.grouped_search_limit
                            ):
                                break
                            candidate = replace(local, **{tier: c})
                            score = measure(candidate)
                            if score < local_best * 0.99:
                                local, local_best = candidate, score
                        report["rows"] = [
                            r for r in report["rows"] if r["layer"] != name
                        ] + [row]
                        write_json(path, report)
                        print(
                            f"{name}: threads={threads}, pass={sweep}, "
                            f"tier={tier}, best={local_best:.3f} us",
                            flush=True,
                        )
                    if local == before:
                        break
                if local_best < best:
                    winner, best = local, local_best
        # Control register allocation using CUTLASS-style residency launch bounds.
        for blocks in () if previous else (1, 2, 3, 4, 6, 8):
            if (
                winner.low.channels * 2 * winner.low.token_warps * blocks > 1536
                or max(
                    c.shared_bytes(t)
                    for t, c in zip(("low", "mid", "high"), winner.tiers)
                )
                * blocks
                > 101376
            ):
                continue
            candidate = replace(winner, min_blocks=blocks)
            score = measure(candidate)
            if score < best:
                winner, best = candidate, score
        # Always compare ordinary/persistent grids and both tile traversals.
        for waves in () if previous else (0, 1, 2, 4):
            for order in (False, True):
                for heavy in (False, True):
                    candidate = replace(
                        winner, waves=waves, m_major=order, heavy_first=heavy
                    )
                    score = measure(candidate)
                    if score < best:
                        winner, best = candidate, score
        separate_configs, span_results = [], []
        for span_index, (tier, a, s, m) in enumerate(groups):
            if tier == "low":
                separate_configs.append(None)
                span_results.append(dict(tier=tier, m=m, kernel="cudnn", candidates=[]))
                continue
            kind = MidWarpConfig if tier == "mid" else HighWarpConfig
            default = kind(
                *warp_tile(m, n, k, tier=tier), warp_splits(m, n, k, tier=tier)
            )
            configs = {default, getattr(winner, tier)}
            old = reference_configs.get(f"{tier}:{m}:{n}:{k}")
            if old:
                configs.add(kind(**old["config"]))
            if previous:
                configs = {
                    kind(
                        **next(r for r in previous["rows"] if r["layer"] == name)[
                            "separate_spans"
                        ][span_index]["config"]
                    )
                }
            measurements = []
            for c in sorted(configs, key=repr):
                fn = partial(gemm, tier, a, s, weights, scales, config=c)
                fn()
                us = statistics.median(time_graph(fn, args.repeats, args.iterations))
                measurements.append(dict(config=asdict(c), median_us=us))
            selected = min(measurements, key=lambda r: r["median_us"])
            separate_configs.append(kind(**selected["config"]))
            span_results.append(
                dict(tier=tier, m=m, **selected, candidates=measurements)
            )
        output = torch.empty_like(oracle)

        def separate(
            groups=groups,
            output=output,
            weights=weights,
            scales=scales,
            separate_configs=separate_configs,
        ):
            start = 0
            for (tier, a, s, m), c in zip(groups, separate_configs):
                output[start : start + m].copy_(
                    gemm(tier, a, s, weights, scales, config=c)[:m]
                )
                start += m

        separate()
        baseline_error = float(
            torch.linalg.vector_norm(output.float() - oracle.float())
            / torch.linalg.vector_norm(oracle.float())
        )
        if not math.isfinite(baseline_error) or baseline_error >= 0.001:
            raise AssertionError(f"Separate baseline failed oracle: {baseline_error}")
        baseline = statistics.median(
            time_graph(separate, args.repeats, args.iterations)
        )

        mixed_input = torch.cat(inputs)

        def whole_linear(x=mixed_input, weights=weights, scales=scales, winner=winner):
            quantized = prepare_activation(x, spans).run()
            return prepare_grouped(quantized, weights, scales, config=winner).run()

        def whole_separate(
            inputs=inputs,
            weights=weights,
            scales=scales,
            n=n,
            separate_configs=separate_configs,
        ):
            result = inputs[0].new_empty((sum(m for _, m in spans), n))
            start = 0
            for (tier, m), x, c in zip(spans, inputs, separate_configs):
                a, s = quantize(x, tier)
                result[start : start + m].copy_(
                    gemm(tier, a, s, weights, scales, config=c)[:m]
                )
                start += m
            return result

        whole_linear()
        start = time.perf_counter()
        whole_linear()
        torch.cuda.synchronize()
        whole_wall = (time.perf_counter() - start) * 1e6
        row.update(
            winner=asdict(winner),
            grouped_graph_us=best,
            separate_graph_us=baseline,
            separate_spans=span_results,
            separate_relative_l2=baseline_error,
            grouped_whole_linear_gpu_us=statistics.median(
                time_cuda(whole_linear, args.repeats, args.iterations)
            ),
            grouped_whole_linear_wall_us=whole_wall,
            separate_whole_linear_gpu_us=statistics.median(
                time_cuda(whole_separate, args.repeats, args.iterations)
            ),
        )
        report["grouped_selected"][key] = dict(
            kernel="grouped_warp_v1", config=asdict(winner)
        )
        report["rows"] = [r for r in report["rows"] if r["layer"] != name] + [row]
        write_json(path, report)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("candidates", "rejected")}
            ),
            flush=True,
        )
    if not report["rows"]:
        raise ValueError("No matching real-weight layers")
    report["status"] = "verified" if previous else "measured"
    write_json(path, report)


def compare_warp(args):
    """Compare preallocated kernels with graph replay to bound CPU launch gaps."""
    import importlib.metadata

    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        M_BUCKETS,
        WARP_TILES,
        HighWarpConfig,
        MidWarpConfig,
        warp_candidates,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        SCHEMA_VERSION,
        database_provenance,
        fingerprint,
        load_database,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.jit import (
        source_digest,
        warp_module,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import (
        quantize,
        warp_splits,
        warp_tile,
    )

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    tier = "mid" if args.compare_mid_warp or args.search_mid_warp else "high"
    search = args.search_high_warp or args.search_mid_warp
    database = load_database(str(args.tactics)) if args.tactics else None
    residual = "p1" if tier == "mid" else "p2"
    activation_format = "mxfp4_block32" if tier == "mid" else "mxfp8_block32"
    config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
    report_path = args.output / f"{tier}-warp.json"
    reference_configs = (
        json.loads(args.reference_configs.read_text())["selected"]
        if args.reference_configs
        else {}
    )
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "cuda": torch.version.cuda,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "kernel_sha256": source_digest(),
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(
            (args.source / "model.safetensors").read_bytes()
        ).hexdigest(),
        "measurement": "CUDA events over graph replay, preallocated GEMM only",
        "scope": "same layer-zero weights repeatedly reused; not serving or cold HBM",
        "graph_unroll": 4 if search else 16,
        "mxfp8_tuning": "heuristic" if search else "autotune_v2 before capture",
        "tactic_database": database_provenance(str(args.tactics))
        if args.tactics
        else None,
        "reference_configs": database_provenance(str(args.reference_configs))
        if args.reference_configs
        else None,
        "rows": [],
    }
    if search:
        report.update(
            schema_version=SCHEMA_VERSION,
            fingerprint=fingerprint(),
            status="running",
            kernel=f"{tier}_warp_v1",
            parameter_space={
                "tokens": [8, 16, 32, 64, 128],
                "channels": [64, 128],
                "block_k": [64, 128, 256],
                "stages": list(range(1, 9)),
                "token_warps": [1, 2, 4],
                "matrix_load": [0] if tier == "mid" else [0, 1],
                "splits": "every integer in [1, K/block_k]",
                "pruning": "tile divisibility; shared <=101376; compiled occupancy >0",
                "register_prefetch": [0, 1],
                "vector_epilogue": [0, 1],
                "secondary_search": "cross both switches on top 32 base configs",
                "limits": "staged search; excludes TMA/warp specialization",
            },
            resources={},
        )

    if search and report_path.exists():
        previous = json.loads(report_path.read_text())
        if (
            previous.get("fingerprint") != report["fingerprint"]
            or previous.get("repeats") != args.repeats
            or previous.get("iterations") != args.iterations
            or previous.get("reference_configs") != report["reference_configs"]
            or previous.get("source_sha256") != report["source_sha256"]
            or previous.get("requested_m") != list(args.m or M_BUCKETS)
        ):
            raise ValueError("Cannot resume incompatible search")
        report = previous
    if search:
        report["requested_m"] = list(args.m or M_BUCKETS)
        report["search_method"] = (
            "All candidates: one screening round; top 32 + reference: "
            "repeated refinement; independent verification afterwards"
        )

    def capture(fn):
        for _ in range(0 if search else 3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(report["graph_unroll"]):
                fn()
        return graph

    layers = load_weights(args.source)
    if search:
        report["planned_shapes"] = len(layers) * len(args.m or M_BUCKETS)
    for name, weight, weights, scales, decoded in layers:
        n, k = weight.shape
        scales = {key: value.flatten() for key, value in scales.items()}
        default_ms = M_BUCKETS if search else (1, 8, 32, 128, 512)
        from flashinfer import mm_mxfp8, mxfp8_quantize

        w8, sw8 = mxfp8_quantize(weight, backend="cute-dsl")
        for m in args.m or default_ms:
            if search and any(row["shape"] == [m, n, k] for row in report["rows"]):
                continue
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            activation, scale = quantize(x, tier)
            qa = quantize_activation(x, activation_format, fp4_scale_rule="ceil")
            expected = (qa @ decoded["p0"].T + qa @ decoded[residual].T).bfloat16()
            output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            calls = {
                "bf16": partial(torch.mm, x, weight.T, out=output),
            }
            scratch = torch.empty(k // 64 * m * n, device="cuda", dtype=torch.float32)
            configurations, rejected = {}, {}
            default_cfg = config_type(
                *warp_tile(m, n, k, tier=tier), warp_splits(m, n, k, tier=tier)
            )

            old = reference_configs.get(f"{tier}:{m}:{n}:{k}")
            if old is not None:
                default_cfg = config_type(**old["config"])
                default_cfg.validate_shape(m, n, k)

            def label_for(cfg):
                return "warp-" + "-".join(str(v) for v in asdict(cfg).values())

            configs = (
                list(warp_candidates(tier, m, n, k))
                if search
                else [
                    config_type(t, c, sp)
                    for t, c in (
                        WARP_TILES
                        if args.high_tiles
                        else (warp_tile(m, n, k, tier=tier),)
                    )
                    for sp in (1, 2, 4, 8)
                    if sp <= k // 128
                ]
            )
            if database is not None:
                selected = database.lookup(tier, m, n, k)
                if selected is None:
                    raise ValueError(f"No verified configuration for {tier} {m, n, k}")
                configs = [selected[0]]

            def add_config(
                cfg,
                *,
                calls=calls,
                configurations=configurations,
                rejected=rejected,
                activation=activation,
                scale=scale,
                weights=weights,
                scales=scales,
                output=output,
                scratch=scratch,
            ):
                label = label_for(cfg)
                module = warp_module(tier, *cfg.variant)
                resource_key = str((cfg.tokens, cfg.channels, *cfg.variant))
                if resource_key not in report.get("resources", {}):
                    info = module.resources(cfg.tokens, cfg.channels)
                    report.setdefault("resources", {})[resource_key] = dict(
                        registers=info & 255,
                        active_blocks=(info >> 8) & 255,
                        local_bytes=info >> 16,
                        shared_bytes=cfg.shared_bytes(tier),
                    )
                if report["resources"][resource_key]["active_blocks"] == 0:
                    rejected[label] = dict(
                        config=asdict(cfg), reason="zero compiled occupancy"
                    )
                    return
                configurations[label] = asdict(cfg)
                calls[label] = partial(
                    getattr(module, f"{tier}_warp"),
                    activation,
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

            for cfg in configs:
                add_config(cfg)
            x8, sx8 = mxfp8_quantize(x, backend="cute-dsl")
            calls["mxfp8"] = partial(
                mm_mxfp8,
                x8,
                w8.T,
                sx8.flatten(),
                sw8.flatten(),
                out=output,
                backend="cutlass",
            )
            if not search:
                from flashinfer import autotune_v2

                # Match vLLM warmup; do not compare against an untuned fallback.
                with autotune_v2():
                    calls["mxfp8"]()
            errors, trials = {}, {label: [] for label in calls}
            base_order = list(calls)
            random.Random(2026 + m + n + k).shuffle(base_order)

            expected_float = expected.float()
            expected_norm = expected_float.norm().clamp_min(1e-20)
            error_buffer = torch.empty_like(expected_float)

            def measure(
                labels,
                *,
                calls=calls,
                trials=trials,
                errors=errors,
                expected_float=expected_float,
                expected_norm=expected_norm,
                error_buffer=error_buffer,
                output=output,
                shape=(m, n, k),
            ):
                for batch_start in range(0, len(labels), 48):
                    batch = labels[batch_start : batch_start + 48]
                    graphs = {}
                    for label in batch:
                        trials.setdefault(label, [])
                        fn = calls[label]
                        fn()
                        if label.startswith("warp-"):
                            torch.sub(output, expected_float, out=error_buffer)
                            relative = error_buffer.norm() / expected_norm
                            errors[label] = float(relative)
                            if (
                                not math.isfinite(errors[label])
                                or errors[label] >= 0.001
                            ):
                                raise RuntimeError(
                                    f"Dual-plane numerical failure: {label} {shape}"
                                )
                        graphs[label] = capture(fn)
                    for repeat in range(1 if search else args.repeats):
                        for label in batch if repeat % 2 == 0 else batch[::-1]:
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            start.record()
                            for _ in range(1 if search else args.iterations):
                                graphs[label].replay()
                            end.record()
                            end.synchronize()
                            trials[label].append(
                                start.elapsed_time(end)
                                * 1000
                                / (
                                    (1 if search else args.iterations)
                                    * report["graph_unroll"]
                                )
                            )
                    del graphs

            measure(base_order)
            secondary_seeds = []
            if search:
                secondary_seeds = sorted(
                    configurations, key=lambda label: statistics.median(trials[label])
                )[:32]
                extra_labels = []
                extras = [
                    replace(
                        config_type(**configurations[seed]),
                        register_prefetch=r,
                        vector_epilogue=e,
                    )
                    for seed in secondary_seeds
                    for r in (0, 1)
                    for e in (0, 1)
                    if r or e
                ]
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(
                        pool.map(
                            lambda variant: warp_module(tier, *variant),
                            sorted({cfg.variant for cfg in extras}),
                        )
                    )
                for seed in secondary_seeds:
                    base = config_type(**configurations[seed])
                    for register in (0, 1):
                        for epilogue in (0, 1):
                            if register == epilogue == 0:
                                continue
                            cfg = replace(
                                base,
                                register_prefetch=register,
                                vector_epilogue=epilogue,
                            )
                            add_config(cfg)
                            label = label_for(cfg)
                            if label in configurations:
                                extra_labels.append(label)
                measure(extra_labels)
            screening = {label: values[:] for label, values in trials.items()}
            refined = list(configurations)
            if search:
                refined = sorted(
                    configurations, key=lambda label: statistics.median(trials[label])
                )[:32]
                refined = list(dict.fromkeys([label_for(default_cfg), *refined]))
                graphs = {label: capture(calls[label]) for label in refined}
                for label in refined:
                    trials[label] = []
                for repeat in range(args.repeats):
                    for label in refined if repeat % 2 == 0 else refined[::-1]:
                        values = time_cuda(graphs[label].replay, 1, args.iterations)
                        trials[label].append(values[0] / report["graph_unroll"])
                del graphs
            row = {
                "group": name,
                "shape": [m, n, k],
                "configurations": configurations,
                "default_kernel": label_for(default_cfg)
                if search
                else next(iter(configurations)),
                "rejected": rejected,
                "screening_trials_us": screening,
                "refined": refined,
                "secondary_seeds": secondary_seeds,
                "relative_l2": errors,
                "trials_us": trials,
                "median_us": {
                    label: statistics.median(t) for label, t in trials.items()
                },
            }
            winner = min(refined, key=row["median_us"].get)
            best = configurations[winner]

            def linear_mid_or_high(
                x=x,
                weights=weights,
                scales=scales,
                output=output,
                scratch=scratch,
                best=best,
            ):
                a, sa = quantize(x, tier)
                getattr(
                    warp_module(tier, *config_type(**best).variant), f"{tier}_warp"
                )(
                    a,
                    sa.flatten(),
                    weights["p0"],
                    scales["p0"],
                    weights[residual],
                    scales[residual],
                    output,
                    scratch,
                    best["splits"],
                    best["tokens"],
                    best["channels"],
                )

            def linear_mxfp8(x=x, w8=w8, sw8=sw8, output=output):
                a, sa = mxfp8_quantize(x, backend="cute-dsl")
                mm_mxfp8(
                    a, w8.T, sa.flatten(), sw8.flatten(), out=output, backend="cutlass"
                )

            complete = {
                f"{tier}-linear-best": linear_mid_or_high,
                "mxfp8-linear": linear_mxfp8,
                f"{tier}-quantize": partial(quantize, x, tier),
                "mxfp8-quantize": partial(mxfp8_quantize, x, backend="cute-dsl"),
            }
            row["complete_trials_us"] = {}
            for label, fn in complete.items():
                graph = capture(fn)
                values = time_cuda(graph.replay, args.repeats, args.iterations)
                row["complete_trials_us"][label] = [
                    v / report["graph_unroll"] for v in values
                ]
            row["complete_median_us"] = {
                label: statistics.median(v)
                for label, v in row["complete_trials_us"].items()
            }
            row["best_kernel"] = winner
            if search:
                base_count = sum(
                    not cfg["register_prefetch"] and not cfg["vector_epilogue"]
                    for cfg in configurations.values()
                ) + sum(
                    not row["config"]["register_prefetch"]
                    and not row["config"]["vector_epilogue"]
                    for row in rejected.values()
                )
                if base_count != len(warp_candidates(tier, m, n, k)):
                    raise RuntimeError("Incomplete dual-plane parameter enumeration")
                winner = min(refined, key=row["median_us"].get)
                row["best_kernel"] = winner
                row["candidate_count"] = len(configurations) + len(rejected)
            report["rows"].append(row)
            write_json(report_path, report)
            print(
                f"{name} M={m}: best={row.get('best_kernel', row['default_kernel'])}",
                flush=True,
            )
            del calls
    if search:
        report["status"] = "measured"
        write_json(report_path, report)


def graph_patterns(m, *, held_out=False):
    """Deterministic ratios and repeated attention-region spans summing to M."""
    tiers = ("low", "mid", "high")
    patterns = [(tuple([(tier, m)])) for tier in tiers]
    for weights in (
        (1, 1, 1),
        (8, 1, 1),
        (1, 8, 1),
        (1, 1, 8),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
    ):
        counts = [m * w // sum(weights) for w in weights]
        counts[max(range(3), key=lambda i: weights[i])] += m - sum(counts)
        patterns.append(tuple((t, c) for t, c in zip(tiers, counts) if c))
    count = min(m, 12)
    patterns.append(
        tuple((tiers[i % 3], m // count + (i < m % count)) for i in range(count))
    )
    if held_out:
        rng = random.Random(871 + m)
        for _ in range(6):
            cuts = sorted(rng.sample(range(1, m), min(m, 12) - 1))
            boundaries = [0, *cuts, m]
            patterns.append(
                tuple(
                    (rng.choice(tiers), b - a)
                    for a, b in zip(boundaries, boundaries[1:])
                )
            )
    return list(dict.fromkeys(patterns))


def tune_mixed_graph(args):
    """Tune the actual capture-capacity path over ratios, not exact-span calls."""
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.kernels.linear.bitblend.base import (
        BackendRequest,
        LayerGeometry,
        PreparedLayer,
    )
    from vllm.model_executor.kernels.linear.bitblend.reference import (
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        TIERS,
        GroupedConfig,
        HighWarpConfig,
        LowWarpConfig,
        MidWarpConfig,
        grouped_coordinate_candidates,
        grouped_tier_candidates,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        SCHEMA_VERSION,
        TacticDatabase,
        fingerprint,
        graph_shape_key,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.graph import (
        MixedGraphWorkspace,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.jit import grouped_module
    from vllm.model_executor.layers.quantization.bitblend.profile import MX_PROFILE
    from vllm.model_executor.layers.quantization.bitblend.runtime import (
        BatchExecutionPlan,
        TierSpan,
    )

    environment = fingerprint()
    with (args.source / "model.safetensors").open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    buckets = args.m or [1, 2, 4, 8, 16, 32, 64, 96]
    if len(buckets) != len(set(buckets)):
        raise ValueError("Duplicate graph token buckets")
    previous = None
    path = args.output / "mixed-graph.json"
    if args.verify_mixed_graph:
        previous = json.loads(args.verify_mixed_graph.read_text())
        if (
            previous["fingerprint"] != environment
            or previous["source_sha256"] != source_hash
            or previous["status"] != "measured"
            or previous["buckets"] != buckets
        ):
            raise ValueError("Graph verification source/environment/buckets mismatch")
        path = args.output / "mixed-graph-verified.json"
    report = dict(
        schema_version=SCHEMA_VERSION,
        status="measuring",
        fingerprint=environment,
        source_sha256=source_hash,
        buckets=buckets,
        layers=args.grouped_layers or ["qkv", "o", "gate_up", "down"],
        repeats=args.repeats,
        iterations=args.iterations,
        warmup_replays=100,
        selected={},
        grouped_selected={},
        graph_selected={},
        rows=[],
        max_spans=12,
        search="multi-family coordinate search; not an exhaustive/global optimum",
        search_limit=args.grouped_search_limit,
        search_passes=args.grouped_search_passes,
        search_families=args.grouped_search_families,
        timing="CUDA event per fused quantization + grouped GEMM (+ split reduction); "
        "repeated inside one graph, updated descriptors outside timing",
        objective="geometric mean of equally weighted pattern latencies / baseline; "
        "finalists must also satisfy worst-pattern regression limit",
        max_pattern_regression=args.graph_max_regression,
        scratch_limit_bytes=64 * 1024**2,
        parameter_space=dict(
            threads=[128, 256, 512, 1024],
            tokens=[8, 16, 32, 64, 128],
            channels=[64, 128],
            block_k=[64, 128, 256],
            stages=list(range(1, 9)),
            token_warps=[1, 2, 4],
            high_matrix_load=[0, 1],
            register_prefetch=[0, 1],
            vector_epilogue=[0, 1],
            splits="each integer 1..K/block_k",
            waves=[0, 1, 2, 4],
            m_major=[False, True],
            heavy_first=[False, True],
            min_blocks=[1, 2, 3, 4, 6, 8],
        ),
    )
    if previous:
        report["verified_report_sha256"] = hashlib.sha256(
            args.verify_mixed_graph.read_bytes()
        ).hexdigest()
        for field in ("search_limit", "search_passes", "search_families"):
            report[field] = previous[field]
        report["verification"] = "fresh input seed and six extra random partitions"
    elif path.exists():
        saved = json.loads(path.read_text())
        for field in (
            "fingerprint",
            "source_sha256",
            "buckets",
            "layers",
            "repeats",
            "iterations",
            "warmup_replays",
            "search_limit",
            "search_passes",
            "search_families",
            "max_pattern_regression",
        ):
            if saved[field] != report[field]:
                raise ValueError(f"Resume mismatch: {field}")
        report = saved
        report["status"] = "measuring"
    torch.manual_seed(9251 if previous else 8321)
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, weight, weights, scales, decoded in load_weights(args.source):
        if args.grouped_layers and name not in args.grouped_layers:
            continue
        n, k = weight.shape
        state = SimpleNamespace(
            weights=weights,
            scales={p: s.flatten() for p, s in scales.items()},
        )
        prepared = PreparedLayer(
            "sm120_mx_128x4_v1",
            state,
            BackendRequest(
                MX_PROFILE,
                LayerGeometry(k, n, k, (n,)),
                TIERS,
                str(weight.device),
                "bfloat16",
                mixed_batch=True,
                graph_mode=True,
            ),
        )
        workspace = MixedGraphWorkspace([(k, n)], max(buckets), "cuda")
        for m in buckets:
            key = graph_shape_key(m, n, k)
            if key in report["graph_selected"]:
                continue
            patterns = graph_patterns(m, held_out=previous is not None)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            a4 = quantize_activation(x, "mxfp4_block32", fp4_scale_rule="ceil")
            a8 = quantize_activation(x, "mxfp8_block32", fp4_scale_rule="ceil")
            low = a4 @ decoded["p0"].T
            oracle_by_tier = dict(
                low=low.bfloat16(),
                mid=(low + a4 @ decoded["p1"].T).bfloat16(),
                high=(a8 @ decoded["p0"].T + a8 @ decoded["p2"].T).bfloat16(),
            )
            plans, oracles = [], []
            for pattern in patterns:
                spans, parts, start = [], [], 0
                for tier, count in pattern:
                    spans.append(TierSpan(tier, start, start + count))
                    parts.append(oracle_by_tier[tier][start : start + count])
                    start += count
                plans.append(BatchExecutionPlan("mx_residual_3plane", tuple(spans)))
                oracles.append(torch.cat(parts))
            row = dict(
                layer=name,
                shape=[m, n, k],
                patterns=patterns,
                candidates={},
                rejected={},
            )
            report["rows"] = [r for r in report["rows"] if r["shape"] != [m, n, k]]
            report["rows"].append(row)
            baseline = workspace.config_for(m, n, k)
            row["baseline_us"] = []

            def measure(
                config,
                *,
                fresh=False,
                row=row,
                m=m,
                n=n,
                k=k,
                workspace=workspace,
                plans=plans,
                prepared=prepared,
                x=x,
                oracles=oracles,
                key=key,
            ):
                label = repr(config)
                if not fresh and label in row["candidates"]:
                    return row["candidates"][label]["score"]
                if label in row["rejected"]:
                    return float("inf")
                splits = max(c.splits for c in config.tiers)
                if splits > 1 and m * n * splits * 4 > report["scratch_limit_bytes"]:
                    row["rejected"][label] = "scratch capacity exceeds 64 MiB"
                    return float("inf")
                for c in config.tiers:
                    c.validate_shape(m, n, k)
                # Exclude impossible residency requests before compiling.
                shared = max(c.shared_bytes(t) for t, c in zip(TIERS, config.tiers))
                if shared * config.min_blocks > 101376:
                    row["rejected"][label] = "shared memory residency"
                    return float("inf")
                module = grouped_module(config.variant)
                info = module.resources()
                if not (info >> 8) & 255:
                    row["rejected"][label] = "zero compiled occupancy"
                    return float("inf")
                workspace.configurations[m, n, k] = config
                workspace.update(plans[0])
                workspace.run(prepared, x, None)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.iterations):
                        output = workspace.run(prepared, x, None)
                for _ in range(report["warmup_replays"]):
                    graph.replay()
                times, errors, samples = [], [], []
                order = list(range(len(plans)))
                random.Random(551 + len(row["candidates"])).shuffle(order)
                results = {}
                for i in order:
                    workspace.update(plans[i])
                    graph.replay()
                    error = float(
                        (output.float() - oracles[i].float()).norm()
                        / oracles[i].float().norm().clamp_min(1e-20)
                    )
                    if not math.isfinite(error) or error >= 0.001:
                        raise AssertionError(
                            f"Mixed graph oracle {error}: {key}, {config}"
                        )
                    values = [
                        v / args.iterations
                        for v in time_cuda(graph.replay, args.repeats, 1)
                    ]
                    results[i] = (statistics.median(values), error, values)
                for i in range(len(plans)):
                    t, error, values = results[i]
                    times.append(t)
                    errors.append(error)
                    samples.append(values)
                ratios = [t / b for t, b in zip(times, row["baseline_us"] or times)]
                score = math.exp(statistics.mean(math.log(r) for r in ratios))
                result = dict(
                    config=asdict(config),
                    pattern_us=times,
                    samples_us=samples,
                    relative_l2=errors,
                    score=score,
                    worst_ratio=max(ratios),
                    registers=info & 255,
                    active_blocks=(info >> 8) & 255,
                    local_bytes=info >> 16,
                )
                row["candidates"][label] = result
                return score

            measure(baseline)
            baseline_times = row["candidates"][repr(baseline)]["pattern_us"]
            row["baseline_us"] = baseline_times
            if previous:
                old = next(r for r in previous["rows"] if r["shape"] == [m, n, k])
                finalists = [GroupedConfig.from_dict(v) for v in old["finalists"]]
            else:
                seeds = []
                for threads in (128, 256, 512, 1024):
                    channels = 64 if threads <= 512 else 128
                    q = threads // (channels * 2)
                    tokens = max(
                        q * 8, min(128, 1 << (max(1, m // 3) - 1).bit_length())
                    )
                    seed = GroupedConfig(
                        **{
                            t: kind(tokens, channels, 1, token_warps=q)
                            for t, kind in zip(
                                TIERS, (LowWarpConfig, MidWarpConfig, HighWarpConfig)
                            )
                        },
                        waves=1,
                    )
                    seeds.append((measure(seed), threads, seed))
                seeds.sort(key=lambda item: item[0])
                for score, threads, seed in seeds[
                    : args.grouped_search_families or None
                ]:
                    local = seed
                    for sweep in range(args.grouped_search_passes):
                        before = local
                        for tier in TIERS:
                            candidates = (
                                grouped_coordinate_candidates(
                                    getattr(local, tier), tier, m, n, k
                                )
                                if args.grouped_search_limit
                                else grouped_tier_candidates(tier, m, n, k, threads)
                            )
                            if args.grouped_search_limit:
                                candidates = list(
                                    islice(candidates, args.grouped_search_limit)
                                )
                                variants = [
                                    replace(local, **{tier: c}).variant
                                    for c in candidates
                                ]
                                # Finish compilation before serialized GPU timing.
                                with ThreadPoolExecutor(
                                    max_workers=args.graph_compile_workers
                                ) as pool:
                                    list(pool.map(grouped_module, variants))
                            for c in candidates:
                                candidate = replace(local, **{tier: c})
                                measured = measure(candidate)
                                if measured < score * 0.99:
                                    local, score = candidate, measured
                            write_json(path, report)
                            print(
                                f"{name} M={m} threads={threads} pass={sweep} "
                                f"{tier}: score={score:.4f}",
                                flush=True,
                            )
                        if local == before:
                            break
                ranked = sorted(row["candidates"].values(), key=lambda r: r["score"])
                best = GroupedConfig.from_dict(ranked[0]["config"])
                for blocks in (1, 2, 3, 4, 6, 8):
                    try:
                        measure(replace(best, min_blocks=blocks))
                    except ValueError:
                        continue
                best = GroupedConfig.from_dict(
                    min(row["candidates"].values(), key=lambda r: r["score"])["config"]
                )
                for waves in (0, 1, 2, 4):
                    for order in (False, True):
                        for heavy in (False, True):
                            measure(
                                replace(
                                    best, waves=waves, m_major=order, heavy_first=heavy
                                )
                            )
                ranked = sorted(row["candidates"].values(), key=lambda r: r["score"])
                finalists = [GroupedConfig.from_dict(r["config"]) for r in ranked[:3]]
            finalists = list(dict.fromkeys([baseline, *finalists]))
            row["finalists"] = [asdict(c) for c in finalists]
            # Remeasure the baseline and finalists; verification also uses new
            # inputs and additional ratios, replaying the very same captured graph.
            measure(baseline, fresh=True)
            baseline_times = row["candidates"][repr(baseline)]["pattern_us"]
            row["baseline_us"] = baseline_times
            row["candidates"][repr(baseline)].update(score=1.0, worst_ratio=1.0)
            for c in finalists:
                if c != baseline:
                    measure(c, fresh=True)
            viable = [
                c
                for c in finalists
                if row["candidates"][repr(c)]["worst_ratio"]
                <= 1 + args.graph_max_regression
            ]
            winner = min(viable, key=lambda c: row["candidates"][repr(c)]["score"])
            result = row["candidates"][repr(winner)]
            if result["score"] >= 0.99:
                winner = baseline
                result = row["candidates"][repr(winner)]
            row["winner"] = asdict(winner)
            row["speedup"] = 1 / result["score"]
            row["baseline_us"] = baseline_times
            report["graph_selected"][key] = dict(
                kernel="grouped_warp_v1", config=asdict(winner)
            )
            write_json(path, report)
            print(
                f"DONE {name} M={m}: {len(row['candidates'])} candidates, "
                f"speedup={row['speedup']:.3f}x",
                flush=True,
            )
    report["status"] = "verified" if previous else "measured"
    if previous:
        if report["graph_selected"].keys() != previous["graph_selected"].keys():
            raise ValueError("Verification did not cover the complete graph table")
        TacticDatabase(report, environment)
    write_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/root/quant/models/Qwen3-0.6B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+")
    parser.add_argument(
        "--reference-configs",
        type=Path,
        help="Remeasure old configuration values; never deploy old timings",
    )
    parser.add_argument(
        "--tactics",
        type=Path,
        help="Compare exact verified tactics instead of candidates",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--compare-activation", action="store_true")
    modes.add_argument("--compare-grouped", action="store_true")
    modes.add_argument("--search-grouped", action="store_true")
    modes.add_argument("--search-mixed-graph", action="store_true")
    modes.add_argument("--verify-mixed-graph", type=Path)
    parser.add_argument("--graph-max-regression", type=float, default=0.10)
    parser.add_argument("--graph-compile-workers", type=int, default=1)
    modes.add_argument(
        "--verify-grouped",
        type=Path,
        help="Independently recheck a measured grouped report",
    )
    parser.add_argument(
        "--grouped-spans", nargs="+", default=["low/1", "mid/1", "high/1"]
    )
    parser.add_argument(
        "--grouped-layers", nargs="+", choices=["qkv", "o", "gate_up", "down"]
    )
    parser.add_argument(
        "--grouped-search-families",
        type=int,
        default=0,
        help="Refine the fastest N measured CTA seeds; 0 refines all four",
    )
    parser.add_argument(
        "--grouped-search-passes",
        type=int,
        default=2,
        help="Coordinate sweeps per common CTA family",
    )
    parser.add_argument(
        "--grouped-search-limit",
        type=int,
        default=0,
        help="Per-tier sweep budget across axes; 0 searches the full product",
    )
    for tier in ("mid", "high"):
        modes.add_argument(
            f"--compare-{tier}-warp",
            action="store_true",
            help=f"Compare unified {tier} with MXFP8 and BF16",
        )
        modes.add_argument(
            f"--search-{tier}-warp",
            action="store_true",
            help=f"Search every {tier} tile and integer K split",
        )
    parser.add_argument(
        "--high-tiles",
        "--all-tiles",
        dest="high_tiles",
        action="store_true",
        help="Compare every compiled tile instead of the heuristic tile",
    )
    args = parser.parse_args()
    if args.repeats < 3 or args.iterations < 1 or (args.m and min(args.m) < 1):
        parser.error("Need >=3 repeats, positive iterations and M")
    if args.tactics and (
        args.search_mid_warp or args.search_high_warp or args.high_tiles
    ):
        parser.error("--tactics requires comparison without --all-tiles")
    if args.search_mixed_graph or args.verify_mixed_graph:
        if (
            args.grouped_search_limit < 0
            or args.grouped_search_passes < 1
            or args.grouped_search_families not in range(5)
            or not 0 <= args.graph_max_regression <= 1
            or args.graph_compile_workers < 1
        ):
            parser.error("Invalid graph search budget or regression limit")
        tune_mixed_graph(args)
    elif args.compare_activation:
        compare_activation(args)
    elif args.compare_grouped or args.search_grouped or args.verify_grouped:
        if (
            args.grouped_search_limit < 0
            or args.grouped_search_passes < 1
            or args.grouped_search_families not in range(5)
        ):
            parser.error("Grouped search limit must be nonnegative")
        compare_grouped(args)
    else:
        compare_warp(args)


if __name__ == "__main__":
    main()
