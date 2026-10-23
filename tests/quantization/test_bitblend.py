# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts, reference numerics and optional SM120 kernel correctness.

Also runnable with pytest --noconftest in a minimal development environment.
"""

import asyncio
import pickle

import pytest

from vllm.model_executor.kernels.linear.bitblend import base as backend_module
from vllm.model_executor.kernels.linear.bitblend.base import (
    BackendRequest,
    BitBlendBackend,
    KernelInvocation,
    LayerGeometry,
    PreparedLayer,
    SupportResult,
    register_backend,
    select_backend,
)
from vllm.model_executor.layers.quantization.bitblend.profile import (
    FP8_PROFILE,
    MX_PROFILE,
)
from vllm.model_executor.layers.quantization.bitblend.runtime import (
    ExecutionPlan,
    execution_context,
    get_execution_plan,
    get_workspace,
)


@pytest.mark.parametrize("tier", ("mid", "high"))
def test_sm120_search_covers_tiles_and_every_integer_split(tier):
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        WARP_TILES,
        HighWarpConfig,
        MidWarpConfig,
        warp_candidates,
    )

    config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
    available = warp_candidates(tier, 9, 136, 640)
    baseline = {config_type(t, c, sp) for t, c in WARP_TILES for sp in range(1, 6)}
    assert baseline <= set(available)
    for cfg in available:
        cfg.validate_shape(9, 136, 640)
        assert cfg.shared_bytes(tier) <= 101376
    assert {c.splits for c in available} == set(range(1, 11))
    assert {c.stages for c in available} == set(range(1, 9))
    assert {c.token_warps for c in available} == {1, 2, 4}
    for values in ((7, 64, 1), (8, 64, 0), (8, 64, True)):
        with pytest.raises(ValueError):
            config_type(*values)
    with pytest.raises(ValueError):
        config_type(8, 64, 6).validate_shape(9, 136, 640)


def test_sm120_database_requires_verification_and_matching_environment():
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        TacticDatabase,
    )

    document = dict(
        schema_version=1, fingerprint={"device": "5090"}, status="measured", selected={}
    )
    with pytest.raises(ValueError, match="verified"):
        TacticDatabase(document, document["fingerprint"])
    with pytest.raises(ValueError, match="fingerprint"):
        TacticDatabase({**document, "status": "verified"}, {"device": "GB10"})
    with pytest.raises(ValueError, match="schema"):
        TacticDatabase({**document, "schema_version": 2}, document["fingerprint"])


@pytest.mark.parametrize("matching", (True, False))
def test_sm120_bundled_database_requires_exact_match(monkeypatch, tmp_path, matching):
    import json

    from vllm.model_executor.kernels.linear.bitblend.sm120 import database

    bundled = tmp_path / "rtx5090.json"
    bundled.write_text(json.dumps({"fingerprint": {"device": "5090"}}))
    monkeypatch.setattr(database, "BUNDLED_DATABASE", bundled)
    monkeypatch.setattr(
        database, "fingerprint", lambda: {"device": "5090" if matching else "other"}
    )
    database.resolve_database_path.cache_clear()
    try:
        assert database.resolve_database_path() == (str(bundled) if matching else None)
        # Explicit overrides are not silently replaced, even when incompatible.
        assert database.resolve_database_path("custom.json") == "custom.json"
        bundled.unlink()
        database.resolve_database_path.cache_clear()
        assert database.resolve_database_path() is None
    finally:
        database.resolve_database_path.cache_clear()


def test_sm120_database_matches_exact_shape_without_bucket_extrapolation():
    from dataclasses import asdict

    from vllm.model_executor.kernels.linear.bitblend.sm120.config import MidWarpConfig
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        TacticDatabase,
    )

    config = MidWarpConfig(8, 64, 1)
    db = TacticDatabase(
        dict(
            schema_version=1,
            fingerprint={},
            status="verified",
            selected={
                "mid:4:128:256": {
                    "kernel": "mid_warp_v1",
                    "config": asdict(config),
                    "tactic": 0,
                }
            },
        ),
        {},
    )
    assert db.lookup("mid", 4, 128, 256) == (config, 0)
    assert db.lookup("mid", 3, 128, 256) is None
    assert db.lookup("high", 4, 128, 256) is None
    assert db.lookup("mid", 4, 128, 512) is None
    assert db.configurations("mid", 128, 256) == {(config, 0)}
    assert db.configurations("mid", 256, 256) == set()


def test_sm120_database_rejects_illegal_tactic_before_forward():
    from dataclasses import asdict

    from vllm.model_executor.kernels.linear.bitblend.sm120.config import MidWarpConfig
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        TacticDatabase,
    )

    config = MidWarpConfig(8, 64, 1)
    with pytest.raises(ValueError, match="Invalid dual-plane"):
        TacticDatabase(
            dict(
                schema_version=1,
                fingerprint={},
                status="verified",
                selected={
                    "mid:4:128:256": {
                        "kernel": "mid_warp_v1",
                        "config": asdict(config),
                        "tactic": 4,
                    }
                },
            ),
            {},
        )


def test_sm120_unified_high_database_validates_shape_and_kernel_family():
    from dataclasses import asdict

    from vllm.model_executor.kernels.linear.bitblend.sm120.config import HighWarpConfig
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        TacticDatabase,
    )

    config = HighWarpConfig(16, 128, 3)
    entry = dict(kernel="high_warp_v1", config=asdict(config), tactic=0)
    document = dict(schema_version=1, fingerprint={}, status="verified")
    db = TacticDatabase({**document, "selected": {"high:9:136:384": entry}}, {})
    assert db.lookup("high", 9, 136, 384) == (config, 0)
    assert db.lookup("high", 8, 136, 384) is None
    assert db.lookup("mid", 9, 136, 384) is None
    for key, bad in (
        ("high:9:136:256", entry),
        ("mid:9:136:384", entry),
        ("high:9:136:384", {**entry, "kernel": "unknown"}),
        ("high:9:136:384", {"config": asdict(config), "tactic": 0}),
        ("high:9:136:384", {**entry, "tactic": 1}),
    ):
        with pytest.raises(ValueError):
            TacticDatabase({**document, "selected": {key: bad}}, {})


def _require_sm120():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires an SM120 CUDA GPU")
    pytest.importorskip("flashinfer")
    return torch


@pytest.mark.parametrize("m", (1, 3, 4, 128))
@pytest.mark.parametrize("tier", ("low", "mid", "high"))
def test_sm120_cutedsl_activation_matches_ceil_oracle(m, tier):
    """Decode swizzled scales independently, including unaligned decode rows."""
    torch = _require_sm120()
    from vllm.model_executor.kernels.linear.bitblend.reference import (
        decode_plane,
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import quantize

    x = (
        torch.tensor(
            [0, 0.125, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 6.5, -6.5],
            device="cuda",
            dtype=torch.bfloat16,
        )
        .repeat((m * 128 + 9) // 10)[: m * 128]
        .reshape(m, 128)
    )
    if m > 1:
        x[0].zero_()
    payload, swizzled = quantize(x, tier)
    assert payload.shape == (m, 128 if tier == "high" else 64)
    scales = (
        swizzled.reshape((m + 127) // 128, 1, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1, 4)[:m]
    )
    plane = MX_PROFILE.planes[2 if tier == "high" else 0]
    actual = decode_plane(payload.view(torch.uint8), scales, plane)
    expected = quantize_activation(
        x,
        "mxfp8_block32" if tier == "high" else "mxfp4_block32",
        fp4_scale_rule="ceil",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if tier != "high":
        assert bool((scales[-1] == 128).all())  # ceil(6.5 / 6) -> scale 2


@pytest.mark.parametrize(
    "m,n,k",
    ((1, 64, 128), (3, 72, 256), (9, 128, 1024), (17, 136, 384), (129, 72, 1024)),
)
@pytest.mark.parametrize("splits", (1, 2, 3, 4, 5, 8))
@pytest.mark.parametrize("tier", ("mid", "high"))
def test_sm120_dual_warp_preserves_planes_scales_and_tails(
    m, n, k, splits, tier, pipeline=(128, 2, 1, 0)
):
    """Guard operand-swapped MMA coordinates, scale swizzle and FP32 reduction."""
    torch = _require_sm120()
    if splits > k // 128:
        pytest.skip("Each K split needs at least one complete stage")
    from flashinfer.quantization import block_scale_interleave

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        decode_plane,
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import WARP_TILES
    from vllm.model_executor.kernels.linear.bitblend.sm120.jit import warp_module
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import quantize
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )

    torch.manual_seed(2026)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    if m > 1:
        x[0].zero_()
    w = torch.randn(n, k, device="cuda") * 0.1
    # Distinct row/block scales expose swapped or accidentally shared plane scales.
    w *= torch.exp2(torch.arange(k // 32, device="cuda") % 5).repeat_interleave(32)
    encoded = encode_weight(w, MX_PROFILE)
    residual = "p1" if tier == "mid" else "p2"
    payloads = [encoded[f"weight_{p}"] for p in ("p0", residual)]
    scales = [encoded[f"weight_scale_{p}"] for p in ("p0", residual)]
    d0, d2 = (
        decode_plane(p, s, plane)
        for p, s, plane in zip(
            payloads,
            scales,
            (MX_PROFILE.planes[0], MX_PROFILE.planes[1 if tier == "mid" else 2]),
        )
    )
    qa = quantize_activation(x, MX_PROFILE.tier(tier).activation, fp4_scale_rule="ceil")
    expected = (qa @ d0.T + qa @ d2.T).bfloat16()
    activation, activation_scale = quantize(x, tier)
    output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(
        0 if splits == 1 else splits * m * n, device="cuda", dtype=torch.float32
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        HighWarpConfig,
        MidWarpConfig,
    )

    config_type = MidWarpConfig if tier == "mid" else HighWarpConfig
    extension = warp_module(tier, *pipeline)
    for tokens, channels in WARP_TILES:
        if tokens % (8 * pipeline[2]):
            continue
        if config_type(tokens, channels, splits, *pipeline).shared_bytes(tier) > 101376:
            continue
        if not (extension.resources(tokens, channels) >> 8) & 255:
            continue
        getattr(extension, f"{tier}_warp")(
            activation,
            activation_scale,
            payloads[0],
            block_scale_interleave(scales[0]).flatten(),
            payloads[1] if tier == "mid" else payloads[1].view(torch.float8_e4m3fn),
            block_scale_interleave(scales[1]).flatten(),
            output,
            scratch,
            splits,
            tokens,
            channels,
        )
        assert bool(torch.isfinite(output).all())
        relative = (output.float() - expected.float()).norm() / expected.float().norm()
        assert float(relative) < 0.001


@pytest.mark.parametrize("tier", ("mid", "high"))
@pytest.mark.parametrize(
    "pipeline",
    (
        (64, 1, 1, 0),
        (64, 3, 2, 0),
        (64, 8, 4, 0),
        (128, 3, 2, 0),
        (256, 2, 4, 0),
        (64, 2, 2, 1),
        (64, 3, 2, 0, 1, 1),
        (256, 2, 2, 0, 1, 0),
        (128, 2, 2, 1, 1, 1),
        (64, 1, 4, 0, 0, 1),
    ),
)
def test_sm120_pipeline_wrap_scale_pairs_and_warp_ownership(tier, pipeline):
    """Odd split boundaries and tails expose scale-pair or ring-buffer errors."""
    if tier == "mid" and pipeline[3]:
        pytest.skip("Mid already uses matrix loads")
    test_sm120_dual_warp_preserves_planes_scales_and_tails(
        33, 136, 2304, 3, tier, pipeline
    )


@pytest.mark.parametrize("use_database", (False, True))
def test_sm120_tier_switching_handles_decode_prefill_empty_and_strided_inputs(
    monkeypatch, use_database
):
    """One prepared layer must retain all planes and safely resize eager scratch."""
    torch = _require_sm120()
    from dataclasses import asdict

    from vllm import envs
    from vllm.model_executor.kernels.linear.bitblend.sm120 import (
        database as database_module,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120 import ops
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        HighWarpConfig,
        MidWarpConfig,
    )
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )

    calls = []
    original_gemm = ops.gemm

    def recording_gemm(tier, *args, **kwargs):
        calls.append((tier, kwargs.get("config"), kwargs.get("tactic")))
        return original_gemm(tier, *args, **kwargs)

    if use_database:
        config = MidWarpConfig(8, 128, 2)
        database = database_module.TacticDatabase(
            dict(
                schema_version=1,
                fingerprint={},
                status="verified",
                selected={
                    "mid:1:128:256": {
                        "kernel": "mid_warp_v1",
                        "config": asdict(config),
                        "tactic": 0,
                    },
                    "high:3:128:256": {
                        "kernel": "high_warp_v1",
                        "config": asdict(HighWarpConfig(16, 128, 2)),
                        "tactic": 0,
                    },
                },
            ),
            {},
        )
        monkeypatch.setattr(envs, "VLLM_BITBLEND_SM120_TACTICS", "test-database")
        monkeypatch.setattr(database_module, "load_database", lambda _: database)
        monkeypatch.setattr(ops, "gemm", recording_gemm)

    torch.manual_seed(42)
    layer = torch.nn.Module()
    for name, value in encode_weight(
        torch.randn(128, 256, device="cuda") * 0.1
    ).items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    request = BackendRequest(
        MX_PROFILE,
        LayerGeometry(256, 128, 256, (128,)),
        ("low", "mid", "high"),
        "cuda:0",
        "bfloat16",
    )
    hardware = select_backend("sm120", request)
    prepared = hardware.prepare(layer, request)
    assert type(select_backend("auto", request)) is type(hardware)
    assert hardware.resolve_invocation(
        prepared, MX_PROFILE.tier("low"), (1, 256), graph_mode=True
    ).graph_mode
    with pytest.raises(ValueError, match="local K"):
        hardware.resolve_invocation(prepared, MX_PROFILE.tier("low"), (1, 128))
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.01
    for shape in (
        (0, 256),
        (1, 256),
        (3, 256),
        (32, 256),
        (33, 256),
        (65, 256),
        (128, 256),
        (129, 256),
        (2, 3, 256),
    ):
        x = torch.randn(*shape[:-1], 512, device="cuda", dtype=torch.bfloat16)[..., ::2]
        for tier in reversed(MX_PROFILE.tiers):
            invocation = hardware.resolve_invocation(prepared, tier, tuple(x.shape))
            actual = hardware.run(prepared, x, invocation, bias, None)
            from vllm.model_executor.kernels.linear.bitblend.reference import (
                decode_plane,
                quantize_activation,
            )

            qa = quantize_activation(x, tier.activation, fp4_scale_rule="ceil")
            expected = (
                sum(
                    qa
                    @ decode_plane(
                        getattr(layer, f"weight_{name}"),
                        getattr(layer, f"weight_scale_{name}"),
                        next(
                            plane for plane in MX_PROFILE.planes if plane.name == name
                        ),
                    ).T
                    for name in tier.planes
                ).bfloat16()
                + bias
            )
            assert actual.shape == (*shape[:-1], 128)
            assert actual.dtype == torch.bfloat16
            assert bool(torch.isfinite(actual).all())
            if actual.numel():
                relative = (
                    actual.float() - expected.float()
                ).norm() / expected.float().norm()
                assert float(relative) < 0.01
    if use_database:
        assert ("mid", MidWarpConfig(8, 128, 2), 0) in calls
        assert all(
            (config is None or isinstance(config, HighWarpConfig)) and tactic == 0
            for tier, config, tactic in calls
            if tier == "high"
        )
        assert ("high", None, 0) in calls
        assert ("high", HighWarpConfig(16, 128, 2), 0) in calls


@pytest.mark.parametrize("tier", ("low", "mid", "high"))
def test_sm120_linear_graph_replays_changed_inputs_and_preserves_other_graphs(tier):
    """Capture quantization, GEMM/reduction and bias; replay must read new inputs.

    Independent live graphs guard against unsafe per-layer shared scratch/output
    caches. Strided inputs exercise the contiguous copy inside the captured path.
    """
    torch = _require_sm120()
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.model_executor.layers.quantization.bitblend.linear import (
        BitBlendLinearMethod,
    )

    torch.manual_seed(43)
    layer = torch.nn.Module()
    for name, value in encode_weight(
        torch.randn(128, 256, device="cuda") * 0.1
    ).items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    layer.bitblend_geometry = LayerGeometry(256, 128, 256, (128,))
    method = BitBlendLinearMethod(BitBlendConfig(backend="sm120", tier=tier))
    method.process_weights_after_loading(layer)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.01
    captured = []
    for shape in ((1, 256), (3, 256), (33, 256), (2, 65, 256)):
        x = torch.randn(*shape[:-1], 512, device="cuda", dtype=torch.bfloat16)[..., ::2]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                method.apply(layer, x, bias)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = method.apply(layer, x, bias)
        captured.append((graph, x, output))

    for repeat in range(3):
        for graph, x, output in reversed(captured):
            x.normal_(mean=repeat * 0.1, std=0.5)
            expected = method.apply(layer, x, bias)
            address = output.data_ptr()
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert output.data_ptr() == address


def test_high_uses_alternative_residual_not_all_planes():
    assert MX_PROFILE.tier("high").planes == ("p0", "p2")
    assert MX_PROFILE.planes[1].residual_of == "p0"
    assert MX_PROFILE.planes[2].residual_of == "p0"
    assert MX_PROFILE.tier("mid").activation == "mxfp4_block32"
    assert MX_PROFILE.tier("high").activation == "mxfp8_block32"
    assert FP8_PROFILE.tier("high").planes == ("h0", "h1")
    assert FP8_PROFILE.planes[1].residual_of == "h0"


def test_required_planes_and_checkpoint_shapes():
    planes = MX_PROFILE.required_planes(("low", "high"))
    assert [p.name for p in planes] == ["p0", "p2"]
    assert planes[0].shapes(128, 256) == ((128, 128), (128, 8))
    assert planes[1].shapes(128, 256) == ((128, 256), (128, 8))
    assert FP8_PROFILE.planes[0].shapes(128, 256) == ((128, 256), (128, 1))
    with pytest.raises(ValueError, match="geometry"):
        planes[0].shapes(128, 33)


def test_invalid_profile_tier_cannot_reach_execution():
    with pytest.raises(ValueError, match="not defined"):
        ExecutionPlan(FP8_PROFILE.name, "mid")


def test_context_restores_after_error_and_plan_survives_ipc():
    plan = ExecutionPlan(MX_PROFILE.name, "high")
    assert pickle.loads(pickle.dumps(plan)) == plan
    workspace = object()
    with execution_context(plan, workspace=workspace):
        with (
            pytest.raises(RuntimeError),
            execution_context(ExecutionPlan(MX_PROFILE.name, "low")),
        ):
            raise RuntimeError("forward failed")
        assert get_execution_plan() == plan
        assert get_workspace() is workspace
    assert get_execution_plan() is None
    assert get_workspace() is None


def test_concurrent_tasks_do_not_share_tier():
    async def forward(tier):
        plan = ExecutionPlan(MX_PROFILE.name, tier)
        with execution_context(plan):
            await asyncio.sleep(0)
            return get_execution_plan()

    async def run():
        return await asyncio.gather(forward("low"), forward("high"))

    assert [p.tier for p in asyncio.run(run())] == ["low", "high"]


def test_missing_kernel_fails_instead_of_lowering_precision():
    request = BackendRequest(
        MX_PROFILE,
        LayerGeometry(256, 128, 256, (128,)),
        ("high",),
        "cpu",
        "bfloat16",
    )
    with pytest.raises(NotImplementedError, match="sm120 requires CUDA"):
        select_backend("auto", request)


def test_backend_selection_respects_explicit_name_and_support(monkeypatch):
    monkeypatch.setattr(backend_module, "_backends", {})

    class LowOnly(BitBlendBackend):
        @classmethod
        def supports(cls, request):
            return SupportResult(request.tiers == ("low",), "low tier only")

        def prepare(self, layer, request):
            return PreparedLayer("test", None, request)

        def resolve_invocation(self, prepared, tier, shape, *, graph_mode=False):
            return KernelInvocation(tier, shape, graph_mode=graph_mode)

        def run(self, prepared, x, invocation, bias, workspace):
            raise NotImplementedError

    register_backend("low_only", LowOnly)
    request = BackendRequest(
        MX_PROFILE, LayerGeometry(256, 128, 256, (128,)), ("low",), "cpu", "bfloat16"
    )
    assert isinstance(select_backend("auto", request), LowOnly)
    with pytest.raises(NotImplementedError, match="not registered"):
        select_backend("missing", request)
    high = BackendRequest(
        request.profile, request.geometry, ("high",), request.device, request.dtype
    )
    with pytest.raises(NotImplementedError, match="low tier only"):
        select_backend("auto", high)
    with pytest.raises(ValueError, match="duplicate"):
        register_backend("low_only", LowOnly)


def test_reference_is_explicit_only_even_after_selection(monkeypatch):
    from vllm.model_executor.kernels.linear.bitblend.reference import ReferenceBackend

    monkeypatch.setattr(backend_module, "_backends", {})
    request = BackendRequest(
        MX_PROFILE, LayerGeometry(32, 2, 32, (2,)), ("low",), "cpu", "bfloat16"
    )
    assert isinstance(select_backend("reference", request), ReferenceBackend)
    with pytest.raises(NotImplementedError, match="sm120 requires CUDA"):
        select_backend("auto", request)
    with pytest.raises(ValueError, match="Invalid"):
        register_backend("reference", ReferenceBackend)


@pytest.mark.parametrize("profile", (MX_PROFILE, FP8_PROFILE))
def test_graph_backend_selection_uses_capabilities_and_public_prepared_metadata(
    monkeypatch, profile
):
    from dataclasses import replace
    from types import SimpleNamespace as NS

    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import ReferenceBackend
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.v1.worker.bitblend_runtime import (
        create_graph_workspace,
        execution_context_for_dummy,
    )

    class GraphBackend(ReferenceBackend):
        supports_graphs = True
        supports_mixed_graphs = True

        def create_graph_workspace(self, layers, max_tokens, device):
            assert all(layer.request.profile == profile for layer in layers)
            assert max_tokens == 16 and device == torch.device("cpu")
            return NS(update=lambda plan: updates.append(plan))

    class FixedGraphBackend(GraphBackend):
        supports_mixed_graphs = False

    monkeypatch.setattr(backend_module, "_backends", {})
    register_backend("fixed_graph", FixedGraphBackend)
    register_backend("other_device", GraphBackend)
    request = BackendRequest(
        profile,
        LayerGeometry(64, 8, 64, (8,)),
        tuple(t.name for t in profile.tiers),
        "cpu",
        "bfloat16",
        mixed_batch=True,
        graph_mode=True,
    )
    with pytest.raises(NotImplementedError, match="mixed CUDA Graph"):
        select_backend("fixed_graph", request)
    assert isinstance(
        select_backend("auto", replace(request, mixed_batch=False)), FixedGraphBackend
    )
    backend = select_backend("auto", request)
    assert type(backend) is GraphBackend
    with pytest.raises(NotImplementedError, match="CUDA Graph"):
        select_backend("reference", request)
    prepared = PreparedLayer("other_layout", object(), request)
    layer = NS(bitblend_backend=backend, bitblend_prepared=prepared)
    model = NS(modules=lambda: iter((layer,)))
    updates = []
    workspace = create_graph_workspace(model, 16, torch.device("cpu"))
    config = BitBlendConfig(
        profile=profile.name, backend="other_device", mixed_batch=True
    )
    with execution_context_for_dummy(config, 16, workspace):
        assert get_workspace() is workspace
        assert get_execution_plan().profile == profile.name
        assert get_execution_plan().num_tokens == 16
    assert len(updates) == 1
    assert updates[0].num_tokens == 16
    assert [span.tier for span in updates[0].spans] == [config.tier]
    assert get_workspace() is None


@pytest.mark.parametrize("enforce_eager", (False, True))
def test_linear_loading_checks_requested_graph_mode_before_preparing(enforce_eager):
    from types import SimpleNamespace as NS

    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.model_executor.layers.quantization.bitblend.linear import (
        BitBlendLinearMethod,
    )

    config = NS(
        model_config=NS(enforce_eager=enforce_eager),
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL),
    )
    method = BitBlendLinearMethod(
        BitBlendConfig(profile=FP8_PROFILE.name, backend="reference", mixed_batch=True)
    )
    layer, _ = _reference_layer(FP8_PROFILE)
    layer.bitblend_geometry = LayerGeometry(64, 3, 64, (3,))
    with set_current_vllm_config(config, check_compile=False):
        if enforce_eager:
            method.process_weights_after_loading(layer)
            assert layer.bitblend_prepared.request.mixed_batch
            assert not layer.bitblend_prepared.request.graph_mode
        else:
            with pytest.raises(NotImplementedError, match="CUDA Graph"):
                method.process_weights_after_loading(layer)
            assert not hasattr(layer, "bitblend_prepared")


def test_reference_decodes_all_fp4_codes_and_independent_block_scales():
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import decode_plane

    # Low nibble is the earlier K element, including signed zero and subnormals.
    payload = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 4], dtype=torch.uint8
    )
    scales = torch.tensor([[127, 128]], dtype=torch.uint8)
    expected_codes = (
        torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        .repeat(4)
        .reshape(1, 64)
    )
    expected_codes[:, 32:] *= 2
    torch.testing.assert_close(
        decode_plane(payload, scales, MX_PROFILE.planes[0]), expected_codes
    )
    # UE8M0 byte 0 is a nonzero subnormal scale, not a zero sentinel.
    tiny = decode_plane(
        torch.full((1, 16), 0x22, dtype=torch.uint8),
        torch.zeros((1, 1), dtype=torch.uint8),
        MX_PROFILE.planes[0],
    )
    assert tiny[0, 0].item() == 2.0**-127


@pytest.mark.parametrize(
    "activation", ["mxfp4_block32", "mxfp8_block32", "fp8_per_token"]
)
def test_reference_activation_rounding_scales_and_zero_rows(activation):
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        quantize_activation,
    )

    if activation == "mxfp4_block32":
        source = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        rounded = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    else:
        source = [448.0, 1.0625, 1.1875, -448.0, -1.0625, -1.1875, 0.0, 2.0]
        rounded = [448.0, 1.0, 1.25, -448.0, -1.0, -1.25, 0.0, 2.0]
    row = torch.tensor(source * 4, dtype=torch.bfloat16)
    # Two independent scale blocks, and an all-zero token.
    x = torch.stack((torch.cat((row, row * 2)), torch.zeros(64))).bfloat16()
    result = quantize_activation(x, activation)
    if activation == "fp8_per_token":
        # One scale across K: the maximum is 896, so the channel-profile scale is 2.
        expected_row = (x[0].float() / 2).to(torch.float8_e4m3fn).float() * 2
    else:
        expected_row = torch.tensor(rounded * 4 + [v * 2 for v in rounded] * 4)
    torch.testing.assert_close(result[0], expected_row)
    assert result.dtype == torch.float32
    assert torch.count_nonzero(result[1]) == 0


def test_reference_fp4_rounded_amax_scale_boundary():
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        quantize_activation,
    )

    x = torch.zeros((2, 32), dtype=torch.bfloat16)
    x[:, 0] = torch.tensor([6.5, 7.0])
    x[:, 1] = 1.5
    # At amax=7 the rounded-amax exponent grows: scale 1 -> 2.
    result = quantize_activation(x, "mxfp4_block32")
    torch.testing.assert_close(result[:, :2], torch.tensor([[6.0, 1.5], [8.0, 2.0]]))


def _reference_layer(profile):
    """Known decoded planes with distinct signs, channels and K-block scales."""
    import torch

    layer = torch.nn.Module()
    decoded = {}
    for i, plane in enumerate(profile.planes):
        if plane.bits == 4:
            # Codes +1 and -2, alternating along K.
            payload = torch.full((3, 32), 0xC2, dtype=torch.uint8)
            values = torch.tensor([1.0, -2.0]).repeat(32).expand(3, 64)
        else:
            values = torch.tensor([1.5, -0.5]).repeat(32).expand(3, 64)
            payload = values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        if plane.scale_block:
            scales = (
                torch.tensor([[126, 127], [127, 128], [128, 126]], dtype=torch.uint8)
                + i
            )
            factors = torch.tensor([[0.5, 1.0], [1.0, 2.0], [2.0, 0.5]]) * (2**i)
            weight = values * factors.repeat_interleave(32, dim=-1)
        else:
            scales = torch.tensor([[0.25], [1.0], [2.0]]) * (i + 1)
            weight = values * scales
        layer.register_parameter(
            f"weight_{plane.name}", torch.nn.Parameter(payload, requires_grad=False)
        )
        layer.register_parameter(
            f"weight_scale_{plane.name}",
            torch.nn.Parameter(scales, requires_grad=False),
        )
        decoded[plane.name] = weight
    return layer, decoded


@pytest.mark.parametrize(
    "profile,tier",
    [
        (MX_PROFILE, "low"),
        (MX_PROFILE, "mid"),
        (MX_PROFILE, "high"),
        (FP8_PROFILE, "low"),
        (FP8_PROFILE, "high"),
    ],
)
def test_reference_plane_combination_bias_and_shared_activation(
    profile, tier, monkeypatch
):
    import torch

    import vllm.model_executor.kernels.linear.bitblend.reference as reference

    layer, decoded = _reference_layer(profile)
    request = BackendRequest(
        profile,
        LayerGeometry(64, 3, 64, (1, 2)),
        tuple(t.name for t in profile.tiers),
        "cpu",
        "bfloat16",
    )
    backend = select_backend("reference", request)
    prepared = backend.prepare(layer, request)
    # Exactly representable in every activation format, with independent token scales.
    x = torch.tensor([1.0, -1.0, 0.5, -0.5]).repeat(16).reshape(1, 1, 64)
    x = torch.cat((x, x * 2), dim=1).bfloat16()
    bias = torch.tensor([0.25, -0.5, 1.0], dtype=torch.bfloat16)
    invocation = backend.resolve_invocation(
        prepared, profile.tier(tier), tuple(x.shape)
    )
    calls = []
    original = reference.quantize_activation

    def counted(value, activation):
        calls.append(activation)
        return original(value, activation)

    monkeypatch.setattr(reference, "quantize_activation", counted)
    actual = backend.run(prepared, x, invocation, bias, None)
    expected = sum(x.float() @ decoded[p].T for p in profile.tier(tier).planes)
    torch.testing.assert_close(actual, (expected + bias.float()).bfloat16())
    assert calls == [profile.tier(tier).activation]
    assert actual.shape == (1, 2, 3)
    assert actual.dtype == torch.bfloat16
    with pytest.raises(NotImplementedError, match="eager"):
        backend.resolve_invocation(
            prepared, profile.tier(tier), tuple(x.shape), graph_mode=True
        )


def test_reference_rejects_invalid_weights_and_runtime_contracts():
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import decode_plane

    layer, _ = _reference_layer(MX_PROFILE)
    request = BackendRequest(
        MX_PROFILE, LayerGeometry(64, 3, 64, (3,)), ("low",), "cpu", "bfloat16"
    )
    backend = select_backend("reference", request)
    prepared = backend.prepare(layer, request)
    with pytest.raises(ValueError, match="Tier"):
        backend.resolve_invocation(prepared, MX_PROFILE.tier("high"), (1, 64))
    with pytest.raises(ValueError, match="local K"):
        backend.resolve_invocation(prepared, MX_PROFILE.tier("low"), (1, 32))
    invocation = backend.resolve_invocation(prepared, MX_PROFILE.tier("low"), (1, 64))
    with pytest.raises(ValueError, match="dtype/device/shape"):
        backend.run(prepared, torch.zeros(1, 64), invocation, None, None)
    with pytest.raises(ValueError, match="finite"):
        backend.run(
            prepared,
            torch.full((1, 64), float("nan"), dtype=torch.bfloat16),
            invocation,
            None,
            None,
        )
    with pytest.raises(ValueError, match="Bias"):
        backend.run(
            prepared,
            torch.zeros(1, 64, dtype=torch.bfloat16),
            invocation,
            torch.zeros(4, dtype=torch.bfloat16),
            None,
        )
    layer.weight_scale_p0[0, 0] = 255
    with pytest.raises(ValueError, match="255"):
        backend.prepare(layer, request)
    with pytest.raises(ValueError, match="payload/scale"):
        decode_plane(layer.weight_p0, torch.ones(3, 1), MX_PROFILE.planes[0])
    with pytest.raises(NotImplementedError, match="BF16"):
        select_backend(
            "reference",
            BackendRequest(MX_PROFILE, request.geometry, ("low",), "cpu", "float32"),
        )


@pytest.mark.parametrize("profile", [MX_PROFILE, FP8_PROFILE])
@pytest.mark.parametrize("shape", [(64,), (0, 64), (2, 0, 64)])
def test_reference_preserves_vector_and_empty_shapes_without_bias(profile, shape):
    import torch

    layer, decoded = _reference_layer(profile)
    request = BackendRequest(
        profile, LayerGeometry(64, 3, 64, (3,)), ("low",), "cpu", "bfloat16"
    )
    backend = select_backend("reference", request)
    prepared = backend.prepare(layer, request)
    x = torch.ones(shape, dtype=torch.bfloat16)
    invocation = backend.resolve_invocation(prepared, profile.tier("low"), shape)
    actual = backend.run(prepared, x, invocation, None, None)
    expected = x.float() @ decoded[profile.tier("low").planes[0]].T
    torch.testing.assert_close(actual, expected.bfloat16())


def test_reference_rejects_fp8_nan_bytes_and_invalid_channel_scales():
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import decode_plane

    plane = FP8_PROFILE.planes[0]
    payload = torch.full((1, 5), 0x38, dtype=torch.uint8)  # E4M3FN +1
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive|finite"):
            decode_plane(payload, torch.tensor([[invalid]]), plane)
    payload[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite"):
        decode_plane(payload, torch.ones(1, 1), plane)


@pytest.mark.parametrize("profile", [MX_PROFILE, FP8_PROFILE])
def test_encoded_weights_match_reference_quantization_and_residual(profile):
    import torch

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        decode_plane,
        quantize_activation,
    )
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )

    weight = torch.randn(7, 64, generator=torch.Generator().manual_seed(42))
    weight[0].zero_()
    encoded = encode_weight(weight, profile)
    base = None
    for plane in profile.planes:
        decoded = decode_plane(
            encoded[f"weight_{plane.name}"],
            encoded[f"weight_scale_{plane.name}"],
            plane,
        )
        target = weight if plane.residual_of is None else weight - base
        activation = (
            "fp8_per_token"
            if not plane.scale_block
            else "mxfp4_block32"
            if plane.bits == 4
            else "mxfp8_block32"
        )
        torch.testing.assert_close(
            decoded, quantize_activation(target, activation), rtol=0, atol=0
        )
        if plane.residual_of is None:
            base = decoded


def test_fp4_encoder_packs_earlier_k_in_low_nibble_and_rounds_ties_even():
    import torch

    from vllm.model_executor.layers.quantization.bitblend.checkpoint import encode_plane

    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    weight = torch.cat([values, -values, torch.zeros(16)]).reshape(1, 32)
    payload, scales = encode_plane(weight, MX_PROFILE.planes[0])
    assert scales.tolist() == [[127]]
    assert payload[0, :8].tolist() == [0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE]


def test_checkpoint_encoder_rejects_nonfinite_source_weights():
    import torch

    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )

    with pytest.raises(ValueError, match="finite"):
        encode_weight(torch.full((1, 32), float("nan")))


def test_qwen3_export_is_complete_and_failure_leaves_no_checkpoint(tmp_path):
    import json

    import torch
    from safetensors.torch import load_file, save_file

    from vllm.model_executor.layers.quantization.bitblend.checkpoint import export_qwen3

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "num_hidden_layers": 1, "tie_word_embeddings": True}
        )
    )
    weights = {
        f"model.layers.0.{module}.{projection}_proj.weight": torch.ones(8, 32)
        for module, projections in (
            ("self_attn", ("q", "k", "v", "o")),
            ("mlp", ("gate", "up", "down")),
        )
        for projection in projections
    }
    weights["model.embed_tokens.weight"] = torch.ones(10, 32).bfloat16()
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
    weights["model.layers.0.self_attn.q_norm.weight"] = torch.ones(8).bfloat16()
    save_file(weights, source / "model.safetensors")
    destination = tmp_path / "export"
    manifest = export_qwen3(source, destination)
    assert manifest["quantized_projections"] == 7
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 45
    tensors = {}
    for filename in set(index["weight_map"].values()):
        tensors.update(load_file(destination / filename))
    assert set(tensors) == set(index["weight_map"])
    torch.testing.assert_close(tensors["lm_head.weight"], weights["lm_head.weight"])
    assert tensors["model.layers.0.self_attn.q_norm.weight"].dtype == torch.bfloat16
    with pytest.raises(FileExistsError):
        export_qwen3(source, destination)
    del weights["model.layers.0.self_attn.q_proj.weight"]
    save_file(weights, source / "model.safetensors")
    failed = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="coverage"):
        export_qwen3(source, failed)
    assert not failed.exists()
    assert not list(tmp_path.glob("incomplete.tmp-*"))


def test_mixed_plan_covers_tokens_and_restores_context():
    from vllm.model_executor.layers.quantization.bitblend.runtime import (
        BatchExecutionPlan,
        TierSpan,
    )

    spans = [TierSpan("low", 0, 2), TierSpan("high", 2, 5)]
    plan = BatchExecutionPlan(MX_PROFILE.name, spans)
    spans.clear()
    assert plan.num_tokens == 5
    assert pickle.loads(pickle.dumps(plan)) == plan
    with execution_context(plan):
        assert get_execution_plan() == plan
    assert get_execution_plan() is None
    for invalid in (
        (),
        (TierSpan("low", 1, 2),),
        (TierSpan("low", 0, 2), TierSpan("mid", 1, 3)),
    ):
        with pytest.raises(ValueError):
            BatchExecutionPlan(MX_PROFILE.name, invalid)
    with pytest.raises(ValueError):
        BatchExecutionPlan(FP8_PROFILE.name, (TierSpan("mid", 0, 1),))


def test_request_tier_defaults_and_fixed_mode_rejects_switching():
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig

    fixed = BitBlendConfig(tier="high")
    assert fixed.request_tier(None) == "high"
    assert fixed.request_tier({"bitblend_tier": "high"}) == "high"
    with pytest.raises(ValueError, match="mixed_batch"):
        fixed.request_tier({"bitblend_tier": "low"})
    mixed = BitBlendConfig(mixed_batch=True)
    for tier in ("low", "mid", "high"):
        assert mixed.request_tier({"bitblend_tier": tier}) == tier
    for invalid in (None, 1, "unknown"):
        with pytest.raises(ValueError):
            mixed.request_tier({"bitblend_tier": invalid})


def test_mixed_linear_matches_independent_tiers_with_repeated_spans():
    import torch

    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.model_executor.layers.quantization.bitblend.linear import (
        BitBlendLinearMethod,
    )
    from vllm.model_executor.layers.quantization.bitblend.runtime import (
        BatchExecutionPlan,
        TierSpan,
    )

    method = BitBlendLinearMethod(BitBlendConfig(backend="reference", mixed_batch=True))
    layer, _ = _reference_layer(MX_PROFILE)
    layer.bitblend_geometry = LayerGeometry(64, 3, 64, (3,))
    method.process_weights_after_loading(layer)
    x = torch.randn(2, 3, 64).bfloat16()
    bias = torch.tensor([0.25, -0.5, 1.0]).bfloat16()
    spans = (
        TierSpan("high", 0, 1),
        TierSpan("low", 1, 3),
        TierSpan("mid", 3, 4),
        TierSpan("high", 4, 6),
    )
    expected = []
    for span in spans:
        with execution_context(ExecutionPlan(MX_PROFILE.name, span.tier)):
            expected.append(
                method.apply(layer, x.reshape(-1, 64)[span.start : span.stop], bias)
            )
    with execution_context(BatchExecutionPlan(MX_PROFILE.name, spans)):
        actual = method.apply(layer, x, bias)
        torch.testing.assert_close(
            actual, torch.cat(expected).reshape(2, 3, 3), rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="token plan"):
            method.apply(layer, x[:1])


@pytest.mark.parametrize(
    "unsupported",
    [
        "compile",
        "piecewise",
        "v2",
        "cpu",
        "tp",
        "speculative",
        "multimodal",
    ],
)
def test_mixed_batch_rejects_execution_modes_without_token_plan(unsupported):
    from types import SimpleNamespace as NS

    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.model_executor.layers.quantization.bitblend.config import (
        BitBlendConfig,
        validate_mixed_batch_config,
    )

    config = NS(
        quant_config=BitBlendConfig(mixed_batch=True),
        device_config=NS(device_type="cuda"),
        model_config=NS(
            enforce_eager=True,
            runner_type="generate",
            is_encoder_decoder=False,
            is_moe=False,
            supports_multimodal_inputs=False,
        ),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
        ),
        use_v2_model_runner=False,
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        compilation_config=NS(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL
        ),
    )
    validate_mixed_batch_config(config)
    config.model_config.enforce_eager = False
    config.quant_config.backend = "other_device"
    validate_mixed_batch_config(config)
    if unsupported == "compile":
        config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    elif unsupported == "piecewise":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    elif unsupported == "v2":
        config.use_v2_model_runner = True
    elif unsupported == "cpu":
        config.device_config.device_type = "cpu"
    elif unsupported == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif unsupported == "speculative":
        config.speculative_config = object()
    else:
        config.model_config.supports_multimodal_inputs = True
    with pytest.raises(ValueError, match="BitBlend mixed_batch"):
        validate_mixed_batch_config(config)
    config.quant_config = BitBlendConfig()
    validate_mixed_batch_config(config)


@pytest.mark.parametrize("capacity", [2048, 8192])
@pytest.mark.parametrize("tuned", [False, True])
def test_sm120_mixed_graph_updates_tiers_without_recapture_or_output_aliasing(
    capacity, tuned
):
    """One captured two-layer graph must follow new spans, not capture-time tiers."""
    torch = _require_sm120()
    from dataclasses import asdict, replace

    from vllm.model_executor.kernels.linear.bitblend.sm120.activation import (
        prepare_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        TIERS,
        GroupedConfig,
        HighWarpConfig,
        LowWarpConfig,
        MidWarpConfig,
        grouped_default_config,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        TacticDatabase,
        graph_shape_key,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.graph import (
        MixedGraphWorkspace,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import prepare_grouped
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.model_executor.layers.quantization.bitblend.linear import (
        BitBlendLinearMethod,
    )
    from vllm.model_executor.layers.quantization.bitblend.runtime import (
        BatchExecutionPlan,
        TierSpan,
    )

    torch.manual_seed(824)
    m, k = 160, 256
    layers = [torch.nn.Module(), torch.nn.Module()]
    method = BitBlendLinearMethod(BitBlendConfig(backend="sm120", mixed_batch=True))
    for layer in layers:
        for name, value in encode_weight(
            torch.randn(k, k, device="cuda") * 0.03
        ).items():
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        layer.bitblend_geometry = LayerGeometry(k, k, k, (k,))
        method.process_weights_after_loading(layer)
    config = (
        GroupedConfig(
            low=LowWarpConfig(8, 64, 1),
            mid=MidWarpConfig(16, 64, 2, block_k=64),
            high=HighWarpConfig(
                32, 64, 4, block_k=64, matrix_load=1, vector_epilogue=1
            ),
            waves=2,
            m_major=True,
            heavy_first=True,
            min_blocks=2,
        )
        if tuned
        else replace(grouped_default_config(((None, None, None, m),)), waves=1)
    )
    database = TacticDatabase(
        dict(
            schema_version=1,
            fingerprint={},
            status="verified",
            graph_selected={
                graph_shape_key(m, k, k): dict(
                    kernel="grouped_warp_v1", config=asdict(config)
                )
            },
        ),
        {},
    )
    assert database.lookup_graph(m + 1, k, k) is None
    assert database.lookup_graph(m, k, k, 3) is None
    assert config in database.grouped_configurations(k, k)
    workspace = MixedGraphWorkspace([(k, k)], capacity, "cuda", database=database)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.01

    def plan_for(groups):
        spans, start = [], 0
        for tier, count in groups:
            spans.append(TierSpan(tier, start, start + count))
            start += count
        return BatchExecutionPlan(MX_PROFILE.name, tuple(spans))

    plan = plan_for((("low", m),))
    workspace.update(plan)

    def forward():
        first = method.apply(layers[0], x, bias)
        second = method.apply(layers[1], first, bias)
        return first, second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), execution_context(plan, workspace=workspace):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with execution_context(plan, workspace=workspace), torch.cuda.graph(graph):
        first, second = forward()
    cases = [
        (("high", m),),
        (("low", 3), ("mid", 9), ("high", 129), ("mid", 19)),
        (("mid", m),),
        tuple((TIERS[i % 3], 13 if i < 11 else 17) for i in range(12)),
        (("low", m),),
    ]
    for groups in cases:
        plan = plan_for(groups)
        x.normal_()
        workspace.update(plan)
        workspace.quantized[k].fill_(255)
        workspace.scales[k].fill_(255)
        graph.replay()
        actual = first.clone(), second.clone()
        expected = []
        value = x
        for layer in layers:
            state = layer.bitblend_prepared.state
            quantized = prepare_activation(value, groups).run()
            value = prepare_grouped(
                quantized, state.weights, state.scales, config=config, bias=bias
            ).run()
            expected.append(value)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="capacity"):
        workspace.update(plan_for((("high", capacity + 1),)))


def test_sm120_mixed_linear_matches_same_shape_independent_invocations():
    torch = _require_sm120()
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )
    from vllm.model_executor.layers.quantization.bitblend.config import BitBlendConfig
    from vllm.model_executor.layers.quantization.bitblend.linear import (
        BitBlendLinearMethod,
    )
    from vllm.model_executor.layers.quantization.bitblend.runtime import (
        BatchExecutionPlan,
        TierSpan,
    )

    layer = torch.nn.Module()
    for name, value in encode_weight(
        torch.randn(128, 256, device="cuda") * 0.1
    ).items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    layer.bitblend_geometry = LayerGeometry(256, 128, 256, (128,))
    method = BitBlendLinearMethod(BitBlendConfig(backend="sm120", mixed_batch=True))
    method.process_weights_after_loading(layer)
    x = torch.randn(2, 20, 512, device="cuda", dtype=torch.bfloat16)[..., ::2]
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.01
    spans = (
        TierSpan("low", 0, 3),
        TierSpan("mid", 3, 7),
        TierSpan("high", 7, 39),
        TierSpan("low", 39, 40),
    )
    expected = []
    for span in spans:
        with execution_context(ExecutionPlan(MX_PROFILE.name, span.tier)):
            expected.append(
                method.apply(layer, x.reshape(-1, 256)[span.start : span.stop], bias)
            )
    with execution_context(BatchExecutionPlan(MX_PROFILE.name, spans)):
        actual = method.apply(layer, x, bias)
    torch.testing.assert_close(
        actual, torch.cat(expected).reshape(2, 20, 128), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "waves,splits,advanced",
    [(0, 1, False), (1, 1, False), (2, 2, False), (1, 2, True), (0, 2, True)],
)
def test_sm120_grouped_preserves_local_scales_tails_and_plane_sums(
    waves, splits, advanced
):
    """Catch wrong group pointers, shared-storage reuse and split output offsets."""
    torch = _require_sm120()
    from dataclasses import replace

    from flashinfer.quantization import block_scale_interleave

    from vllm.model_executor.kernels.linear.bitblend.reference import (
        decode_plane,
        quantize_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.config import GroupedConfig
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import (
        prepare_grouped,
        quantize,
    )
    from vllm.model_executor.layers.quantization.bitblend.checkpoint import (
        encode_weight,
    )

    torch.manual_seed(7281)
    n, k = 136, 768 if advanced else 640
    encoded = encode_weight(torch.randn(n, k, device="cuda") * 0.1, MX_PROFILE)
    weights, scales, decoded = {}, {}, {}
    for plane in MX_PROFILE.planes:
        p, s = encoded[f"weight_{plane.name}"], encoded[f"weight_scale_{plane.name}"]
        weights[plane.name] = p if plane.bits == 4 else p.view(torch.float8_e4m3fn)
        scales[plane.name] = block_scale_interleave(s).flatten()
        decoded[plane.name] = decode_plane(p, s, plane)
    config = GroupedConfig(
        waves=waves,
        m_major=bool(waves % 2),
        heavy_first=advanced,
        min_blocks=2 if advanced else 1,
    )
    configs = {}
    for tier in ("low", "mid", "high"):
        c = replace(getattr(config, tier), splits=splits)
        if advanced:
            c = replace(
                c,
                tokens={"low": 16, "mid": 32, "high": 64}[tier],
                block_k={"low": 64, "mid": 128, "high": 256}[tier],
                stages={"low": 3, "mid": 2, "high": 1}[tier],
                token_warps=2,
                matrix_load=int(tier == "high"),
                register_prefetch=1,
                vector_epilogue=1,
            )
        configs[tier] = c
    config = replace(config, **configs)
    groups, expected = [], []
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.01
    for tier, m in (("low", 3), ("mid", 9), ("high", 129), ("mid", 17)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x[0].zero_()
        a, s = quantize(x, tier)
        groups.append((tier, a, s, m))
        qa = quantize_activation(
            x, MX_PROFILE.tier(tier).activation, fp4_scale_rule="ceil"
        )
        value = qa @ decoded["p0"].T
        if tier != "low":
            value += qa @ decoded["p1" if tier == "mid" else "p2"].T
        expected.append(value.bfloat16() + bias)
    call = prepare_grouped(
        groups,
        weights,
        scales,
        config=None if (waves, splits, advanced) == (0, 1, False) else config,
        bias=bias,
    )
    call.output.fill_(float("nan"))
    actual, oracle = call.run(), torch.cat(expected)
    snapshot = actual.clone()
    relative = torch.linalg.vector_norm(
        actual.float() - oracle.float()
    ) / torch.linalg.vector_norm(oracle.float())
    assert torch.isfinite(actual).all() and relative < 0.001
    # Capture only the prepared GEMM/reduction; mixed model graphs remain separate.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call.run()
    call.output.zero_()
    graph.replay()
    torch.testing.assert_close(call.output, snapshot, rtol=0, atol=0)


def test_grouped_search_covers_pipeline_axes_with_common_thread_constraint():
    """Independent best tiles must not form an unlaunchable mixed bundle."""
    from dataclasses import replace

    from vllm.model_executor.kernels.linear.bitblend.sm120.config import (
        GroupedConfig,
        grouped_coordinate_candidates,
        grouped_tier_candidates,
    )

    for tier in ("low", "mid", "high"):
        candidates = list(grouped_tier_candidates(tier, 9, 136, 768, 256))
        assert {c.block_k for c in candidates} == {64, 128, 256}
        assert {c.stages for c in candidates} == set(range(1, 9))
        assert {c.register_prefetch for c in candidates} == {0, 1}
        assert {c.vector_epilogue for c in candidates} == {0, 1}
        assert {c.token_warps for c in candidates} == {1, 2}
        assert {c.matrix_load for c in candidates} == (
            {0, 1} if tier == "high" else {0}
        )
        assert {c.splits for c in candidates if c.block_k == 64} == set(range(1, 13))
        assert all(c.channels * 2 * c.token_warps == 256 for c in candidates)
    base = GroupedConfig()
    neighbors = list(grouped_coordinate_candidates(base.high, "high", 129, 136, 768))
    early = neighbors[:12]
    assert any(c.block_k != base.high.block_k for c in early)
    assert any(c.stages != base.high.stages for c in early)
    assert any(c.register_prefetch for c in early)
    assert any(c.vector_epilogue for c in early)
    assert any(c.matrix_load for c in early)
    assert any(c.splits != 1 for c in early)
    assert {c.splits for c in neighbors} == set(range(1, 7))
    with pytest.raises(ValueError, match="same CTA"):
        replace(base, mid=replace(base.mid, channels=128))


def test_grouped_database_matches_ordered_spans_and_validates_resources():
    from dataclasses import asdict

    from vllm.model_executor.kernels.linear.bitblend.sm120.config import GroupedConfig
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        TacticDatabase,
        grouped_shape_key,
    )

    spans = [("low", 3), ("high", 129), ("low", 1)]
    key = grouped_shape_key(spans, 136, 768)
    config = GroupedConfig(heavy_first=True, min_blocks=3)
    document = dict(
        schema_version=1,
        status="verified",
        fingerprint={},
        grouped_selected={key: dict(kernel="grouped_warp_v1", config=asdict(config))},
    )
    database = TacticDatabase(document, {})
    assert database.lookup_grouped(spans, 136, 768) == config
    assert database.lookup_grouped(list(reversed(spans)), 136, 768) is None
    assert database.lookup_grouped(spans, 136, 640) is None
    document["grouped_selected"][key]["config"]["high"]["stages"] = 8
    document["grouped_selected"][key]["config"]["high"]["tokens"] = 128
    document["grouped_selected"][key]["config"]["high"]["block_k"] = 256
    with pytest.raises(ValueError, match="shared memory"):
        TacticDatabase(document, {})


@pytest.mark.parametrize(
    "k,spans",
    [
        (128, (("low", 1), ("mid", 1), ("high", 1))),
        (640, (("high", 129), ("low", 3), ("mid", 17), ("high", 1))),
        (1024, (("low", 256), ("mid", 256), ("high", 256))),
        (3072, (("mid", 17), ("low", 3))),
        (1024, (("high", 2049),)),
        (1024, (("high", 1025), ("mid", 3), ("high", 1023))),
        (128, (("low", 129),)),
    ],
)
def test_sm120_mixed_activation_matches_official_bytes_and_graph_updates(k, spans):
    """Preserve official rounding, span-local swizzles, padding and live inputs."""
    torch = _require_sm120()
    from vllm.model_executor.kernels.linear.bitblend.sm120.activation import (
        prepare_activation,
    )
    from vllm.model_executor.kernels.linear.bitblend.sm120.ops import quantize

    x = torch.randn(sum(m for _, m in spans), k, device="cuda", dtype=torch.bfloat16)
    call = prepare_activation(x, spans)
    call.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call.run()
    for multiplier in (0.0, 1.0, 448.0, 2.0**-120):
        x.normal_().mul_(multiplier)
        x[:, :8] = (
            torch.tensor(
                [0.0, -0.0, 1.0, -1.0, 6.0, -6.0, 448.0, -448.0],
                device=x.device,
                dtype=x.dtype,
            )
            * multiplier
        )
        for _, a, s, _ in call.groups:
            a.view(torch.uint8).fill_(255)
            s.fill_(255)
        graph.replay()
        start = 0
        for tier, actual, scale, m in call.groups:
            expected, expected_scale = quantize(x[start : start + m], tier)
            torch.testing.assert_close(
                actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(
                scale.flatten(), expected_scale.flatten(), rtol=0, atol=0
            )
            start += m
    for invalid in ([], [("bad", x.shape[0])], [("low", x.shape[0] + 1)]):
        with pytest.raises(ValueError):
            prepare_activation(x, invalid)
