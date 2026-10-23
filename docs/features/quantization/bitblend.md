# BitBlend framework

BitBlend provides residual-plane quantization contracts and a vLLM linear-layer
adapter. An SM120 hardware backend supports fixed-tier MX inference on
RTX 5090; an explicit PyTorch reference backend provides numerical checks.
Eager execution and CUDA Graph replay are supported, including request-fixed
mixed-tier batches on SM120. Adaptive tier selection is pending.
Hardware-specific dependencies are isolated from the generic framework.

## Configuration and checkpoint

The built-in quantization names are `bitblend` and the MX-only compatibility
name `bitblend_mx`. Configuration is read by the standard model loader from
`config.json`'s `quantization_config`, or from `bitblend_config.json`:

```json
{
  "quant_method": "bitblend",
  "schema_version": 1,
  "profile": "mx_residual_3plane",
  "tier": "low",
  "backend": "auto",
  "ignored_layers": []
}
```

`tier` defaults to `low`; `backend` defaults to `auto`. Unsupported tiers,
profiles and schema versions fail validation. `ignored_layers` uses exact
vLLM module names and the model's fused-module mapping. `lm_head` is always
unquantized. Activations/output use BF16. This adapter expects already
quantized weights; it does not quantize an ordinary BF16 checkpoint on load.

| Profile | Planes | Tiers |
| --- | --- | --- |
| `mx_residual_3plane` | P0=MXFP4(W), P1=MXFP4(R), P2=MXFP8(R), R=W−dequant(P0) | Low=P0, Mid=P0+P1, High=P0+P2 |
| `fp8_residual_2plane` | H0=FP8(W), H1=FP8(W−dequant(H0)) | Low=H0, High=H0+H1 |

MX Low/Mid share dynamic MXFP4 activation quantization within each layer call;
High shares dynamic MXFP8 activation quantization. Hopper's proposed profile
uses E4M3, independent FP32 per-output-channel weight scales and shared dynamic
per-token activation quantization. Its hardware kernel is still pending.

For each module, serialized parameters are `weight_<plane>` and
`weight_scale_<plane>`, with lowercase plane names. For logical weight `[N,K]`:

| Plane format | Payload | Scale |
| --- | --- | --- |
| MXFP4 E2M1 | uint8 `[N,K/2]`, adjacent K values packed in low/high nibble | UE8M0 bytes `[N,K/32]` |
| MXFP8 E4M3 | uint8 `[N,K]`, FP8 payload bytes | UE8M0 bytes `[N,K/32]` |
| FP8 E4M3 | uint8 `[N,K]`, FP8 payload bytes | FP32 `[N,1]` |

Checkpoint scales are linear, not pre-swizzled. Parameters use vLLM's V2
weight loader for fused and partitioned linear layers. Qwen3-0.6B TP=1 loading,
including QKV and gate/up fusion, is verified on an RTX 5090; distributed TP
sharding remains unvalidated. All planes are currently allocated and
loaded. A fixed `tier` does not reduce resident plane memory.

## Extension boundaries

- `model_executor/layers/quantization/bitblend/`: configuration, linear adapter,
  immutable profiles, checkpoint conversion and eager execution-plan context.
- `model_executor/kernels/linear/bitblend/base.py`: backend registry, supports/prepare,
  invocation resolution and run interfaces, plus the
  `MixedGraphWorkspace` protocol. `PreparedLayer.request` exposes immutable
  profile/geometry/execution metadata; `state` is private to the backend.
  Register implementations at worker startup using `register_backend`.
  The built-in `sm120` backend is discovered lazily for `auto` or `sm120`.
  `auto` tries registered implementations followed by lazy built-ins, selecting
  the first compatible device/profile/execution capability. Unsupported requests
  fail explicitly, with no precision downgrade.
- `model_executor/kernels/linear/bitblend/reference.py`: explicit-only numerical
  oracle. It is loaded lazily by explicit selection, excluded from `auto`, and
  reserves its name against registry overrides.
- `model_executor/kernels/linear/bitblend/sm120/`: hardware support, prepared weights,
  activation quantization, three-tier dispatch, JIT and tactic database.
- `csrc/quantization/bitblend/sm120/`: CUDA entry points and the shared dual-plane implementation.
  Wheel builds automatically include each architecture directory under its
  corresponding Python kernel package for JIT loading.
- `v1/worker/bitblend_runtime.py`: shared runner adapter.
  Groups already scheduled requests within attention regions and binds token
  plans/workspaces around a forward. Request admission and token/KV scheduling
  use the native vLLM scheduler; tiers execute together in mixed batches.

The maintained execution paths are:

| Mode | Activation quantization | GEMM | Graph storage |
| --- | --- | --- | --- |
| SM120 fixed tier (`mixed_batch=false`) | FlashInfer public CuTe-DSL API | Low: cuDNN; Mid/High: `dual_warp.cuh` specializations | Native PyTorch graph pool |
| SM120 mixed eager | One fused CuTe-DSL launch in `sm120/activation.py` | `grouped.cu`, sharing `dual_warp.cuh` | Per-call allocations |
| SM120 mixed full Graph | Same fused quantizer | Same grouped GEMM | Runner-owned `sm120/graph.py` workspace plus native graph pool |
| Explicit reference | PyTorch numerical oracle | Decoded FP32 plane GEMMs | Eager only |

The fixed-tier entry points and reference oracle remain active tools, not legacy
kernel fallbacks. No alternative same-tier service scheduler is maintained.
`runtime.py` contains only fixed-tier overrides, mixed token spans and their
execution context; graph keys and memory budgets belong to native vLLM.

Offline tools have separate scopes: `benchmark_bitblend.py` and its independent
verifier currently tune SM120 with Qwen projection shapes; they are not generic
hardware tuning tools. The three offline examples handle checkpoint validation,
generation smoke checks and WikiText evaluation. The legacy `bitblend_mx` name
remains a checkpoint compatibility alias, not a second algorithm/backend.

### Adding a hardware backend

1. Choose the checkpoint profile explicitly. B200 can implement the existing MX
   profile; H100's planned `fp8_residual_2plane` has Low/High only and needs its
   own converted checkpoint. `auto` never converts profiles or changes tiers.
2. Add device code under `model_executor/kernels/linear/bitblend/<architecture>/`
   and any JIT sources under `csrc/quantization/bitblend/<architecture>/`.
   Subclass `BitBlendBackend`: `supports` checks device, profile, dtype, geometry
   and dependencies; `prepare` owns packing; `resolve_invocation` selects already
   prepared tactics; `run` performs the linear and owns temporary allocations.
   Return `PreparedLayer(layout, private_state, request)`. Override `run_grouped`
   for a native grouped implementation; the default executes spans separately.
3. Set `supports_graphs` only after fixed-tier capture/replay works. For mixed
   graphs, also set `supports_mixed_graphs` and implement
   `create_graph_workspace`. Its `update(plan)` updates dynamic tier descriptors;
   `run(prepared, x, bias)` uses stable buffers/addresses across replays. Updates
   and forwards use the runner's serialized compute stream. The common runner
   handles request ordering, attention regions, token padding and native graph
   dispatch without knowing the architecture or plane count.
4. Add one lazy import entry in `_BUILTIN_BACKENDS`, or call `register_backend`
   during every worker's startup before model loading. Built-in discovery needs
   no changes to the quantization config, request processor or GPU runner.
5. Validate weight loading, all profile tiers, grouped plans and graph replay on
   the actual target device. Generate an independently verified device-specific
   tactic table; never reuse another device's table or bypass source fingerprints.

The shared config validator checks vLLM execution restrictions; backend selection
checks `BackendRequest.mixed_batch`/`graph_mode` and the backend's Graph flags
before preparing weights. Tests with opaque prepared state and a two-tier profile
cover this boundary, but do not establish B200 or H100 hardware support. Only
SM120 and the explicit reference backend are currently implemented.

`execution_context(plan, workspace=workspace)` binds a runner-owned workspace
and restores the previous context after success or failure
and isolates concurrent Python tasks. Captured graphs additionally require
device metadata updates; changing this context alone does not switch recorded
operations. An unbound adapter uses its configured fixed
tier. A bound plan with a different profile fails rather than reinterpreting
weights. A backend `run` owns activation sharing, independent scale handling,
local-plane combination, output shape and bias. The runner owns the backend
workspace lifetime; allocation happens through `create_graph_workspace`.
Other temporaries use ordinary PyTorch allocations and its graph pool.
SM120 mixed graphs use a
runner-owned sequential workspace; general concurrent slot management is pending.

## Implementation status

The framework contains configuration registration, profile validation, parameter
creation, backend dispatch and execution-plan binding. Native vLLM owns graph
keys and KV memory budgeting. CPU contract tests cover these
device-independent semantics and reference numerics. Separate real-checkpoint
validation covers native Qwen3 loading and CUDA reference execution, as described
in the [validation report](../../bitblend-sm120-validation.md).

### SM120 hardware backend

Select `"backend": "sm120"` for `mx_residual_3plane` on compute capability 12.0,
or use `auto`. Inputs must be BF16, local K divisible by 128 and N by 8.
Low runs FlashInfer/cuDNN A4W4; Mid and High run
fused kernels. No auxiliary dense A8W4 backend is included.
The validated environment uses FlashInfer 0.7.2's bundled CUTLASS 4.5.0,
CUDA toolkit 13.2, a C++ compiler and Ninja. Sources ship inside the Python
wheel and JIT compile during loading; editable installs use the standard
`csrc/quantization/bitblend/sm120/` tree. No old-repository paths are needed.
The first load may take several minutes.

Low/Mid use FlashInfer CuTe-DSL MXFP4 activation quantization; High uses
CuTe-DSL MXFP8. Both directly produce 128x4 scale layouts. Low/Mid now use
ceil-to-power-of-two scales (`amax / 6`), unlike the reference backend's
rounded-amax recipe. High uses ceil scales (`amax / 448`) without the old
CUDA quantizer's row padding. Weight encoding is unchanged. Previous quality
and performance results and tactic databases must be revalidated.
Weight scales are prepared in 128x4 layout once. Mid and High use the shared
dual-plane implementation described below. The backend does not reuse GB10
tactic databases or benchmark during forward.
Offline SM120 search tunes configurations within the unified dual-plane family.
The package includes `sm120/configs/rtx5090.json`, an independently verified
mixed-Graph table with 44 exact-shape entries. It loads automatically only when
the full device/environment/source fingerprint matches; otherwise default
configurations are used. Missing shapes also use defaults.
Set `VLLM_BITBLEND_SM120_TACTICS` to override the bundled table with another
independently verified database. Explicit measured-only or incompatible tables
fail loading. See
[the shape database guide](../../bitblend-sm120-database.md) for the search,
verification, timing scope and reproducible commands.
Bias is added to the BF16 GEMM result, so its rounding can differ from the
reference backend's FP32 bias addition.

On RTX 5090, every Mid and High GEMM uses its specialization of one shared
dual-plane kernel template. Mid uses native packed FP4/FP4 `m16n8k64` MMA
and `ldmatrix`, sharing activation loads across P0/P1. It computes
A4×P0 + A4×P1 with independent MX block-32 scales. High
uses CUTLASS native SM120 mixed F4/F8 and F8/F8 blockscaled MMA, an
operand-swapped tile with 8/16/32/64/128 tokens and 64/128 channels,
configurable asynchronous staging and
integer split-K with FP32 reduction. P0 remains packed FP4, each plane retains
its own MX scales, and High still computes A8×P0 + A8×P2. Activation quantization
continues to use FlashInfer CuTe-DSL. Tile selection happens within this one
implementation. Offline search covers ten tile pairs, K stages 64/128/256,
1–8 shared buffers, token warp groups 1/2/4, every legal integer split and
High's shared-gather/`ldmatrix` choice. Register prefetch and vector epilogue
switches are crossed on the fastest 32 screened base configurations, followed
by repeated timing and independent verification. The vector epilogue reuses
stage storage. Resource screening queries compiled occupancy and records spills.
All selected specializations load before graph capture; forward never tunes.
For shapes absent from the verified table, the default heuristic selects
narrow tiles for small M, `(64,64)` for intermediate shapes, and `(128,128)`
for High at M>=1024 and N>=2048. Mid uses `(128,64)` at M>=512 and
N>=2048. This rule is validated on Qwen3-0.6B shapes and does
not guarantee the fastest candidate on other models.
Verified entries tagged `mid_warp_v1` or `high_warp_v1` can override the default configuration
through `VLLM_BITBLEND_SM120_TACTICS`. Lookup matches exact M/N/K and checks the
device, dependency versions and source fingerprint. Missing shapes use the
same template's heuristic. Use
`benchmark_bitblend.py --search-mid-warp` (or `--search-high-warp`) and independent verification as
described in the [shape database guide](../../bitblend-sm120-database.md).
GEMM speedups do not imply an equivalent end-to-end serving speedup or a win
over BF16 at every shape.

Native Qwen3-0.6B loading and 336 linear checks passed on RTX 5090.
Prepared-layer tests cover tier switching, empty and noncontiguous inputs,
padding, bias and activation rounding. Fixed-tier CUDA Graph capture includes
activation quantization, GEMM/reduction and bias. PyTorch's graph memory pool
keeps temporary addresses stable during replay; no per-layer scratch cache or
separate BitBlend graph manager is needed. Temporary memory still contributes
to peak usage. All canonical planes remain
resident alongside swizzled scales. Distributed TP and production performance
remain unvalidated. See the [SM120 validation report](../../bitblend-sm120-validation.md).

### Standard HTTP serving

Fixed-tier OpenAI-compatible HTTP serving is verified on RTX 5090 with the
Qwen3-0.6B MX checkpoint, TP=1 and prefix caching disabled.
Each server uses one tier for all requests, with the original vLLM scheduler.
Chat/completion endpoints, SSE streaming, eight concurrent completion requests,
concurrent streaming/nonstreaming chat and restart checks passed. This is a
functionality check, not a throughput or long-duration stability benchmark.

The exporter writes `backend=reference` and `tier=low`. Override both at startup
without changing the checkpoint. In the validated CUDA/FlashInfer environment:

```bash
vllm serve /path/to/Qwen3-0.6B-BitBlend-MX \
    --served-model-name bitblend-qwen3 --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enforce-eager --max-model-len 512 \
    --max-num-seqs 4 --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.4 --no-enable-prefix-caching \
    --attention-config '{"backend":"TRITON_ATTN"}' \
    --generation-config vllm --seed 42 --shutdown-timeout 10 \
    --hf-overrides '{"quantization_config":{"quant_method":"bitblend","schema_version":1,"profile":"mx_residual_3plane","backend":"sm120","tier":"high"}}'
```

Change `tier` to `low`, `mid` or `high`. A compatible bundled table loads
automatically; set `VLLM_BITBLEND_SM120_TACTICS` before launch to use a custom
verified table. For Qwen3 chat requests, use
`"chat_template_kwargs":{"enable_thinking":false}` to reproduce the offline
smoke checks; `"stream":true` enables SSE. See the
[validation report](../../bitblend-sm120-validation.md) for local artifacts and
the default immediate-shutdown behavior observed during validation.

### Mixed-tier batches

Opt in with `"mixed_batch": true` in the BitBlend quantization configuration.
The configuration's `tier` remains the default. Set
`VLLM_USE_V2_MODEL_RUNNER=0`. For example, extend the `--hf-overrides` above
with `"mixed_batch":true`. Keep `--enforce-eager` for eager execution, or replace
it with the following to capture the whole model for both prefill and decode:

```bash
--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'
```

`FULL_DECODE_ONLY` is also supported. Mixed graphs require the SM120 backend
and compilation mode 0; Dynamo and piecewise compilation are not supported.
Graph eligibility and token padding follow the native attention backend and
capture-size dispatcher. Batches outside its captured range execute eagerly.
Prefix caching can be enabled: the first block hash includes the resolved
request tier, and subsequent blocks inherit that identity through parent hashes.
The caller's `cache_salt` is preserved.

Python requests use the existing sampling extension:

```python
params = SamplingParams(
    temperature=0, max_tokens=32, extra_args={"bitblend_tier": "mid"}
)
```

OpenAI-compatible completion and chat requests use the existing field:

```json
{"vllm_xargs": {"bitblend_tier": "high"}}
```

Valid MX tiers are `low`, `mid` and `high`; an omitted tier uses the configured
default. Invalid tiers fail request validation. Fixed-tier servers reject requests
that select another tier. Each request retains its tier for its entire lifetime.
Admission normalizes a private copy of the sampling parameters, without mutating
caller-owned parameters or adding engine RPC fields.

The native scheduler retains admission, token/KV budgets, priorities, chunked
prefill, preemption and request completion. After attention's native reordering,
`InputBatch.swap_states` groups requests by tier while preserving any required
decode/extend/prefill regions. Token buffers, KV block tables, sampling state and
async output mappings follow the existing batch machinery. A per-forward
`BatchExecutionPlan` describes contiguous token spans; a tier may appear in
multiple spans when attention requires region separation.

One model forward processes the batch. On SM120, mixed-batch linears launch one
CuTe DSL activation quantization kernel, then one heterogeneous grouped GEMM.
The quantizer reuses FlashInfer's official BF16 vector-load, UE8M0 ceil-scale,
FP4/FP8 conversion and scale-swizzle primitives. Each CTA processes one format:
Low/Mid use one thread per 32-element FP4 block; High uses four cooperating
threads, or two when the combined High input has at least 65,536 scale blocks.
Padding rows are processed in parallel. No shared-memory staging is needed.
Low uses one FP4 plane, Mid two FP4 planes, and High mixed FP4/FP8 planes.
All groups write directly to one BF16 output. Repeated spans retain their own
local activation-scale origin. Single-span mixed batches use this same grouped
family; fixed-tier execution and its existing CUDA Graph path remain separate.
The explicit reference backend retains independent per-span execution.

The grouped kernel shares the fixed-tier SM120 tile routines. It supports
ordinary grids and CUTLASS-style round-robin persistent grids. Each tier can
select its own token/channel tile, K block, pipeline depth, warp groups, register
prefetch and vector epilogue; High also selects shared loads or `ldmatrix`.
All tier configurations must use the same CTA thread count. Shared storage is
reused between jobs, with the largest tier's allocation reserved for the launch.
Scheduling also supports heavy-first group ordering and a tunable compiler
residency target. Host-precomputed CUTLASS dividers reduce task-mapping overhead;
the output retains its original token order for either group ordering.
Split-K uses FP32 scratch and one additional grouped reduction. Bias preserves
the existing BF16-result-then-BF16-bias rounding order.

`benchmark_bitblend.py --compare-grouped` compares preallocated graph-replayed
GEMM calls and also reports the whole eager linear including quantization,
allocations and descriptor upload. `--search-grouped` coordinate-searches all
legal per-tier axes in each common thread family; it does not exhaust the joint
Cartesian product. `--grouped-search-limit` bounds an exploratory search.
Finite budgets interleave parameter axes; `--grouped-search-passes` controls
repeated sweeps. Separate Mid/High baselines remeasure default, selected grouped,
and optional historical configurations (`--reference-configs`), including copies
in the complete baseline timing.
`--verify-grouped REPORT` rechecks numerical accuracy with fresh inputs before
producing a verified database accepted by `VLLM_BITBLEND_SM120_TACTICS`.
Lookup matches N/K and the complete ordered tier/token-span list. Unlisted
patterns use four precompiled configurations in the same grouped family.
The largest span selects token tiles 8/32/64/128 at boundaries 8/32/128;
all use 64 channels, K=128, two stages and no Split-K. The 128-token tile uses
two token warp groups. These bounded defaults avoid using a decode tile for
large prefill, but do not replace exact-pattern offline tuning.

The quantizer keeps independent payload and scale slices for every span,
including repeated tiers, while handling FP4 and FP8 in one launch. Its
`prepare_activation` interface owns preallocated outputs and a device task
descriptor; `run` is allocation free and supports graph replay with new inputs.
K and High's cooperation size select the compiled variant; span counts and M
stay dynamic. Both variants are compiled during model loading. Quantization
remains separate from GEMM.

For whole-model graphs, the runner owns a `MixedGraphWorkspace`. Each distinct
K shares reusable quantized payload/scale storage across sequential layers;
each K/N geometry has stable GEMM descriptors. Before each forward, one pinned
H2D copy updates all descriptors with the current tier ranges. Both kernels
read the actual task count from those descriptors, bounded by captured launch
capacities. Empty trailing descriptors carry no work. At most twelve spans
cover the native four attention regions and three tiers; graph padding extends
the final span. GEMM outputs remain separate graph-pool allocations so residual
tensors are not overwritten by workspace reuse. Graph count depends on native
batch descriptors, not tier ratios; capture with Low also replays Mid/High.

Graph linears look up `graph_selected` entries in the matching bundled table or
the `VLLM_BITBLEND_SM120_TACTICS` override by exact padded token count and local N/K.
One configuration covers all tier ratios in that bucket; eager exact-pattern
entries are not applied. Unlisted buckets/geometries use the bounded default
tile family, one persistent wave and no Split-K. Tuned entries can select
independent tier tiles, pipelines, Split-K, traversal and persistent-grid settings.
Split-K scratch comes from the graph pool and is bounded by the bucket's maximum
tier split count, even when that tier is absent in a particular replay.
An eager fallback in the same graph-enabled runner uses this same workspace
and tile policy. With `--enforce-eager`, the existing exact-pattern database path
remains available. Source changes invalidate historical databases as usual.
Use `benchmark_bitblend.py --search-mixed-graph` followed by
`--verify-mixed-graph REPORT` to generate a compatible graph table. See the
[database guide](../../bitblend-sm120-database.md#whole-model-graph-tuning).

This first stage supports the V1 GPU runner, single-GPU dense decoder-only text
generation in eager or native full-graph mode, including async scheduling.
Distributed execution, speculative decoding, LoRA, MoE, multimodal input,
resumable sessions and KV/encoder transfer are explicitly rejected. Fixed-tier
CUDA Graph support remains available separately. Shape/tactic and attention
reduction changes can alter numerical results when batch composition changes;
this mode does not promise batch-invariant generation. See the validation report
for controlled comparisons and broader cases that produced different tokens.

### Fixed-tier CUDA Graphs

To enable CUDA Graphs, replace `--enforce-eager` in the fixed-tier command above with:

```bash
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Python API users can pass the same dictionary as `compilation_config` and leave
`enforce_eager=False`. `mode=0` disables `torch.compile`, independently of CUDA
Graphs. The SM120 backend does not yet support Dynamo tracing; simply removing
`--enforce-eager` would also enable the default compilation pipeline.

The original vLLM runner warms each capture shape, owns graph pools, pads batches
and dispatches replay. `FULL_DECODE_ONLY` captures decode and leaves prefill eager;
`FULL` also allows graph execution for compatible prefill/mixed batches. Optional
`cudagraph_capture_sizes` follows normal vLLM configuration; shapes outside the
captured range run eager with the same kernels. Direct backend users must warm up
each shape before capture, including FlashInfer's quantizer and Low GEMM.

One server retains a fixed tier. A captured graph freezes its tier, tensors and
tactics: changing a Python execution context does not switch the recorded GPU
operations. Mixed-tier execution uses the request tier described above;
its graph workspace and tuning policy are described separately above.
Compatible verified fixed-tier tactic tables work in either fixed-tier execution mode;
strict source fingerprints still apply after backend changes, requiring
revalidation rather than bypassing the database check.

### Reference backend

Set `"backend": "reference"` in the quantization configuration to use the
PyTorch oracle through the existing linear adapter. It expects the same packed,
already-quantized checkpoint parameters; it does not convert BF16 checkpoints.
Both profiles and all their tiers are implemented. Weight decoding validates
local shapes, payload/scale dtypes, devices and finite values. UE8M0 byte 255
is rejected; byte 0 denotes `2**-127`. FP8 channel scales must be positive.

Each call quantizes/dequantizes the BF16 input once, computes each selected
plane's GEMM in FP32 with independent weight scales, sums the contributions,
adds optional BF16 bias once, and returns BF16. Leading input dimensions and
empty token batches are preserved. Activation rules are explicit:

- MXFP4: block32, vLLM's rounded-amax E8M0 rule from `mxfp4_utils.py`
  (`((amax_bits + 0x200000) & 0x7f800000)`, exponent minus 2, clamped
  to [-127,127]), nearest-even E2M1 payload rounding with saturation.
- MXFP8: block32, `ceil(log2(amax/448))` clamped to [-127,127],
  saturated E4M3FN payload rounding.
- FP8 per token: dynamic `amax/448` FP32 scale, floored at FP32's smallest
  normal value; all-zero tokens use scale 1. Payload uses saturated E4M3FN.

The oracle uses ordinary CPU/CUDA PyTorch operations and retains decoded FP32
weights alongside checkpoint parameters. It allocates temporary tensors and
synchronizes for validation. Graph mode is explicitly rejected. Compile,
Tensor Core accumulation equivalence, performance, distributed TP collectives
and serving under concurrent load remain unvalidated. Native Qwen3-0.6B loading,
all three MX tiers on CUDA, and fixed-tier eager generation smoke checks have
passed. This backend is intended for correctness checks, not performance
measurements.

Run the CPU tests with an environment containing PyTorch, pytest and the
minimal vLLM import dependencies:

```bash
.venv/bin/python -m pytest --noconftest tests/quantization/test_bitblend.py -q
```

### Offline Qwen3 export and loading validation

The exporter accepts an unquantized dense Qwen3 directory containing a single
`model.safetensors` file. It writes a separate checkpoint, preserves embeddings,
norms and the unquantized output head, and encodes all seven projection matrices
per layer. The source checkpoint is unchanged; existing output directories are
rejected. A failed export removes its temporary output. MX P1 and P2 both encode
the residual from decoded P0. Output includes a standard safetensors index,
schema-version-1 quantization configuration, and source SHA-256 provenance.
The exported backend is explicitly `reference`.

```bash
.venv/bin/python -m vllm.model_executor.layers.quantization.bitblend.checkpoint \
    /path/to/Qwen3-0.6B /path/to/Qwen3-0.6B-BitBlend-MX --device cuda
.venv/bin/python examples/offline_inference/bitblend_validate.py \
    --source /path/to/Qwen3-0.6B \
    --checkpoint /path/to/Qwen3-0.6B-BitBlend-MX --report /tmp/loading.json
.venv/bin/python examples/offline_inference/bitblend_smoke.py \
    --model /path/to/Qwen3-0.6B-BitBlend-MX --tier high --backend sm120 \
    --report /tmp/smoke.json
```

The validation script uses the native `DefaultModelLoader` and
`Qwen3ForCausalLM`, checks checkpoint coverage and fused shard placement,
regenerates planes from the source weights, and compares every linear layer at
every tier with the oracle. It requires a working CUDA vLLM installation and
runs with TP=1. The generation script also accepts the source BF16 model when
`--tier` is omitted. Both scripts disable compilation; generation uses Triton
attention and disables prefix caching. Two prompts are only a smoke check.

For quality evaluation, `examples/offline_inference/bitblend_eval.py` reuses
lm-evaluation-harness's WikiText task through its vLLM adapter. Run the source
BF16 model without `--tier`, and the BitBlend checkpoint with each tier and
`--backend sm120`, keeping all other settings identical. Reports contain word
perplexity, the harness configuration and dependency versions. See
[validation report](../../bitblend-sm120-validation.md) for commands and
the distinction between limited checks and full test-set results.

The following are **not implemented**:

- Hardware backends for other capabilities/profiles, automatic online tuning,
  and compile-compatible custom op schemas/fake implementations.
- General concurrent execution slots, compiler specialization and adaptive
  tier-aware graph dispatch. Fixed-tier capture,
  warmup, padding and replay use the existing vLLM runner.
- TP/PP plan broadcasts,
  adaptive precision and segment-level tier history. Request-fixed eager mixing
  and prefix-cache tier isolation are implemented using native request transport,
  scheduling and batch management, with native full-model graph support on
  SM120. Scheduling uses native vLLM admission and token/KV budgets.
- General checkpoint conversion, sharded-source conversion, old manifest
  migration, selective plane loading, LoRA/MoE/speculative compatibility.
- Distributed TP/GQA shard validation, broad hardware kernel coverage,
  model quality and serving performance measurements.

See [the architecture plan](../../bitblend-integration-plan.md) for the complete
target architecture. Files absent from that plan's proposed tree are not implied
to be implemented. Hardware-specific dependencies are determined with kernels.

