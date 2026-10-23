# RTX 5090 BitBlend shape database

Fixed-tier Mid and High use the unified dual-plane template in
`csrc/quantization/bitblend/sm120/dual_warp.cuh`; fixed-tier Low uses FlashInfer/cuDNN.
Mixed batches use `grouped.cu` with native Low/Mid/High specializations of the
same device tile routine. See the grouped search section below.
GB10 timings and obsolete collective configurations are not deployed.

## Mixed grouped search

The grouped launch supports independent per-tier token/channel tiles, K blocks,
pipeline depth, warp groups, register prefetch, vector epilogue, every legal
integer K split, and High's gather/`ldmatrix` switch. Repeated spans of a tier
share its configuration. All tiers must have a common CTA size: 128, 256, 512,
or 1024 threads. Shared storage is reused, so the maximum tier allocation must
fit 101,376 bytes. The compiled combined kernel's occupancy and spills are
measured; independent kernels' occupancy is not substituted.

Scheduling choices are ordinary grid (`waves=0`) or persistent round-robin
(`waves=1/2/4`, multiplied by SM count and compiled resident blocks), crossed
with the two M/N tile traversal orders. The implementation does not use TMA,
producer/consumer warp specialization or cross-CTA completion counters.
Heavy-first ordering is also searched; descriptors retain original output rows,
so changing execution order does not change the token order. Task coordinates use
host-precomputed CUTLASS `FastDivmod` parameters. The residency launch-bound target
is 1/2/3/4/6/8 CTAs per SM, constrained by threads and shared memory; actual
compiled occupancy and local memory are recorded rather than assumed.

Search is a coordinate sweep over the full legal per-tier space within each
common thread family, followed by scheduling comparisons. It is not exhaustive
over all three tiers' Cartesian product. With a positive search limit, parameter
axes are interleaved around the current configuration, rather than taking a
prefix of the Cartesian enumeration. `--grouped-search-passes` controls repeated
sweeps (default 2); each common thread family starts from a shape-sized tile.
The budget and passes are recorded. Neither path claims a global optimum.
`--grouped-search-families N` first measures all four CTA-size seeds and refines
the fastest N families; the default 0 refines all four. A seed's ranking is not
a proof that other configurations in its family are slower. Coordinate updates
require at least a 1% improvement to limit noise-driven changes.

Example commands use the environment described in the validation report:

```bash
python benchmarks/kernels/benchmark_bitblend.py --compare-grouped \
  --grouped-spans low/3 mid/9 high/129 mid/17 --output /tmp/grouped
python benchmarks/kernels/benchmark_bitblend.py --search-grouped \
  --grouped-spans low/3 mid/9 high/129 mid/17 \
  --grouped-layers qkv --grouped-search-limit 12 --grouped-search-passes 2 \
  --output /tmp/grouped-search
python benchmarks/kernels/benchmark_bitblend.py \
  --verify-grouped /tmp/grouped-search/grouped.json \
  --grouped-spans low/3 mid/9 high/129 mid/17 \
  --grouped-layers qkv --output /tmp/grouped-search
```

Omit the search limit for a full coordinate sweep; this compiles many bundles
and can be expensive. Each candidate must have finite output and relative
L2 < 0.001 against independently decoded plane products. Independent verification
uses a fresh input seed, freezes the measured winner and writes
`grouped-verified.json`. This verifies the candidate, not global optimality.
Only verified, matching-fingerprint files can be deployed through the existing
`VLLM_BITBLEND_SM120_TACTICS` setting. Exact lookup includes N/K and the entire
ordered tier/M list, retaining repeated spans. Mixed pattern entries coexist
with fixed-tier entries in `grouped_selected` and `selected` respectively.
Unlisted patterns select one of four precompiled defaults by their largest span:
token tile 8/32/64/128 with boundaries 8/32/128, 64 channels, K=128,
two stages and no Split-K. The 128-token tile uses two token warp groups;
other defaults use one. Model loading precompiles all four, so this heuristic
does not add a new JIT variant during forward.

GEMM timing graph-replays preallocated calls; Split-K reduction is included.
Each graph contains the requested number of iterations, so Python replay gaps
are amortized outside the per-GEMM measurement. Timings from older graphs with
one GEMM per replay should be remeasured before comparing small kernels.
The comparison baseline remeasures separate Mid/High GEMMs with their default
heuristics and the selected grouped tile/pipeline parameters, choosing the faster
configuration per span before measuring the complete separate sequence and copies.
`--reference-configs FILE` additionally remeasures exact historical shape configs;
it never imports old timings or bypasses database fingerprint checks. Fixed-tier
Low uses cuDNN's implementation. This finite baseline search is not an exhaustive
independent per-tier tuning. Reports include its configurations and accuracy.
Additional eager linear timings include fused mixed quantization, allocation and
descriptor upload. These kernel graphs do not enable mixed-model CUDA Graphs.

Whole-model mixed graphs are enabled separately through the native runner's
`compilation_config` (`mode=0`, `cudagraph_mode=FULL`). Their stable workspace
uses a separate exact-token-bucket table, with defaults on misses. Eager exact-
pattern tables are not applied inside these graphs. See the mixed-tier section
of [the feature guide](features/quantization/bitblend.md) for descriptor updates,
capture bounds and eager fallback behavior.

## Whole-model graph tuning

`--search-mixed-graph` benchmarks `MixedGraphWorkspace.run`, including fused
activation quantization, the capacity-bounded grouped launch and any Split-K
reduction. Descriptor uploads occur before replay, outside the timed interval,
as in the model runner. This is a linear GPU measurement, not end-to-end serving.
It captures each candidate once and changes its device descriptors to replay
single tiers, balanced/imbalanced triples, missing tiers and up to twelve repeated
spans. The token bucket remains constant. These timings must not be substituted
with exact-span GEMM timings: captured grid capacities can differ substantially.

The parameter axes and common-CTA constraints match the grouped search above.
The same budget/family/pass options apply; a zero limit enumerates the legal
per-tier product in each coordinate sweep, not the joint three-tier product.
All four thread-family seeds are measured before selecting families to refine.
`--graph-compile-workers N` prebuilds a finite sweep's independent bundles in
parallel; GPU correctness checks and timing remain serial and start after those
builds finish. The default is one compiler worker.
Candidates exceeding 64 MiB of Split-K scratch per linear are excluded from this
search. The runtime supports each tier's split count, M/N tile and heavy-first
ordering without changing a captured graph when the request mix changes.

The objective is the geometric mean of pattern latency ratios against the
original graph default, with equal weight per distinct pattern. The baseline
and three best candidates are remeasured; a candidate must improve the mean by
at least 1% and stay within `--graph-max-regression` (default 10%) for every tested
pattern. Otherwise the baseline is retained. This finite sample is not a promise
of improvement for every possible ratio or a globally optimal configuration.

```bash
python benchmarks/kernels/benchmark_bitblend.py --search-mixed-graph \
  --m 1 2 4 8 16 32 64 96 128 256 512 \
  --grouped-search-limit 12 --grouped-search-passes 1 \
  --grouped-search-families 1 --repeats 5 --iterations 50 \
  --output /tmp/graph-search
python benchmarks/kernels/benchmark_bitblend.py \
  --verify-mixed-graph /tmp/graph-search/mixed-graph.json \
  --m 1 2 4 8 16 32 64 96 128 256 512 \
  --repeats 7 --iterations 100 --output /tmp/graph-verified
```

Verification reranks the saved finalists with a fresh input seed and six extra
deterministic random partitions per bucket. All measured outputs must have finite
relative L2 < 0.001 against independently decoded FP32 plane products. Only the
completed `mixed-graph-verified.json` may be loaded through
`VLLM_BITBLEND_SM120_TACTICS`. Its `graph_selected` keys are
`graph:12:M:N:K`; both the twelve-span capacity and exact padded M/N/K must match.
Existing environment/source fingerprint checks remain mandatory. The report also
records weights, samples, oracle errors, resource use and the measured report hash.
Changing split counts or reduction order can change model outputs slightly;
linear correctness does not replace model quality evaluation.

`--compare-activation` compares the one-launch mixed CuTe DSL quantizer with
official per-span quantizers. It also compares quantization plus the same default
grouped GEMM and reports eager timings including allocation and descriptor
upload. Graph timing uses preallocated outputs, 100 iterations when requested,
and alternating measurement order. The official baseline keeps its normal
PDL-enabled kernels. Example:

```bash
python benchmarks/kernels/benchmark_bitblend.py --compare-activation \
  --grouped-spans low/3 mid/9 high/129 mid/17 --repeats 7 --iterations 100 \
  --output /tmp/mixed-activation
```

The source fingerprint includes `activation.py`; historical databases must be
independently reverified after this integration before they can be loaded.

Source changes for grouped GEMM invalidate previous source fingerprints.
Historical fixed-tier databases require remeasurement and verification before
deployment; their hashes and timings must not simply be relabeled.

## Search domain

`sm120/config.py` enumerates every implemented choice for each tier:

| Parameter | Available choices |
| --- | --- |
| Token tile | 8, 16, 32, 64, 128 |
| Channel tile | 64, 128; every token/channel pair is compiled |
| K splits | Every integer from 1 to K/block_k, including uneven partitions |
| K stage / buffers | 64, 128, 256 / 1–8 |
| Token warp groups / CTA threads | 1, 2, 4 / 2 × channels × token warp groups |
| Payload loading | `cp.async`; Mid uses `ldmatrix`, High chooses gathers or `ldmatrix` |
| Register prefetch / vector epilogue | Each off/on, crossed on the top 32 screened base configurations |
| Scales | Independent block-32 UE8M0 for each plane and activation |
| Accumulation | FP32; BF16 output; deterministic split reduction |

The search is staged, not the full Cartesian product of all switches. It first
measures every legal tile/K-stage/buffer/token-warp/load/split combination with
register prefetch and vector epilogue disabled. For K=1024 there are 4,356 Mid
and 6,640 High base candidates after shared-memory and divisibility pruning.
It then crosses both additional switches on the fastest 32 base configurations,
adding up to 96 candidates per shape. The resulting top 32 and reference
configuration receive repeated refinement timing. Each measured candidate is
checked against independently decoded arithmetic. Fresh independent verification
then chooses among three finalists and the reference.

Shared allocation must fit 101,376 bytes; compiled occupancy must be nonzero.
Reports preserve rejected resource configurations, register counts, local-memory
bytes and active CTAs per SM. Spilled but launchable configurations remain
measured. TMA, producer/consumer warp specialization and persistent scheduling
are outside this implemented search space.

The first workload uses real Qwen3-0.6B layer-zero weights:

| Projection | N | K |
| --- | ---: | ---: |
| QKV | 4096 | 1024 |
| Attention output | 1024 | 2048 |
| Gate/up | 6144 | 1024 |
| MLP down | 1024 | 3072 |

The default M set is `1,2,4,8,16,24,32,64,128,256,512,1024,2048,4096,8192`.
Exact `(tier,M,N,K)` lookup permits reuse across models with the same geometry
and compatible environment; it does not establish another model's quality.
Missing shapes use the same template's heuristic, without bucket extrapolation.

## Measurement and independent verification

The benchmark checks every configuration against FP32 products of independently
decoded quantized planes, with finite output and relative L2 < 0.001. Weight
encoding is unchanged. Low/Mid activation quantization uses ceil-scale MXFP4;
High uses ceil-scale MXFP8. An arithmetic failure aborts the search.

GEMM measurements use preallocated output/scratch, four calls per standalone
CUDA Graph replay during search (16 during comparison), shuffled candidate
order. Every base and secondary candidate receives one screening round;
refinement uses alternating reverse rounds. CUDA-event timing includes split
reduction. Native MXFP8 uses the FlashInfer/CUTLASS backend used by vLLM; BF16
uses PyTorch. Additional measurements report activation quantization and
quantization plus GEMM for the best measured configuration. These graphs reuse
layer-zero weights; they are not serving throughput or cold-HBM measurements.
Bias, attention and request scheduling are outside this microbenchmark.

Independent verification uses fresh inputs, the three fastest candidates plus
the reference, and six paired timing rounds. `--reference-configs` can import
configuration values from an earlier table to be remeasured under current code;
its old timings and fingerprint are never deployed. A finalist replaces the default
when its median improves by at least 2% and it wins at least five rounds.
Otherwise the independently checked default is retained. Only this step writes
`verified-database.json`; an unverified search cannot be deployed.

## Reproduce

Run with exclusive GPU use in the existing validation environment:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export CXX=/usr/bin/g++-11
export MAX_JOBS=2
export PATH=/root/quant/.bitblend-validation/.venv/bin:/usr/local/cuda-13.2/bin:$PATH

python benchmarks/kernels/benchmark_bitblend.py --search-mid-warp \
  --source /root/quant/models/Qwen3-0.6B --output /path/to/new-mid-search \
  --m 1 2 4 8 9 16 17 24 32 33 64 65 128 129 256 512 1024 1025 2048 4096 8192 \
  --repeats 5 --iterations 5
python benchmarks/kernels/verify_bitblend_database.py --mid-warp \
  --directory /path/to/new-mid-search --iterations 20
export VLLM_BITBLEND_SM120_TACTICS=/path/to/new-mid-search/verified-database.json
```

For High replace `mid` with `high` in both flags. Comparison without exhaustive
integer splits uses `--compare-mid-warp` or `--compare-high-warp`; add
`--all-tiles` to compare all compiled tiles. Searches can resume completed shapes when source/environment, requested M,
reference artifact and timing arguments match. For four projections and the
21 M values above, each tier covers 84 shapes. The base space contains 640,332
Mid and 976,080 High configurations, plus up to 8,064 secondary candidates per
tier. Candidate counts and screening/refinement trials are retained separately.

Search and verification artifacts remain outside source control. `<tier>-warp.json` records all
candidates, raw timings, numerical errors and the source weight SHA-256;
`<tier>-verification.json` records fresh finalist checks;
`verified-shapes.csv` summarizes the deployable entries. Fingerprints cover
GPU/resources, driver, CUDA/C++ compiler, PyTorch, FlashInfer, cuDNN, CUTLASS
headers, native sources including `.cuh`, and Python runtime source hashes.

## Runtime selection

The package ships a verified RTX 5090 mixed-Graph table at
`vllm/model_executor/kernels/linear/bitblend/sm120/configs/rtx5090.json`.
Its 44 entries load automatically only when the complete environment and source
fingerprint matches. A different device/software environment or missing shape
uses the normal default configurations. This is a mixed-Graph table, not a
complete fixed-tier/eager tuning database or a guarantee for every model.
`VLLM_BITBLEND_SM120_TACTICS` takes precedence over the bundled table; an explicit
invalid or incompatible table still fails loading. The JSON is included in
package data for installed distributions. Full timing/verification artifacts
remain external; a deployable verified table is the exception stored in the package.

Entries must be tagged `mid_warp_v1` or `high_warp_v1` with tactic=0 and a
legal tile/split configuration. Runtime rejects obsolete collective families,
unverified results and incompatible fingerprints. All selected specializations are
compiled and initialized during layer preparation. Forward performs only
exact-shape lookup. CUDA Graph capture freezes that choice for replay;
mixed-tier full-model graphs use the separate `graph_selected` section described
above. Dynamo remains unsupported.

After a kernel or runtime change, an older database fails loading. Rerun
search/verification or explicitly revalidate its selected configurations with
new inputs and the new fingerprint, retaining provenance and original timing
scope. Never silently overwrite fingerprints. Before adopting a table, run
real-weight loading, generation and the documented WikiText quality evaluation.

## Previous Mid/High table and final comparison

The previous RTX 5090 / Qwen3-0.6B table (before expanded pipelines) is
`/root/quant/validation/bitblend-mid-warp/verified-database.json`:
84 freshly searched Mid entries and 84 retained High selections, all 168
revalidated as complete linears against independent decoded arithmetic and
changed-input CUDA Graph replay. High timings retain their historical scope.
The table SHA-256 is
`4fa59140bf7395dfb5e821ac74fb37f51c1afde952e9920143da388286cefbb4`.

The expanded table is generated under
`/root/quant/validation/bitblend-expanded-autotune/`. Old tables fail the new
source fingerprint and serve only as configuration references. Compare a newly
verified table with the official autotuned MXFP8 baseline:

```bash
python benchmarks/kernels/benchmark_bitblend.py --compare-mid-warp \
  --tactics /root/quant/validation/bitblend-expanded-autotune/verified-database.json \
  --m 1 8 32 128 1024 8192 --repeats 7 --iterations 30 \
  --output /path/to/comparison
```

Comparison mode runs FlashInfer `autotune_v2` for MXFP8 before graph capture,
as native vLLM does. Search mode records the default MXFP8 heuristic only;
those baseline timings must not be used to claim gains over autotuned MXFP8.
Final timing, HTTP FULL Graph results and quality checks are documented in
`docs/bitblend-sm120-validation.md` and the artifact directory's `summary.md`.

The expanded 2026-10-23 table contains 84 newly searched and independently
verified entries per tier, including M=9/17/33/65/129/1025 boundary shapes.
Its SHA-256 is
`4bacc61231690f0782fe39161f2d9a4e90946fce451ef2e4c9ece6478513c002`.
Verification adopted 41 Mid and 69 High winners; other shapes retained their
remeasured reference configurations under the same expanded kernel template.
All 168 deployed complete linears passed changed-input graph replay and
independent decoded-arithmetic validation. Previous HTTP throughput figures
refer to the previous kernel/table, not this expanded search.
