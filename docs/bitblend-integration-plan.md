# BitBlend 在当前 vLLM 中的集成方案

本文记录目标设计与当前目录分层；目录以第 3 节为准，其他设计项不表示已实现。
验证结果见 [当前验证说明](bitblend-sm120-validation.md)。阅读对象为相邻的
`Quantization/BitBlend` 项目及本 checkout（HEAD `dc61c6d027`）。
目标按最新要求设为 Hopper 与 Blackwell，不包含 A100。

当前代码框架与未实现清单见 [BitBlend framework](features/quantization/bitblend.md)。
本文描述目标架构；下文目录和接口不代表所有执行链路已经接通。

## 当前实现：同一次 forward 的混档批次（2026-10-23）

第一阶段保留原生 scheduler 的准入、优先级、token/KV 预算、chunked
prefill、抢占与恢复，只改变 runner 对已选请求的排列和线性层分段执行。
不等待三档凑齐、不改变请求精度、不创建独立队列或另一套服务引擎。

数据路径：`SamplingParams.extra_args["bitblend_tier"]`（HTTP 使用已有
`vllm_xargs`）→ InputProcessor 校验并在请求私有参数中补齐默认档位 →
现有 EngineCoreRequest / NewRequestData → runner 的 CachedRequestState。
无需增加 RPC 字段。量化配置 `mixed_batch=true` 显式开启；未指定档位的
请求仍使用量化配置中的默认 tier，关闭该功能时拒绝切换到其他 tier。

runner 在原生 attention 重排后、refresh_metadata 和输入准备前，通过
`InputBatch.swap_states()` 稳定按 Low/Mid/High 排列。若 attention 要求
decode/short-extend/long-extend/prefill 四分区，则先保留分区，再在各区
按档位排列；因此通用计划允许同一档位出现多个连续区间，而不破坏
attention 后端的合法顺序。普通无重排约束的 attention 下最多三个区间。
KV block table、采样参数、logits processor、异步输出映射继续由原生
InputBatch 管理，不逐层 gather/scatter hidden states。

每次 forward 从最终请求顺序与本轮实际 scheduled token 数构建不可变的
`BatchExecutionPlan(profile, spans)`；span 定义 tier 和半开 token 区间。
计划只在 model forward 上下文中绑定，异常后恢复。线性 adapter 检查
区间覆盖完整输入。SM120 的混档配置现在对所有区间（包括单区间）调用
统一 grouped GEMM，直接写入一个 BF16 输出；reference backend 保留逐段
计算作为对照。激活已通过复用官方 CuTe-DSL primitives 合并为一次 FP4/FP8
混合量化调用。计划不含 GEMM tile、设备类型或 kernel 选择。

grouped kernel 共用 Mid/High 的设备端 tile 主循环并增加 Low 单平面路径。
任务映射和 persistent round-robin 调度参考 CUTLASS grouped scheduler；
异步加载、swizzle 和窄 tile 参考 FlashInfer SM12x/CUTLASS SM120 实现。
支持独立的三档 tile/流水线配置，但同次 launch 的 CTA 线程数必须一致；
动态共享内存取三档需求的最大值。普通/persistent grid、tile 遍历顺序及
全部已实现流水线开关进入离线坐标搜索；不是整个联合空间的穷举。
Split-K 使用一次 grouped GEMM 和一次统一归约，不引入跨 CTA 全局同步。

prefix cache 的首 block hash 增加已解析的固定请求 tier，同时保留用户
cache_salt，避免跨档命中。后续 block 沿已有 parent hash 链传播隔离。
请求生命周期内固定 tier；抢占/重算继承同一 SamplingParams。

本阶段支持普通 dense、decoder-only 文本生成、原生 V1 GPUModelRunner、
TP/PP/DP=1、eager 或原生 FULL CUDA Graph。Graph 使用 mode=0，runner
在重放前更新地址固定的设备描述符，同一个 token 桶支持不同三档比例。
Graph 配置按 padded token 桶与局部 N/K 从独立 `graph_selected` 表查找，
支持各档不同 tile、Split-K 和任务遍历；未命中时使用原有默认配置。
离线搜索同时评估多个比例，独立复验还增加新输入和未参与搜索的随机分区。
启动时明确拒绝 mixed_batch 与 Dynamo/piecewise 编译、V2 runner、
speculative decoding、LoRA、MoE、多模态、微批流水线及 KV 传输的组合；
原固定档位路径不受此限制。请求内动态切档、跨 rank 计划传播仍未支持，
后续继续复用 vLLM 的 graph dispatch/forward context。

验证先覆盖计划与边界、真实 InputBatch 重排、档位参数拒绝、cache hash
隔离和 backend 分段数学等价；再用 Qwen3-0.6B 验证混档 prefill/decode、
chunked prefill、请求加入/退出、相同 prompt 跨档缓存隔离和 HTTP 参数。
对照为同一档位独立执行，检查请求对应关系与 logprob/token 输出；
验证报告明确区分容差、逐位一致、真正混合批次和仅同档批次。

本阶段实现与验证已经落地，相关 pytest 为 239 passed、26 skipped；
HTTP 记录到 52 次混档 forward，三档缓存首次 miss、同档复用命中，
异步低 KV 容量验证实际触发 4 次抢占并保持档位。数值对照的容差与
宽泛对照中的 token 差异见 [验证报告](bitblend-sm120-validation.md)。
当前不保证 batch-invariant generation，不将分段执行误写为单 launch。

设计方式：先定义完整框架、模块边界、数据合同和资源生命周期，再按这些
接口逐步填充实现。本文不按硬件或功能安排开发先后。现有代码是算法语义、
格式与验证证据的参考，不是必须保留的实现结构；允许重写 loader、dispatch、
kernel 和 graph 管理，只要满足下述数学、布局和数值合同。

框架实现不以确定所有硬件依赖为前提。CUDA、CUTLASS、FlashInfer 等具体版本，
可用指令、tile/stage、scale swizzle、编译参数以及 Graph 支持细节，均由
对应硬件 backend 实现时再验证和确定。当前提供最小配置、能力查询和扩展接口，
不预先建立完整版本矩阵；未确定项标记为待验证，不等同于已支持。
算法语义、数据所有权和执行计划一致性仍属于框架必须明确的合同。

## 1. 已有实现与目标的区别

已阅读 BitBlend 的 `README.md`、`AGENTS.md`、`docs/bitblend-design.md`、
`docs/final-implementation.md`，并核对量化转换、vLLM adapter、custom op、
graph bank、Mid/High CUDA 实现及 tactic 配置。

现有 Blackwell 数学定义必须保留：

```text
P0 = MXFP4(W)
R  = W - dequant(P0)
P1 = MXFP4(R)
P2 = MXFP8(R)

Low  = A4W4(X4, P0)
Mid  = A4W4(X4, P0) + A4W4(X4, P1)
High = A8W4(X8, P0) + A8W8(X8, P2)
```

P1 与 P2 是同一个 R 的两种编码。High 不包含 P1，两个 High 分支共享
X8/SFA；Mid 共享 X4/SFA。不同平面各自保留权重 scale。
现有转换代码从匹配的 Native MXFP4 checkpoint 复制 P0，然后量化残差。

当前限制：

- adapter 为 `BitBlendMxLinearMethod`，`create_weights()` 明确要求 TP=1。
- 使用 BF16 输入/输出，保留 lm_head 非量化；非 LinearBase 返回 None。
- `BitBlendMxQuantConfig` 从环境变量或 checkpoint 读取整个引擎的固定档位。
- Mid/High 默认 separate；显式选项才能进入 fused 路径。
- 自有融合实现使用 `cutlass::arch::Sm120`，JIT 限制 major=12。
  不能仅修改最低 capability 或编译 flag 就认为适配 SM100。
- tactic 数据库绑定 GB10/SM121、固定软件/源码指纹和 Qwen3-32B 的四类形状。
- `mx_graph_bank.py` 直接操作旧运行环境的私有 graph 字典，不是稳定接口。
- bench adapter 关闭 V1 multiprocessing；缺少当前生产多进程路径验证。
- 项目没有 Hopper 双 FP8 实现。保留结果是历史 SM121 证据，不能作为本次迁移验证。

## 2. 统一入口，两个格式 profile，三个架构 backend

对外统一使用 `bitblend` 量化入口，由 manifest 中的 profile 决定平面格式。
已有 `bitblend_mx` 名称通过显式格式适配器保留兼容，不能静默改变旧 checkpoint
的解释。Hopper 和 Blackwell 共享配置/运行时抽象，不复制两套服务引擎。

```text
BitBlendConfig / BitBlendLinearMethod
  ├─ profile=fp8_residual_2plane
  │    └─ backend=sm90
  └─ profile=mx_residual_3plane
       ├─ backend=sm100_family
       └─ backend=sm120_family
```

Hopper 的数学定义建议明确为以下新增规范（不是仓库已有事实）：

```text
H0 = FP8(W)
H1 = FP8(W - dequant(H0))
Low  = A8W8(X8, H0)
High = A8W8(X8, H0) + A8W8(X8, H1)
```

这表示两个权重平面、两个运行档位。不是给同一个 W 准备两个独立 FP8 副本。
Hopper 建议使用 E4M3、逐输出通道静态权重 scale、逐 token 动态激活 scale；
H0/H1 scale 独立，激活共用。该粒度是待质量验证的设计选择，应进入版本化
profile，不能在 kernel 内偷偷改变。不要将 Blackwell UE8M0/block32 合同
直接套在普通 Hopper FP8 GEMM 上。

设备 auto 负责选择兼容 backend；checkpoint profile 决定平面编码。
载入不兼容 profile 时明确报错，转换由离线工具显式完成。可维护两个
checkpoint 子目录及一个顶层 manifest，让启动器根据目标设备选择路径。
同一 TP/PP 执行组的支持合同为同构设备、相同 profile 和一致的执行计划。
异构设备部署不属于本框架的已定义运行模式。

## 3. 目录分层

当前已实现的目录如下；不再使用顶层 `vllm/bitblend/` 聚合各层代码。

```text
vllm/model_executor/layers/quantization/bitblend/
  __init__.py           # checkpoint 配置注册
  config.py             # profile、tier、ignore、backend 的校验
  linear.py             # 参数创建、加载后处理、apply
  profile.py            # 平面编码、档位、几何与 scale 合同
  checkpoint.py         # 离线编码、导出与格式校验
  runtime.py            # 不可变的每次 forward 执行计划

vllm/model_executor/kernels/linear/bitblend/
  base.py               # 支持检查、prepare/run、backend 注册与选择
  reference.py          # 解码后的数值参考，不作为性能结果
  sm120/
    backend.py          # 设备支持检查与权重准备
    ops.py              # 激活量化与三档 GEMM
    jit.py              # 编译标准 csrc 树或 wheel 中的源码
    config.py           # 编译参数与候选空间
    database.py         # 指纹校验与精确形状查表

csrc/quantization/bitblend/
  sm120/                # Mid/High 共用双平面 CUDA 模板

vllm/v1/worker/bitblend_runtime.py        # V1 GPU runner 的重排、计划与能力校验
benchmarks/kernels/benchmark_bitblend.py
docs/features/quantization/bitblend.md
```

SM90/SM100 backend、自适应 scheduler 策略和编译可见 custom op
仍是后续目标。请求固定档位混批和 SM120 整模型 CUDA Graph 已接入原生 runner，
不另外实现一套 Graph 管理器。kernel 选择器通过统一
`vllm.model_executor.kernels.linear` 入口导出。

推荐算法与 kernel 分层，复用当前 linear kernel 的支持检查风格，但为
多平面定义专用接口。已有单平面 backend 的 apply 通常自己量化激活，
不能直接调用两次还宣称共享了一次激活量化。

现有文件的修改点：

| 现有位置 | 修改目的 |
| --- | --- |
| `layers/quantization/__init__.py` | 内置名称、延迟导入、配置映射 |
| `layers/linear.py` | 使用其 V2 loader 注册装饰器；避免到处添加类名 |
| `model_executor/parameter.py` | 优先复用 PackedvLLMParameter、ModelWeightParameter、GroupQuantScaleParameter |
| `model_loader/weight_utils.py` | 原则上复用标准配置和 safetensors 加载，必要时才扩展 |
| `config/vllm.py`、`engine/arg_utils.py` | 传递经过类型校验的 BitBlend 运行配置并计入配置 hash |
| `_custom_ops.py`、`csrc/torch_bindings.cpp`、`CMakeLists.txt` | 如果将 kernel 编译进 vLLM 扩展，在这里接入 |

算子定义统一 schema，并预留 torch.library fake/meta 接口；具体绑定与编译
方式由 backend 实现确定，不能影响上层量化/调度接口。
CUTLASS 默认直接使用官方包或官方发行源码，不要求继承旧项目子模块或补丁。
旧项目记录的 4.7.0 和本地修改属于历史环境信息；按当前设计输入，最终计算
路径不依赖该修改，因此不将其列为迁移依赖。具体包形态、版本和所需头文件
在实现对应硬件 kernel 时确认；仅在实际需要时引入并记录补丁。

当前 `vllm/config/quantization.py` 的 `linear/moe` schema 是单平面 online
量化配置，不能假设向其任意添加 P0/P1/tier 字段就会生效。
使用专用 BitBlendConfig，统一由 EngineArgs/VllmConfig 解析和分发。
配置控制值和 checkpoint 格式元数据应分开。

## 4. 权重加载、布局与 TP

兼容适配器识别 Blackwell 六个参数名：weight_p0/1/2、weight_scale_p0/1/2。
Hopper 使用 H0/H1 或明确 profile 约束的两平面命名。
manifest 至少记录 schema_version、profile、编码、scale 粒度和 dtype、
平面公式、源模型 hash、logical shape、量化模块清单、未量化模块策略。
不要假定所有 profile 都有 global scale；存在时必须明确数学语义并验证加载。

流程固定为：

```text
读取/校验 profile → 选择 backend → 创建局部逻辑参数
→ 标准权重加载及融合/分片 → backend 专属 pack/swizzle
→ 预热并分配 workspace → 执行
```

持久化 layout 保持可移植，runtime swizzle 根据 backend 生成一次。
不能提前 swizzle 全局矩阵后再直接切 TP 分片。
默认采用离线转换后的 checkpoint；在线转换若存在，必须是显式启动操作，
并独立核算峰值，不能出现在请求执行或 graph capture 热路径。

迁移到当前 vLLM 时，将方法继承 LinearMethodBase，并在验证参数语义后使用
`register_weight_loader_v2_supported_method`。当前 vLLM 按方法类名决定是否
使用 V2 loader；旧 adapter 不能原样复制后假设 Packed 参数自然生效。
TP 与 QKV/gate-up 融合属于加载接口的固有合同：

- Column/QKV/MergedColumn：按各逻辑输出分片，处理 GQA KV head 复制。
- RowParallel：沿 K 分片，MX block 边界需对齐 32，payload/scale 同步切分。
- 两平面贡献在本 rank 合并，再由 RowParallel 外层执行一次归约。
- bias 必须遵守外层 skip_bias_add 与 TP 合同，不能重复加。
- tactic 使用实际局部 N/K，不能复用原始整层形状配置。

量化目标限定为指定 LinearBase 模块；默认保留 embeddings、norm、lm_head。
MoE、LoRA、speculative decoding、多模态特有层由兼容矩阵约束，未实现
组合必须启动报错，不能静默进入错误路径。

## 5. Kernel 接口与数值合同

各 backend 可注册 fused 与 separate 实现，由能力和成本选择；两者必须
共享一次激活量化的接口，而不是让每个平面私自重复量化。
separate 同时提供运行回退与融合验证基线。解码 BF16 路径只用于调试，
必须显式标记，不能当作原生 MX/FP8 性能。

融合目标是数学值合并后只做一次输出舍入，而不是机械共享裸 accumulator：

- Blackwell 每个 K block 的 SFA/SFB 在 blockscaled MMA 中参与计算；
  保留现有独立 plane scales 和共享激活的数学语义。
- Hopper 两个 FP8 平面的 scale 不同。对逐 token/逐通道 scaling，
  需要 `sx * (s0 * acc0 + s1 * acc1)`；不能先把两个未缩放 acc 相加再
  乘某一个 scale。双 accumulator + fused epilogue 是可行起点。
- separate 路径如果每个 GEMM 先输出 BF16，再相加，与 fused FP32 合并
  不会逐 bit 相同。正确性同时对照高精度参考和容差，不要求盲目 bitwise。
- 增加残差平面不会保证端到端准确率单调提高；激活量化误差仍存在。

SM12x 当前使用 Mid single-A compact 和统一 High mixed-plane kernel。
SM100 backend 按其 MMA、scale layout 和 pipeline 合同实现；Hopper 按普通
FP8 合同实现。共享 profile 和数值规范，不强行共享硬件 mainloop。
把 Sm120 改成 Sm100 不构成完成移植。

tactic 接口保留设备/SM、profile、tier、M/N/K、dtype、layout 与实现标识；
软件版本和源码指纹扩展字段在实际调优时确定。框架提供默认选择和可替换的
数据库入口，不要求搭建时已有各硬件的调优数据库。实测记录包含 correctness、
重复测量和 fallback。
未知形状选择可验证默认实现，不要在在线请求或 graph capture 内调优/JIT。
性能数据库不能直接跨 SM121 与 SM100 共用。

## 6. 档位切换与 CUDA Graph

统一定义三种运行配置，而不是三套演进中的接口：

- engine_fixed：整个引擎固定 tier。
- request_fixed：请求入队时确定 tier，生命周期内不变。
- scheduled：策略在请求允许的集合内选择 tier；每次提交冻结执行计划，并
  保存档位历史以支持抢占重算和 cache provenance。

默认关闭混档，沿用 engine_fixed。显式开启 mixed_batch 后采用 request_fixed，
同一次 forward 可包含多个固定请求档位，使用 BatchExecutionPlan 的 token
区间、单 launch grouped GEMM 与原生整模型 CUDA Graph；
不能用一份 forward 全局 tier 冒充 token 混档能力。

当前 BatchExecutionPlan 包含 profile 和各 span 的 tier/区间，backend 及
版本由已准备的权重与 backend 持有。请求档位通过原有 sampling params
传递，runner 在最终 token 排列确定后构造计划。不要靠环境变量、共享 config.tier 或全局
current_tier 表达在途任务的档位；异步调度会发生覆盖风险。

当前 request_fixed 数据路径：

```text
SamplingParams.extra_args 或正式请求字段
  → vllm/v1/request.py
  → 原生 Scheduler / vllm/v1/core/sched/output.py
  → V1 GPUModelRunner / InputBatch 重排
  → BatchExecutionPlan / forward 执行上下文
  → BitBlendLinearMethod
```

当前只接入 `v1/worker/gpu_model_runner.py`；
`v1/worker/gpu/model_runner.py`（V2）尚未接入混档，启动时明确拒绝。
后续应复用相同的 token 计划合同，不复制档位/Graph 控制逻辑。
旧项目 graph bank 依赖私有 manager.graphs 等字段，不能直接搬进两者。

框架统一管理 eager、编译和按 tier 专门化的 CUDA Graph 执行：

- 同一 padded shape 的不同 tier 对应不同 graph，capture 与 replay key 一致。
- tier 需进入实际编译专门化边界，不能只给 graph descriptor 加字段却让
  torch.compile 继续复用首次追踪的 Python 分支。
- 图捕获前完成各 backend 的预热、workspace 与 tactic 确定。
- warmup、dummy run、capture 和真实 forward 都要传递有效执行计划。
- 所有 TP/PP rank 对同一逻辑 step 使用一致计划。
- Graph 常驻内存要进入启动 profiling/KV 容量预算，不能在 KV 已耗满剩余
  显存后再额外捕获三套图。共享 pool 的并发和缓冲区生命周期需要验证。

固定档位与多档共享同一 graph registry；只按启用的 capture 集合分配资源。

## 7. KV cache 与调度策略

这是正确性约束：不同档位产生的 hidden states/K/V 可能不同，因此不能
仅因 tokens 相同就在不同档位请求间共用 prefix cache。

以 checkpoint/profile/tier 及必要的历史 provenance 建立 cache namespace，
可在请求创建、hash 生成之前结合已有 cache_salt
机制隔离，或扩展 `v1/core/kv_cache_utils.py` 的 extra keys。
同时保留用户原 salt 的隔离语义，不能简单覆盖。

同一请求中途切档，其已有 KV 仍来自旧档位；这是一种混合历史策略，切换
不会自动升级过去的 KV。它也影响 prefix cache provenance、抢占重算的
可复现性。scheduled 模式必须维护逐 segment 的执行历史和重算规则。
如果不保留完整历史，禁止跨历史复用 prefix cache，并明确重算的数值语义；
不能用最后一次 tier 简单代表整个 prefix。

区分两个 scheduler：

1. Kernel scheduler：tile、Stream-K、Split-K、persistent swizzle，放 kernel dispatch。
2. 服务 scheduler：请求、token budget、chunked prefill、档位队列，放 V1 Scheduler。

request_fixed 混档直接使用原生同步/异步 Scheduler，不另建档位队列；
runner 只重排已选请求，原 KV 准入/抢占、placeholder 与在途映射继续生效。
不在 schedule() 完成 KV 分配后删减输出。不支持按档位分轮执行，
原同档选择策略及专用交接接口已移除。未来引入 SLO/公平性策略时需单独验证。

测量 `T(profile,tier,M,N,K)` 可辅助 token budget 与组 batch，但不自动
授权降低用户精度。默认固定用户档位；只有请求明确允许的档位集合内，
策略才可以根据 SLO/负载降档，并记录原因与实际档位。

## 8. 内存目标必须单独核算

Blackwell 全平面常驻的理论线性权重开销（不含 padding/metadata）：

```text
P0: 4 + 8/32 = 4.25 bit/weight
P1: 4 + 8/32 = 4.25 bit/weight
P2: 8 + 8/32 = 8.25 bit/weight
总计            16.75 bit/weight
```

仅 Low 为 4.25，支持 Low+Mid 为 8.5，支持 Low+High 为 12.5。
Hopper 双 FP8 payload 为 16 bit/weight，另有 scale 开销。
全档常驻的收益目标是多档共享与计算/带宽选择，不应声称比 BF16 更省权重显存。
三平面相对三个完整独立档位模型可以共享 P0，但不是免费压缩。

配置中的 enabled_tiers 决定所需平面的并集，由统一资源规划器决定参数创建、
checkpoint tensor 选择与后处理；不能少分配参数却继续加载全部。
默认所需平面全部常驻。按需换入若启用，必须由独立驻留策略提供 ready event、
传输预算和 graph 地址稳定保证，不能在 kernel 内隐式换入。

## 9. 框架的数据对象与所有权

下表描述目标架构的数据合同，未全部实现。当前执行对象为 ProfileSpec、
LayerGeometry、PreparedLayer、KernelInvocation、ExecutionPlan 和
BatchExecutionPlan。Graph 身份与显存预算复用原生 vLLM，不保留无调用者的
GraphKey/ResourceBudget 类或零大小 workspace 查询接口。

| 对象 | 关键内容 | 创建者 / 生命周期 |
| --- | --- | --- |
| ProfileSpec | plane 格式、scale 合同、残差定义、tier 表达式、激活规则、版本 | 配置解析 / 引擎不可变 |
| LayerGeometry | 全局/局部 N/K、融合模块区间、TP shard、block 对齐 | 模型构建 / 层生命周期 |
| PlaneSet | 所需平面、原始参数、profile 与源权重身份 | loader / 层生命周期 |
| PreparedLayer | layout、backend 私有 state、公开只读 BackendRequest（profile/geometry/tiers/device/dtype/混档与 Graph 请求） | backend.prepare / 层生命周期 |
| DeviceCapabilities | 本 rank SM、backend 能力及可选依赖信息；精细探测由 backend 补齐 | worker / 引擎启动时生成 |
| RequestTierState | 允许档位、选定档位、质量约束、segment 历史 | 请求与策略 / 请求生命周期 |
| StepExecutionPlan | step_id、profile/tier、策略版本、请求映射、历史标记 | scheduler / 单次提交不可变 |
| KernelInvocation | 局部形状、layout、实际 token/padding、预选 tactic | worker / 单次层调用 |
| GraphKey | 执行专门化签名 + 原 vLLM batch descriptor | graph registry / 捕获生命周期 |
| ResourcePlan | 常驻权重、workspace、图/编译资源、KV 预算、并发上限 | worker planner / 启动或显式重配置 |

StepExecutionPlan 不携带 GPU 指针、层级 tactic 或逐层 Python 字典：
scheduler 不应了解每层 kernel 内部实现。worker 以紧凑专门化 ID 查询本地
PreparedLayer 和 tactic 表。设备能力汇总为协商后的执行能力，交给控制面
校验；scheduler 不假设本进程可访问实际 GPU。

配置与执行状态分离：配置被冻结且可序列化，执行计划按提交生成，workspace
按并发槽位租用，PreparedLayer 只持有可安全共享的只读权重及描述符。

## 10. 端到端控制流与接口

```text
配置 + manifest + worker 能力
  → 校验 profile/并行/启用档位与 backend 兼容矩阵
  → 资源规划、平面集合确定、标准分片加载、backend prepare
  → 预热/编译/capture，与 KV 预算统一收敛

请求 + 质量约束
  → request_fixed 校验并解析请求档位
  → 原 Scheduler 执行 token/KV 准入、分配与抢占
  → runner 重排并绑定 BatchExecutionPlan（eager / SM120 整模型 Graph）
  → LinearMethod 查询 PreparedLayer 并执行多平面算子
  → 原引擎采样/更新 + 实际 tier/性能观测
```

backend 接口至少包含：

```text
supports(profile, device, geometry) → SupportResult(reason)
prepare(plane_set, geometry, layout_contract) → PreparedLayer
resolve_invocation(prepared, tier, shape, graph_mode) → KernelInvocation
run(prepared, activation, invocation, workspace, output) → output
create_graph_workspace(layers, max_tokens, device) → workspace(update, run)
```

profile 定义算法，backend 声明硬件与布局能力，dispatch 选择已验证实现，
policy 选择服务行为；任何一层都不能私自更改另一层的语义。

当前实现由 `check_support(request)` 统一检查固定档/混档 Graph 能力，再调用
设备 backend 的 `supports(request)`。`supports_graphs` 和
`supports_mixed_graphs` 默认为 False，未声明能力的实现不会被 Graph 请求选中。
公共 runner 只读取 `PreparedLayer.request`，不访问 backend 私有 state。
混档 Graph workspace 使用 `update(plan)` / `run(prepared, x, bias)` 协议，
具体设备描述符、缓冲区和 kernel 发射均归 backend 所有。新增内置 backend
只需扩展 lazy registry；新增架构的 JIT 源码由 setup.py 按目录自动打包。
具体接入步骤见 [BitBlend framework](features/quantization/bitblend.md#adding-a-hardware-backend)。

fallback 只能使用相同 profile/tier 的合法实现；不能把缺少 High kernel
解释成自动执行 Low。激活量化可由融合 kernel 内联，但必须保持相同格式、
scale 与共享语义，不强制为了接口在全局内存中物化一次性中间结果。

调度成本模型与 kernel tactic 表分别管理。前者估计 batch 的模型级成本，
包含 attention、通信、采样和 graph 开销；不能只累加 GEMM 时间就声称
可以预测 TTFT/TPOT。两者均使用版本化离线测量，运行时查询而不在线搜索。

## 11. 并发、性能与一致性边界

- StepExecutionPlan 从提交到完成不可修改，通过 step_id 关联结果；所有
  rank 在执行前取得一致 tier。每个 PP microbatch 保留自己的计划。
- scheduler 记录的 segment 在实际提交时绑定，抢占后按原历史重算；
  speculative/失败 step 不得提前提交为已完成精度历史。
- request_fixed 混档复用原生准入顺序与公平性；允许档位来自请求合同，
  不因负载高而自动降低精度。未来策略需另行定义公平性约束。
- workspace 按最大并发与 graph 重放约束分槽；输入量化缓冲区不可被另一个
  在途 forward 覆盖。常驻权重和稳定描述符可以共享。
- 热路径不读取环境变量、不解析 JSON、不做源码哈希、不编译 kernel，
  不逐层执行跨进程通信，也不为了读取 tier 调用 GPU→CPU 同步。
- 加载期预构建 profile/tier/backend 到层执行句柄的映射；shape 查询使用
  已验证表或有界默认规则。控制面的一次档位选择不扩散成每层复杂 Python 策略。
- 能力接口覆盖 SM、profile、dtype、shape、TP/PP、runner、graph 和 async
  等维度。具体版本与特性组合随硬件 backend 实现补齐；框架可采用最小探测
  和默认规则，不要求提前穷举。未验证与未实现状态必须可区分，均不能静默
  宣称支持真实 GPU 执行。

## 12. 验证矩阵与框架完成标准

| 责任域 | 必须验证的行为 |
| --- | --- |
| 算法 | 每个 profile 的残差来源、独立 scale、激活共享、档位表达式 |
| 权重 | manifest 往返、融合映射、TP 分片/复制、runtime layout 解码 |
| Kernel | scale 合并、舍入容差、padding、边界形状、bias 与 fallback |
| 调度 | token/KV 不变式、公平性、质量约束、在途任务无串档 |
| 缓存 | 固定档隔离、混合历史 provenance、抢占重算一致性 |
| 执行 | 两种 runner、TP/PP、async、eager/compile/graph 语义一致 |
| 资源 | 启用档位的平面并集、workspace 并发、图地址与 KV 容量预算 |
| 性能 | 各实际设备的 kernel/模型级成本、图收益和端到端质量/延迟 |

kernel 正确性测试放已有 tests/kernels/quantization 附近，量化加载测试放
tests/quantization，调度测试放 tests/v1/core；优先扩展相邻测试。
性能实验放 benchmarks/kernels，端到端用统一模型、workload、软件环境、
graph 设置，分别报告质量、TTFT/TPOT、吞吐、常驻/峰值内存。
现有历史 JSON 不改写；本次迁移的新结果使用独立路径。

框架搭建完成的标准是：上述对象、接口、注册入口、runner 适配、能力校验、
资源所有权和错误传播路径均已具体化；未实现 backend 有清楚的扩展点和
能力状态。硬件相关能力校验可采用最小实现，完整依赖版本矩阵、具体硬件参数
和特性支持不作为框架完成的前置条件。不能仅以创建空目录和若干空类作为
完成框架。具体实现可以逐项
填充，但不应反复改变数学定义或依赖临时全局变量连接模块。

## 13. kernel 设计经验与官方参考（2026-10-22）

本节为源码分析与设计依据。范围为当前 SM120 源码、FlashInfer 0.7.2
及其 CUTLASS 4.5 头文件。H100 按用户确认采用双 FP8 profile，Low/High
两档；5090/B200 保留 MX 三平面 Low/Mid/High。H100 不凭空补一个 Mid。

### 当前实现的机制

- Mid 的 `DualCollectiveMma` 每个逻辑 K tile 消费两个权重平面，A/SFA
  只加载一份。High 在统一 mixed-plane 模板中复用 A8 寄存器，分别使用
  F4/F8 和 F8/F8 MMA。共享激活落实到搬运和片段复用。
- MX block scale 在 MMA 路径内参与计算，两平面的贡献进入同一个 FP32
  累加片段，最后一次 BF16 转换和输出。这样省去中间结果、独立求和与
  重复 epilogue；FP32 累加仍须按容差验证，不能保证与拆分运算位级一致。
- Mid 的 compact 共享存储、A/B 独立 stage、寄存器预取、P0 early release
  都是可测候选。它们分别交换共享内存、寄存器和同步开销，不能全开后
  假设性能更好。early release 必须在所有相关 consumer 完成读取后发生。
- 保留 per-shape tactic 和独立正确性 oracle。旧参数与历史成绩只提供
  候选依据，不跨设备复用最优结论。搜索域必须对应各模板实际支持的参数。

Mid 默认 128x128x128、CTA 拓扑及 producer/consumer 划分来自具体实现，
不应成为新实现的跨设备约束。Low 已走 FlashInfer/cuDNN MXFP4 GEMM，
可作为库基线；新项目代码应集中在有实际价值的多平面融合。

旧 Mid 代码仍可在 GB10 原仓库中查阅。当前 5090 源码为
`csrc/quantization/bitblend/sm120/{mid_warp.cu,high_warp.cu,dual_warp.cuh}`；
新 Mid 保留单激活流、独立平面缩放和 FP32 累加，使用官方 packed FP4
MMA 与 `ldmatrix`，不再保留旧 collective 作为运行时分支。

### FlashInfer / CUTLASS 中可借鉴的机制

- 以经过验证的单平面 mainloop、MMA atom、TMA、layout 与 epilogue
  为构件，围绕多平面数据流增加融合，减少自行重建底层协议的范围。
  CuTe-DSL 是实现工具，使用它本身不保证比 CUTLASS C++ 更快。
- SM100 block-scaled dense kernel 使用 `tcgen05`、TMEM 和 TMA/UMMA
  pipeline；SM120 使用另一套 MMA 与寄存器数据流。二者共享算法和接口，
  不应强行共享完整 mainloop。Hopper 则参考 SM90 TMA/WGMMA FP8 路径。
- SM100 实现有 `swap_ab` 下的窄 tile 检查，小 token 维度可映射到更
  合适的指令轴。其 scale TMEM 列偏移有合法性限制，不能只枚举更小 tile。
  小 M 可考虑方向交换、Split-K 或专门 GEMV，具体收益需要实测。
- SM100 Split-K 实现使用 cluster 内 FP32 归约，说明增加 K 并行度与
  归约协议要一起设计；该机制不能原样假定适用于 SM120 或所有形状。
- weights/scale 在加载期准备为 runtime layout；激活量化直接产生消费端
  所需布局。官方相邻接口提供预处理权重、量化激活复用或 out/workspace
  参数，可借鉴其生命周期划分；不能推定所有 backend 都支持所有接口。
- dtype 名称不足以判断兼容性，仍需检查 block 大小、scale 编码、布局、
  对齐、形状与硬件。NVFP4 和 MXFP4 尤其不能直接互换。

参考源码位于 FlashInfer 包的
`gemm/kernels/dense_blockscaled_gemm_sm100.py`、
`dense_blockscaled_gemm_sm100_splitk.py`、`gemm/gemm_mm_fp4_cute_dsl.py`，
以及 CUTLASS 的 `sm120_blockscaled_mma_tma.hpp`、
`sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp`。

### H100 双 FP8 的特殊数值约束

现有 FP8 profile 的权重 scale 是 per-channel，激活是 per-token。
令未缩放的两个 GEMM 结果为 C0、C1，则目标为：

```text
Y[m,n] = Sa[m] * (S0[n] * C0[m,n] + S1[n] * C1[m,n]) + bias[n]
```

S0 与 S1 通常不同，不能先算 C0+C1 再乘一个公共权重 scale。可研究
两个 FP32 累加片段在 epilogue 内分别缩放后求和，或其他等价的正确数据流；
相较 MX 的单累加器方案，它会改变寄存器压力和最佳 tile。
还应保留官方 FP8 accumulation/promotion 的精度控制并做长 K 验证。
这一 profile 需要独立 checkpoint 和质量评估，不能沿用 MX 的 PPL 结论。

### 后续设计的判定标准

先按实际形状区分小 M decode 与大 M prefill，以官方单平面、双平面拆分
和现有融合实现为对照；拆分路径会引入额外输出舍入，只作为有明确容差的
性能对照，独立 FP32 oracle 仍是数值验证依据。测量同时记录权重流量、
并行 tile 数、共享内存/寄存器、同步与归约、GPU 时间和主机提交开销。
小 M 不仅检查有效计算比例，也检查是否能充分利用 SM；不以一个 tile
覆盖所有 M。权重准备、编译和 tactic 选择移到加载/预热期，workspace
所有权、并发和 graph 地址稳定在接口阶段明确，CUDA Graph 支持单独验证。

重写目标是共享格式合同、验证与资源接口，同时允许各架构有自己的数据流。
具体 tile、stage、融合粒度和语言选择等待这些依据形成设计后再确定，
本节不预设性能倍数或跨架构通用最优配置。
