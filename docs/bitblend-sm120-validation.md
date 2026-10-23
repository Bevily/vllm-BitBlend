# BitBlend：RTX 5090 验证与复现

更新：2026-10-23。当前支持 MX Low/Mid/High 的固定档位 eager /
原生 vLLM CUDA Graph，以及显式开启的单 GPU 混档 eager / 整模型 Graph 批次。
后文保留各阶段记录，最新混档 Graph 验证见末节。功能范围见 [主文档](features/quantization/bitblend.md)，
参数搜索见 [形状数据库](bitblend-sm120-database.md)。

## 模型与环境

- 原始模型：`/root/quant/models/Qwen3-0.6B`。
- 量化副本：`/root/quant/models/Qwen3-0.6B-BitBlend-MX`。
- Python：`/root/quant/.bitblend-validation/.venv/bin/python`。
- RTX 5090（SM120）、CUDA 13.2、PyTorch 2.14.0+cu132、FlashInfer 0.7.2、
  bundled CUTLASS 4.5.0、G++ 11；以下端到端验证均为 TP=1。

固定档位 Low 使用 FlashInfer/cuDNN；Mid/High 分别由 `mid_warp.cu` 和
`high_warp.cu` 实例化共用的 `dual_warp.cuh`。下方历史端到端结果中的
Mid 仍为旧 collective，不能代表新 Mid；新验证记录见本文新增章节。激活量化均为 FlashInfer CuTe-DSL：Low/Mid 为
ceil-scale MXFP4，High 为 ceil-scale MXFP8。权重编码与 checkpoint 不变。

## SM120 三档 grouped GEMM（2026-10-23）

当前混档路径改为 `grouped.cu`：Low/Mid/High 共用一个 launch，CTA 按任务
选择对应的原生 MMA 主循环，直接写统一输出。固定档位保留原生 Graph 路径；
混档即使只有一个区间也使用 grouped family。激活暂时仍按区间调用官方
CuTe-DSL；合并成最多两次量化 launch 尚未实施。Split-K 需要额外一次统一
FP32 reduction。实现复用 `dual_warp.cuh`，没有复制另一套 Mid/High 主循环。

新增验证材料位于 `/root/quant/validation/bitblend-grouped/`：

- `tests.log`：量化测试套件 **128 passed、26 skipped**，包含固定档位
  CUDA Graph 回归、混档数值、完整可选参数轴和精确区间数据库校验。
  使用验证环境执行
  `python -m pytest --confcutdir=tests/quantization tests/quantization/test_bitblend.py -q`；
  本套件不依赖仓库根 conftest 的额外模型测试依赖。
- 四种配置验证普通/persistent grid、M/N 遍历、不同档位不同 K 分块和 tile、
  3 级流水、token warp groups、High `ldmatrix`、寄存器预取、向量化输出、
  非整齐 K 分片、N 尾部、重复 Mid 区间和各组独立 scale 原点。
  与独立解码数学参考比较，要求 finite 和 relative L2 < 0.001。
  另检查预分配 GEMM/reduction 的 CUDA Graph replay；不代表混档模型 Graph。
- `memcheck-kernels.log`、`synccheck.log`：四种配置各 4 passed，两个工具均
  `ERROR SUMMARY: 0 errors`。首次默认 memcheck 的 `memcheck.log` 记录了
  PyTorch `cuGetProcAddress_v2` 查询超出驱动版本的 14 个 API 探测报告；
  复跑使用 `--report-api-errors no`，保持 kernel 内存和同步检测开启。
- `model.py`、`model.log`、`offline-async-preemption.json`：真实 Qwen3-0.6B，
  eager、单进程原生引擎、async scheduling，32 个 KV blocks，12 个请求各
  生成 64 tokens。记录 164 次混档 forward、4 次实际抢占，覆盖三档；
  档位、chunked prefill、decode/prefill 同轮执行与默认档位行为验证通过。
  控制形状 9/9、同档批次对照 12/12 token 一致，最大 selected-token
  logprob 差为 0.0111513。广泛不同 prompt 对照 10/12 一致，不能声称
  batch-invariant generation。该记录未启用旧 verified tactic 表。

- `launches.py/.json/.log`：PyTorch CUDA profiler 实测四个区间的 GEMM
  部分。无 Split-K 恰好 1 个 `bitblend::grouped::kernel`；Split-K=2
  恰好再增加 1 个 `bitblend::grouped::reduce`，没有分档 GEMM 或输出 copy。
- `search.log`、`search-smoke/grouped.json`：QKV 真实权重，区间
  Low=3 / Mid=9 / High=129 / Mid=17，N=4096、K=1024；每轴限制 1 个
  候选，但遍历全部四种线程族和调度选项，共检查 23 个配置。
  `verify.log`、`search-smoke/grouped-verified.json` 用新输入独立复核选中配置，
  relative L2=4.10e-6。选中三档 `(16,64)`、K=128、2 stages、2 token warp
  groups、无 Split-K、persistent waves=1、N 连续遍历。
  5 轮、每轮 30 次 graph replay 的中位数：grouped GEMM **20.79 µs**，
  原分段 GEMM/归约/拷贝 **31.06 µs**。含逐区间量化、分配和描述符准备的
  eager CUDA 时间线分别为 **222.52 / 742.97 µs**；grouped 单次 wall 约
  324.43 µs，仅作流程诊断，不代表稳定 CPU 延迟或端到端服务吞吐。
  对照使用旧路径默认 heuristic，没有沿用已失效的最佳配置表。

离线工具新增 grouped 对比、受限/完整坐标搜索、独立复核及精确区间查表，
命令和搜索边界见 [数据库说明](bitblend-sm120-database.md#mixed-grouped-search)。
受限搜索只验证四种 CTA 线程族的流程，不代表已经完成全部混合形状调优。
旧表因 native/Python 指纹改变而失效，不自动迁移历史性能或验证状态。

## 原生混档批次第一阶段（历史基线，2026-10-23）

实现方案见 [集成方案开头](bitblend-integration-plan.md)，使用方式见
[混档批次](features/quantization/bitblend.md#mixed-tier-batches)。本阶段
使用原生 V1 GPUModelRunner、同步/异步 Scheduler、InputBatch 与请求传输；
不修改调度预算和 KV 准入顺序。最终 token 区间在一次 forward 中绑定，
线性层逐区间调用现有 backend 并 copy_ 拼接结果。尚未实现单 launch
异构 grouped GEMM 或混档 CUDA Graph，不声称已有端到端加速。

复现材料位于 `/root/quant/validation/bitblend-mixed-batch/`：

- `offline.py`、`offline-sync.json/.log`、`offline-async.json/.log`：
  本地原生引擎对照，开启 prefix cache、chunked prefill，token budget=96、
  max_num_seqs=8、max_model_len=512、seed=42。覆盖不同 prompt 长度、
  请求完成/新请求加入、decode 与 chunked prefill 同轮执行。
- `offline-async-preemption.json/.log`：额外限制 32 个 KV block（512 tokens，
  含原生保留 block），max_model_len=384，各请求生成 64 tokens。
  实际发生 4 次原生抢占，覆盖 Low/Mid/High；12 个请求均完成，
  记录到 164 次混档 forward，整个请求历史中 tier 保持不变。
- `validate_http.py`、`http-report.json`、`http-server.log`、
  `http-trace.jsonl`：标准多进程 HTTP 服务，开启原生 async scheduling。
  `mixed_probe.py` 是目录内的验证观察器，仅在原生 Worker 的执行上下文
  入口记录 CPU 计划，不改变模型计算；不属于运行时依赖。
- `tests.log`：相关 pytest 共 **239 passed、26 skipped**。跳过项为原有
  kernel 配置形状条件不满足。覆盖配置拒绝、计划覆盖与隔离、真实
  InputBatch 的四分区重排、prefix hash 链、reference 分段数值，以及
  SM120 非连续输入/重复 span/bias 的逐位等价与固定档位 Graph 回归。

环境沿用本文顶部；启用的表为
`/root/quant/validation/bitblend-expanded-autotune/verified-database.json`，
SHA256=`4bacc61231690f0782fe39161f2d9a4e90946fce451ef2e4c9ece6478513c002`。
本轮没有修改 native kernel、SM120 backend 或查表指纹文件。

HTTP 验证结果：

- 9 个并发 completion 请求按各自档位完成；工作进程观察到 **52 次**真正
  多档位 forward。这里不把仅同档区间的 forward 计为混档。
- 受控短 prompt 的三档混合/逐档对照 **6/6 文本一致**。
- 相同 149-token prompt、相同用户 cache_salt，依次 Low/Mid/High 首次
  `cached_tokens=[0,0,0]`；再次分别为 `[144,144,144]`，验证跨档隔离与
  同档复用。启动参数 `--enable-prompt-tokens-details` 用于读取该统计。
- 不指定 tier 与显式默认 Low 一致；错误名称或非字符串 tier 均返回
  HTTP 400，`param=bitblend_tier`，之后健康检查正常。
- 三档并发 streaming chat 均正常收到 `[DONE]`。服务退出码 0，
  无强制结束，本次最终服务日志未出现 EngineDeadError；不代表历史
  关闭竞态已经被本次修改修复，也不是长时间压力或性能评估。

数值范围与未解决项：

1. GPU 分段 adapter 与相同输入/相同形状的独立 backend 调用逐位一致。
   本地受控短 prompt（各 tier GEMM 形状匹配）9/9 生成 token 一致；
   同步/异步结果相同，选中 token 最大 logprob 差为 `0.0111513`。
   验证脚本对该受控模型对照使用绝对 logprob 阈值 `0.02`，不宣称逐位一致。
2. 改变批次组成、chunk 划分或缓存命中后，宽泛混合/逐请求对照有
   **3/12** 个请求生成 token 不同；HTTP 宽泛对照亦有 **3/9** 不同。
   这些记录完整保留，未把其标为逐请求数值一致通过。
3. 同档且只有一个 span、走原有 linear 路径的对照，也可复现 High 的
   token 差异和最高约 `0.262` 的选中 token logprob 差；Mid 有 `0.0111513`
   的差异。因此不能仅凭自由生成不同就认定是混档映射错误。
   形状/tactic、attention reduction 和量化阈值的逐操作贡献尚未进一步
   归因；当前实现**不保证 batch-invariant generation**。后续 grouped
   GEMM 除性能外，仍需独立检查数值与质量，不能只要求 token 完全相同。

以上是混档执行与调度生命周期验证，不替代全量任务质量评估。
已通过的独立档位 PPL 与离线 autotune 记录仍见下文。

## 当前 High 的数值和质量

- 完整测试覆盖混合平面、独立 scale、尾部、非连续输入、bias、空输入、
  档位切换和配置表派发。各次结果与源码指纹保存在对应验证目录。
- 统一 High 模板的 memcheck 为 0 errors；racecheck 为 0 errors、0 warnings。
  使用 `--kernel-name kns=bitblend::high_warp --report-api-errors no`，
  设备检测仍开启；未宣称整个环境的 CUDA API 检查通过。
- Qwen3 原生加载核对 1291 个 checkpoint tensor、786 个参数、115 个
  未量化 tensor；112 个融合线性层的三档共 336 次检查通过。
- WikiText 全测试集 62 篇，context=1024、batch=1、eager、seed=42：
  High word PPL 为 `33.13628341564168`。该结果与使用相同 High 配置表的
  复测一致；不能用于跨模型比较或替代更多任务的质量评估。
- 全部十种 tile 和合法整数 split-K 的扫描、独立复测细节见下文。

下文为相应版本的验证记录；源码变化后，历史表不能绕过指纹检查直接部署。
清理后的兼容表见 `bitblend-high-cleanup/verified-database.json`。

## 新 High 的完整参数扫描与配置表（2026-10-22）

本轮在同一新 High 模板内补齐 `(8,128)`、`(16,128)`、`(32,128)`
实例，形成 token={8,16,32,64,128} × channel={64,128} 的十种组合；
split-K 从仅 1/2/4/8 扩展为全部整数 1…K/128，含非均分 K 分段。
主循环、block32 MX 语义、K=128、双缓冲、加载方式和 FP32 确定性
归约不变。完整可调/固定参数清单见 `bitblend-sm120-database.md`。
这里的完整性指已实现模板的注册参数域，不代表穷尽硬件可实现的设计。

搜索与独立复测目录为 `/root/quant/validation/bitblend-high-search-v1/`。
环境仍为 RTX 5090、CUDA 13.2、Torch 2.14.0+cu132、FlashInfer 0.7.2，
原始 Qwen3-0.6B layer-zero 四组真实权重；核心源码 SHA256 为
`45c5c01e64fedb0fd9c8cc24380e27c2ed2ad4aa9f8dace40d4b05ce083195db`。

- M=`1,2,4,8,9,16,17,24,32,33,64,65,128,129,256,512,1024,1025,2048,4096,8192`，
  共 84 个形状、11,760 个配置，全部完成 oracle 和计时，无静默丢弃。
  最大相对 L2 为 `1.04528e-4`，阈值为 `1e-3`。
- 独立复测使用新随机输入 seed=20261023，每形状复测扫描前三名及
  默认配置，六轮交替顺序、每轮 20 次 replay、每 replay 四次 GEMM。
  33 个形状采用至少 2% 且至少五轮获胜的配置；其余 51 个保留默认值。
  所有原始扫描赢家仍留存，并未当作失败候选删除。
- 相对同轮默认启发式，最终配置在 84 个形状上的加速几何平均为
  `1.05015x`，最大 `1.44115x`；最大值为 Down M=129，默认
  `64x64/split2` 的 13.348 μs 降至 `32x128/split4` 的 9.262 μs。
  这是预分配 GEMM 的结果，包含 split-K 归约，不含激活量化、分配、
  HTTP 或完整模型开销；不应解释成模型吞吐提高 5%。
- 最终配置实际采用 split-K={1,2,3,4,5,6,8}，说明只测 2 的幂次会
  漏掉有效候选。完整参数与耗时见 `verified-shapes.csv`；每候选原始
  五轮数据见 `high-warp.json`，复测数据见 `high-verification.json`。
- 十种实例均无 register spill，详见 `resources.txt`；新增三种实例
  分别为 98/103/116 个寄存器，静态 shared 1024 B。
- 完整 pytest：75 passed、12 skipped，跳过原因是对应 K 无法提供
  足够分段。新增测试覆盖新配置 schema、精确形状匹配、非法分段与
  kernel family 校验与查表执行。
- Compute Sanitizer：memcheck 0 errors；racecheck 0 errors/0 warnings。
  仍使用新 High kernel 过滤与 `--report-api-errors no`，范围同前文。
- 启用 `verified-database.json` 后，原生模型加载验证的 1291 tensor、
  786 参数、115 个未量化 tensor 和 336 次线性层检查均通过。
  High 对 reference 最大逐层相对 L2 为 `8.56675e-5`。

启用本轮配置表的 WikiText 全测试集评估也已通过：62 篇，context=1024、
batch=1、eager、seed=42，High word perplexity 为 `33.13628341564168`，
与上一轮新模板的启发式参数评估一致。模型评估报告同时记录配置表的
路径和 SHA256，详见 `wikitext-high.json`。Python Ruff/format 与 diff
空白检查均通过。

部署表沿用 `VLLM_BITBLEND_SM120_TACTICS`，以 `kernel=high_warp_v1`
标识当前 High 模板。环境/源码指纹不符或未验证表均拒绝
加载；只查精确 M/N/K，未收录形状使用同一模板的启发式参数。该表只含 High，Mid 使用默认配置，Low 不受影响。

## 查表后 High 与 BF16 速度复测（2026-10-22）

复用上述标准 HTTP 服务脚本，重新运行 BF16 与 High，各八种负载、
两轮正反顺序。High 启用 `bitblend-high-search-v1/verified-database.json`，
SHA256 为 `8a0f420e819d1194940c81b5983d30f58614350127148de7e6145b0b7a71092d`。
仍为 Qwen3-0.6B、TP=1、eager、Triton attention、关闭 prefix cache，
固定 6 GiB KV cache；输出固定 128 token。两服务顺序运行。

| 输入 token | 并发 | BF16 输出 token/s | High 输出 token/s | High / BF16 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 53.93 | 37.27 | 0.691x |
| 128 | 4 | 229.99 | 146.46 | 0.637x |
| 128 | 16 | 904.05 | 592.33 | 0.655x |
| 128 | 32 | 1873.26 | 1229.42 | 0.656x |
| 1024 | 1 | 57.96 | 36.76 | 0.634x |
| 1024 | 4 | 228.46 | 144.97 | 0.635x |
| 1024 | 16 | 875.97 | 564.28 | 0.644x |
| 1024 | 32 | 1709.30 | 1133.38 | 0.663x |

32 组、896 个正式请求全部通过，逐请求长度与错误检查通过。两个服务
退出码均为 0，均未记录 `EngineDeadError`。单并发 TPOT：短输入
BF16/High 为 18.34/26.57 ms，长输入为 17.03/26.91 ms。模型加载显存
为 1.12/1.22 GiB，负载整卡峰值为 8.495/8.545 GiB，均含固定 KV cache。
High 仍常驻全部 canonical 平面，未在本轮精简权重驻留。

HTTP 服务退出后，单独复测 layer-zero 四类真实线性层，
M={1,8,32,128,512,2048,8192}，共 28 个形状。High 使用相同验证表，
BF16 为 `torch.mm`；均预分配输出，High 另预分配 split-K workspace。
六轮交替顺序，CUDA events 计时，每轮 20 次 replay、每 replay 四次
GEMM，取中位数。此处 graph 只用于隔离 CPU 提交间隙，不表示服务
已支持 CUDA Graph。独立反量化 oracle 检查全部通过，最大相对 L2
为 `4.25881e-5`。

纯 GEMM 中 High 的 27 个形状更快，O/M512 基本持平。M=1 的
BF16/High 耗时分别为：QKV 5.657/3.667 μs、O 5.146/3.097 μs、
Gate/up 6.656/4.608 μs、Down 6.170/3.584 μs，即约 1.44–1.72x 加速。
这些数据不含激活量化、分配、查表、其他模型算子和 HTTP，不能直接
解释为模型加速。HTTP 吞吐仅为 BF16 的 63%–69%，说明纯 GEMM
优势尚未转化为端到端收益；本轮未重新做 profiler，不能精确归因。

完整报告、逐请求数据、两轮吞吐范围与 TTFT/TPOT/ITL/E2EL 见
`/root/quant/validation/bitblend-high-vs-bf16-tuned/summary.md`、
`summary.csv`、`summary.json`；纯 GEMM 配置及六轮计时见 `gemm.json`，
运行脚本为同目录 `gemm.py`。这是固定并发短批次测试，服务未交替
多次重启；不代表 CUDA Graph、其他模型或长期服务性能。

## 固定档位 CUDA Graph 接入与复测（2026-10-22）

SM120 后端现可由 vLLM 原生 runner 捕获与重放完整线性层。移除两处
eager-only 拒绝并保留 invocation 的 graph 标志；数学语义、原生 kernel、
配置选择与权重布局不变。临时激活、输出和 split-K scratch 由 PyTorch
Graph 显存池保持稳定地址，无新增模型分支、每层 scratch 缓存、图管理器
或调度器。新 High 在所有形状仍使用同一新模板。

启用方式为取消 `--enforce-eager`，指定
`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`。
`mode=0` 关闭 Dynamo，与开启 CUDA Graph 独立。原生 runner 完成形状
预热、padding、图显存池和重放；未捕获形状仍使用同一套 kernel 的 eager
路径。当前为固定档位支持，不包含动态切档或 `torch.compile` 适配。

- 完整 pytest：78 passed、12 skipped。新增三档完整线性层捕获测试，
  覆盖量化、GEMM/归约、bias、非连续和多维输入、新输入重放，以及多张
  存活图间的输出/scratch 隔离；重放与同形状 eager 逐位一致。
- High 配置表的 84 个真实权重形状全部复验，每形状捕获完整线性层，
  更换两次输入后重放；均与 eager 逐位一致，独立反量化 oracle 最大
  相对 L2 为 `4.76359e-5`。复验确认指纹仅 `backend.py` 变化，原生
  kernel、ops、配置空间与环境不变；保留原 GEMM 计时与选择，并在新表
  记录原表 SHA、旧/新指纹和复验报告。未覆盖或修改历史表。
- Low/Mid 在 `FULL` 模式完成标准 HTTP 启动与生成；High 在
  `FULL_DECODE_ONLY` 模式完成捕获与 HTTP 测量。捕获尺寸为
  [1,2,4,8,16,24,32]，服务日志确认实际 `FULL` 重放以及 3→4、17→24
  等 padding。`FULL` 支持范围还取决于 attention 后端与捕获尺寸。

重新测量 BF16/High × eager/decode Graph 四组：Qwen3-0.6B、TP=1、
Triton attention、关闭 prefix cache、固定 6 GiB KV cache；其他配置
沿用前述 HTTP 测试。本次只测并发 1/32、输入 128/1024、输出 128，
每组两轮正反顺序，共 32 次正式测试、1152 个正式请求，0 失败，
逐请求输入/输出长度通过。四个服务退出码均为 0，均无 `EngineDeadError`。
两轮均值为输出 token/s：

| 输入 token | 并发 | BF16 eager | High eager | BF16 Graph | High Graph | Graph 下 High/BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 58.60 | 37.40 | 567.17 | 594.84 | 1.049x |
| 128 | 32 | 1806.33 | 1213.29 | 10600.78 | 10869.75 | 1.025x |
| 1024 | 1 | 56.28 | 37.53 | 518.87 | 538.96 | 1.039x |
| 1024 | 32 | 1740.98 | 1140.53 | 4632.65 | 3817.98 | 0.824x |

短输入 Graph 吞吐已接近，不能将两轮小幅领先当作显著优势。
`FULL_DECODE_ONLY` 不捕获 prefill/混合步骤；长输入负载日志同时出现
`NONE` 和 `FULL`，尚不能用本轮数据精确分解残余差距。

另外进行了并发 3/17、各输出 32 token 的跨服务文本对比。并发 3 两档
均 3/3 相同；并发 17 的 BF16 为 6/17，High 为 11/17 相同，长度全部
相同。原控制脚本在全部速度测试和关闭完成后，因要求 BF16 跨运行并发
文本全部相同而退出 1；该失败及原始文本保留。不同批处理形状/调度下的
逐字可复现性未得到保证，不能把此项当作固定形状重放数值检查，也不能
声称所有端到端文本逐位一致。

补充的 High 单请求对照固定 decode M=1：四个提示各生成 64 token，
eager/Graph 的文本、token 和所选 token 的 logprob 全部完全相同
（最大 logprob 差为 0）。报告为 `serial-generation/comparison.json`。

全部结果位于 `/root/quant/validation/bitblend-cudagraph/`：新兼容表为
`verified-database.json`，复验明细为 `database-revalidation.json`，
四组速度原始记录为 `graph/`、`eager/` 下的 `summary.json`/`summary.csv`
及逐请求结果。运行脚本和未通过的跨运行文本断言均保留，汇总见
同目录 `summary.md`。

## 边界

其他设备、TP/PP、Dynamo、动态切档图调度和长期压力尚未验证。
当前常驻全部 canonical 平面和额外 scale，不承诺显存压缩；workspace
在 eager/捕获时由 PyTorch 分配，图重放复用图显存池中的地址。
BF16 输出后加 bias 与 reference 的 FP32 bias 舍入顺序
可能不同。形状数据库的 GEMM 计时不能直接当作模型吞吐收益。

## 新 Mid packed FP4 kernel（2026-10-22）

环境：RTX 5090 SM120，CUDA 13.2、PyTorch 2.14.0+cu132、FlashInfer 0.7.2（CUTLASS 4.5），Qwen3-0.6B。

## 实现

Mid 保持 A4×P0 + A4×P1、block32/UE8M0 独立缩放和 FP32 累加；使用原生 m16n8k64 FP4 MMA、ldmatrix 和 cp.async 双缓冲，同一份激活寄存器供两平面复用。Mid/High 共用 dual_warp.cuh；每档所有形状均走各自的新 specialization，无旧 kernel fallback。Low 不变。

主要参考 CUTLASS mma_sm120.hpp / mma_traits_sm120.hpp、FlashInfer independent_small.py / dense_blockscaled_gemm_sm120_b12x.py。未使用 TMA、persistent scheduler 或 warp specialization。

编译 10 种 tile：tokens={8,16,32,64,128} × channels={64,128}；K stage=128，支持全部合法整数 split-K。离线覆盖本模型 84 形状、11,760 配置，并以新输入、六轮配对计时验证选择。它是已实现空间的穷举，不代表所有硬件上可能的参数。

## 线性层性能

使用真实 layer-zero 权重、固定复用权重、预分配 GEMM output/scratch，CUDA Graph unroll=16，7 轮×30 replay。MXFP8 在捕获前经官方 autotune_v2，Mid 使用已验证表。下表为四类投影的加速比几何均值，定义 MXFP8耗时/Mid耗时；大于1表示Mid更快。完整列包含激活量化+GEMM及必要归约，不含bias、attention、请求调度。

| M | 纯 GEMM 加速比 | 激活量化+GEMM 加速比 |
| ---: | ---: | ---: |
| 1 | 2.031x | 1.675x |
| 8 | 1.946x | 1.516x |
| 32 | 1.612x | 1.289x |
| 128 | 1.228x | 1.091x |
| 1024 | 0.891x | 0.896x |
| 8192 | 0.895x | 0.881x |

小 batch 已快于 MXFP8，大 batch 完整线性耗时平均约高12%–14%；单个大 batch 形状最高约高23%，不能声称所有形状持平。搜索过程中的 MXFP8 为默认 heuristic，仅用于搜索记录，最终对比应使用 final-comparison/mid-warp.json 的 autotune 结果。

## 标准 HTTP 服务

原生 vLLM、TP1、Triton attention、prefill/decode FULL CUDA Graph，max_model_len/max_num_batched_tokens=2048、max_num_seqs=32、固定6GiB KV cache，关闭prefix cache。输出128token，两轮正反顺序。各档288个正式请求，共576个，失败0。单位为输出 token/s，两轮均值。

| 输入 | 并发 | MXFP8 | Mid | Mid/MXFP8 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 587.00 | 590.60 | 1.006x |
| 128 | 32 | 11397.06 | 11264.66 | 0.988x |
| 1024 | 1 | 544.57 | 589.79 | 1.083x |
| 1024 | 32 | 6248.11 | 6314.61 | 1.011x |

整体吞吐接近；长输入单并发轮间波动较大，两轮不足以确认小幅优势。Mid模型常驻仍包含全部canonical平面及scale，服务ready显存8.602GiB（MXFP8 8.159GiB，均含固定KV/Graph等分配），并非显存压缩结果。

两档服务退出码0、GPU显存均回到基线。Mid在全部请求结束后的关闭阶段仍记录一次已有的EngineDeadError；不能声称关闭问题已消失。

## 正确性与质量

- pytest：94 passed、24 skipped（非法split组合）。
- Compute Sanitizer memcheck/racecheck：BitBlend原生kernel过滤范围内0错误，racecheck 0警告；CUDA API错误检查关闭。
- Mid84 + High84个完整线性层重新验证，换两次输入后Graph与eager逐位相同；独立解码oracle最大相对L2=0.0001023895。High旧选择经过重新正确性/Graph验证后保留，历史计时未伪装为新调优。
- 真实checkpoint：1291 tensor验证，112线性层×3档共336检查，Mid对reference最大相对L2=0。
- 完整WikiText2 test（62文档，context1024）：Mid word PPL=44.87324658342172，与旧Mid结果完全一致。

## 复现与产物

启用表：`VLLM_BITBLEND_SM120_TACTICS=/root/quant/validation/bitblend-mid-warp/verified-database.json`。表含84个新Mid、84个重新验证High条目；SHA256：`4fa59140bf7395dfb5e821ac74fb37f51c1afde952e9920143da388286cefbb4`。

- final-comparison/mid-warp.json：经过官方autotune的最终微基准，含各形状原始轮次。
- http-full/summary.json / summary.csv、逐请求JSON、server.log及settings.json：服务速度、延迟、显存与环境。
- search/：全配置原始计时与独立验证结果。
- linear-graph-revalidation.json、loading.json、wikitext-mid.json、tests.log、memcheck.log、racecheck.log：验证证据。

适用范围是当前5090及Qwen3-0.6B四类投影；其他模型需重新测量，B200/H100、TP/PP、动态切档调度未在本轮实现或验证。


## Mid/High 优化诊断（2026-10-23）

设备已安装 Nsight Compute 2026.1，但完整 sections 与单独 LaunchStats
均因 `ERR_NVGPUCTRPERM` 失败：驱动 `RmProfilingAdminOnly=1`，容器缺少
`CAP_SYS_ADMIN`。尚未获得 Tensor Core 利用率、warp stall、bank conflict
或实际带宽计数器，因此不能宣称已确认唯一主要瓶颈。

替代采集使用 Nsight Systems 的 CUDA Graph 节点时间线、相同 cubin 的
CUDA occupancy API 和 SASS。QKV/gate-up 的 M=1/128/8192 结果显示：
大 batch 的 Mid tile=128×64，每 block 34,816 B shared、128线程，最多
2 block/SM；High tile=128×128，每 block 84,992 B shared、256线程，
最多1 block/SM。两者理论 warp 占用率上限均16.7%，不是实测利用率。
仅在 occupancy 查询中去掉动态 shared 约束，上限分别为4、2 block，
说明 shared 限制驻留；这不证明减少 shared 必然提速。所选编译实例
未发现寄存器 spill 迹象。小/中 batch 的部分 grid 只有96/128 block，
无法覆盖170个SM；增加split又有独立归约成本，必须比较总耗时。

优先实验为 High 的 FP8 操作数矩阵加载（`ldmatrix`）、紧凑 FP4 的
加载/寄存器重排，以及两档沿 token/channel 同时划分 warp。当前一个
warp 承担全部 token panels，大 tile 每线程持有64个FP32累加器。
K-stage和pipeline深度应与shared/寄存器联合搜索，不宜直接增加stage。
诊断未修改运行时kernel或配置表。完整证据、推断边界及解除计数器权限后
的复现脚本位于 `/root/quant/validation/bitblend-ncu/summary.md`。

流水线理论预算补充见
`/root/quant/validation/bitblend-ncu/pipeline-theory.md`，完整候选域为
`pipeline-space.csv`（1152组合）。预算包含每block的1KiB驱动shared预留，
K64的4字节scale行槽和显式寄存器估算；当前配置的驻留预测与前述12组
CUDA occupancy查询一致。重点候选为Mid保持K128双缓冲、沿token方向分配
warp，以及High的K64双/三缓冲与warp分工。新配置未编译、未计时，不能
把理论驻留上限当成加速比。计算中还比较了输出shared空间复用和High
FP4在shared中展开的容量代价，详见报告。

## 扩充 Mid/High 离线 autotune（2026-10-23）

本节记录上述理论候选落地后的实际结果。环境仍为 RTX 5090 SM120、
CUDA 13.2、PyTorch 2.14.0+cu132、FlashInfer 0.7.2 / CUTLASS 4.5。
Mid/High 共用 `dual_warp.cuh`，所有形状使用该模板的 specialization，
不存在按形状退回旧 kernel 的运行时分支。

### 实现与搜索范围

在原十种 token/channel tile 和全部合法整数 split-K 上新增：

- K 分块 64/128/256、共享内存环形流水线 1–8 级。
- token warp 分工 1/2/4；token panels 必须均分，且满足共享内存限制。
- High 的 shared gather / `ldmatrix` 加载；Mid 保持 packed FP4 矩阵加载。
- 下一 MMA 权重片段的寄存器预取，以及复用流水线 shared 的向量化输出。
- 编译后的寄存器、local-memory 和 occupancy 查询；零驻留配置排除。
- 层准备阶段预加载所选 specialization，保持 CUDA Graph 捕获安全。

搜索使用真实 Qwen3-0.6B 四类 layer-zero 投影，M 为
`1,2,4,8,9,16,17,24,32,33,64,65,128,129,256,512,1024,1025,2048,4096,8192`。
每档 84 个形状；Mid 实测 648,396 个候选，High 实测 984,144 个候选，
合计 1,632,540 个，资源排除为 0。每个候选均检查独立解码 oracle。

这是分阶段搜索：基础 tile/K/stages/warp/load/split 空间完整枚举，
在初筛最快 32 个基础配置上交叉预取与向量化输出，再对最快 32 个候选
和原配置做五轮精测。两个附加开关未在所有基础配置上穷举。
随后换输入对前三名和原配置做六轮配对复验：至少快 2% 且赢得至少
五轮才替换原配置。Mid 采用 41 个新配置、保留 43 个原配置；High
采用 69 个新配置、保留 15 个原配置。保留的是配置值，旧计时和旧
fingerprint 不会部署。所选 Mid 实例没有 local memory；4 个所选 High
实例有非零 local memory，仍保留实测结果，不声称全部没有 spill。

### 扩充前后性能

旧 kernel 只作为仓库外归档的基准，不是运行时 fallback。新旧使用
相同真实权重和输入、预分配输出/scratch、CUDA Graph unroll=16，
七轮交替顺序、每轮 30 次 replay。下表是四类投影的加速比几何均值，
定义为旧耗时/新耗时；大于 1 表示更快。GEMM 包含必要的 split 归约，
完整线性列包含激活量化，不含 bias、attention 或请求调度。

| M | Mid GEMM | Mid 完整线性 | High GEMM | High 完整线性 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.977x | 1.004x | 1.016x | 1.016x |
| 8 | 0.974x | 0.974x | 1.006x | 1.010x |
| 32 | 0.991x | 0.990x | 1.011x | 1.008x |
| 128 | 0.995x | 0.994x | 1.038x | 1.040x |
| 1024 | 1.016x | 1.015x | 1.140x | 1.134x |
| 8192 | 1.036x | 1.024x | 1.144x | 1.131x |

24 个形状整体：Mid GEMM 0.9980x、完整线性 1.0000x，基本持平；
High GEMM 1.0575x、完整线性 1.0551x，大 batch 改善较明显。
这不是所有形状都提速：Mid gate/up M=8 的完整线性耗时增加约 5.86%；
纯 GEMM 还有 High QKV M=1、attention output M=32、down M=32 等
超过 5% 的退化项，详见 summary.json。独立复验比较的是当前模板下
的配置，不能保证它在后续七轮测试中快于归档旧二进制。
结果限于固定复用权重的微基准，不是 HTTP 吞吐或冷 HBM 性能。
本轮未重新测 HTTP，前节服务结果仍属于旧 kernel/旧表。

### 正确性、质量与产物

- pytest：112 passed、26 skipped。
- Compute Sanitizer：BitBlend 原生 kernel 过滤范围内 memcheck 0 errors，
  racecheck 0 errors / 0 warnings；关闭 CUDA API 错误报告。
- 168 个选中完整线性层、每个更换两次 Graph 输入，Graph 与 eager
  逐位一致；相对独立解码 oracle 的最大 L2 为 0.0001402807。
- 真实 checkpoint：1291 tensor、112 线性层、三档 336 项检查通过。
- 完整 WikiText2 test、context=1024：Mid word PPL=44.87324658342172，
  与此前相同；High=33.13628341564168，历史值=33.136899079701436，
  微小差异处于 kernel 累加舍入变化范围，不作质量提升结论。

全部产物在 `/root/quant/validation/bitblend-expanded-autotune/`。
启用 `VLLM_BITBLEND_SM120_TACTICS` 指向其中 `verified-database.json`；
包含两档各 84 项，SHA256 为
`4bacc61231690f0782fe39161f2d9a4e90946fce451ef2e4c9ece6478513c002`。
`summary.md` / `summary.json` 汇总参数分布、性能和退化项；
`mid/` / `high/` 保存全部候选、误差、原始计时和独立复验；
`matched-comparison.json` 含前后对照和官方 autotuned MXFP8 / BF16 基线。
`linear-graph-revalidation.json`、`loading.json`、`wikitext-*.json`、
`memcheck.log`、`racecheck.log` 和 `tests.log` 保存验证证据。

本轮未实现 TMA、producer/consumer warp specialization、persistent
scheduling、B200/H100 kernel 或混档请求调度；Nsight Compute 硬件
计数器权限仍受限，不能将编译资源信息解释为实测 Tensor Core 利用率。

## Grouped GEMM 优化与独立复验（2026-10-23）

本轮只优化混档 grouped GEMM，激活仍按原有 span 使用官方 CuTe DSL
量化，没有合并激活量化。所有混档形状仍使用同一 grouped 模板族。
实现参考 CUTLASS grouped scheduler、`FastDivmod` 和 FlashInfer 的
资源约束方式：

- host 预计算两个快速除法参数，替换任务映射中的动态整除；每组任务数
  必须小于 2^31。descriptor 从 8 扩为 14 个 int64。
- 搜索普通网格、round-robin persistent 网格、M/N 遍历顺序，以及按
  工作量排序的 group；排序保留原输出行位置，split-K 归约按行区间找组。
- 搜索 launch-bounds 驻留目标；保留各档独立 tile、K 分块、流水级数、
  warp 分工、预取、向量输出与合法 split-K。去掉无后续任务/流水级时
  不必要的同步；共享内存复用仍保留同步。
- 无精确表项时选择四个预编译 tile 默认值之一（8/32/64/128 token），
  模型加载期编译，forward 不新增 JIT。

搜索器支持完整合法的逐档空间；本次实测使用有限预算：每档每轮 12 个
交错坐标候选、一轮搜索，先比较四种 CTA seed 再细化其中最快一种，
每层约 56–60 项。之后补测历史 Mid/High 配置的兼容组合、Low split、
驻留目标和任务顺序。不是全空间穷举，不声称全局最优。
最终使用新输入独立复验，当前源码和环境 fingerprint 严格匹配的表共
12 项：四类 Qwen3-0.6B layer-zero 投影，各三种 span 组合。

### 新旧对照

下表为 GEMM 微秒中位数，包含必要的 split-K 归约，不含激活量化、
descriptor 上传和 Python 包装。decode 为 low/1, mid/1, high/1；
prefill 为三档各 256 token；不均衡为 low/3, mid/9, high/129, mid/17。
同一预量化输入、固定复用权重、预分配输出/scratch；每个 CUDA Graph
包含 100 次 GEMM，测量 7 轮后除以 100，避免 Python replay 间隙混入
微秒级 kernel 计时。旧代码仅在仓库外保存并用于对照，不参与服务。

| 组合 | 投影 | 旧默认 | 新默认 | 旧代码+所选 tile | 新代码+所选配置 |
| --- | --- | ---: | ---: | ---: | ---: |
| decode | QKV | 6.77 | 6.70 | 6.28 | 5.03 |
| decode | O | 8.18 | 7.44 | 5.43 | 5.39 |
| decode | gate/up | 6.19 | 5.98 | 6.04 | 5.69 |
| decode | down | 11.28 | 10.32 | 6.07 | 6.05 |
| prefill | QKV | 82.50 | 25.96 | 22.22 | 21.35 |
| prefill | O | 47.50 | 23.60 | 15.34 | 14.89 |
| prefill | gate/up | 120.20 | 38.73 | 31.33 | 30.44 |
| prefill | down | 64.15 | 34.09 | 21.59 | 21.12 |
| 不均衡 | QKV | 26.94 | 20.22 | 13.07 | 11.84 |
| 不均衡 | O | 15.40 | 23.68 | 11.23 | 10.72 |
| 不均衡 | gate/up | 37.81 | 30.40 | 16.72 | 15.21 |
| 不均衡 | down | 21.80 | 34.13 | 12.76 | 12.36 |

所选配置较旧默认加速 1.09–3.95 倍，主要来自 tile/调度参数选择。
旧代码使用匹配 tile 时差距明显缩小；旧代码不支持新的 heavy-first
和驻留控制，因此该列仍不是完全隔离源码变化的实验。新默认在不均衡
O/down 上分别退化约 54%/57%，不能将命中调优表的结果推广到任意组合。
表采用严格的 N/K 和有序 span 匹配，不对未知组合套用相邻形状结果。

独立复验中，grouped 相比逐 span GEMM（包括输出拼接）均更快：
decode 约 2.28–2.82 倍、prefill 约 1.37–1.79 倍、不均衡约 1.89–2.12
倍。逐 span 基线重新计时了默认、当前配置和可用历史配置，Low 另比较
cuDNN。两套对照独立运行，GPU 时钟不同，不能相减推断单项开销。
所有选中项相对独立解码 oracle 的 L2 误差小于 0.001。

Nsight Compute 仍因 `ERR_NVGPUCTRPERM` 无法读取硬件计数器。
默认 specialization 的静态 SASS 指令数由 2379 降至 2186，MUFU.RCP
由 7 降至 3；寄存器由 125 增至 136，均无 local spill。这些是静态信息，
不能据此宣称实测 Tensor Core 利用率或动态瓶颈。

产物目录：`/root/quant/validation/bitblend-grouped-opt/`。
`verified-database.json` 是本轮可加载的 12 项表；`native-comparison.json`
保存新旧原始样本，三个形状子目录保存搜索和独立复验记录。
这不是 HTTP 吞吐、全模型速度或冷 HBM 测试。

### 回归与真实模型边界

- 完整 BitBlend pytest：129 passed、26 skipped。
- Compute Sanitizer memcheck / synccheck：各 5 个 grouped 测试通过，
  均为 0 errors；memcheck 关闭 CUDA API 错误报告，kernel 内存检查开启。
- Ruff check、format check 和 `git diff --check` 通过。
- 真实 Qwen3-0.6B、eager 混档、异步调度、32 KV blocks 强制抢占：
  164 次混档 forward、4 次抢占，覆盖 chunked prefill 与 decode 同批，
  请求 tier 保持正确。默认配置受控对照 9/9 token 一致，同档批处理
  12/12 一致；默认请求等价显式 Low，非法 tier 被拒绝，正常 shutdown。

真实模型发现一个明确的数值限制，不能把带表验证描述为全部通过：
启用精确离线表时，4592 次线性调用命中表，受控混档/单请求对照只有
6/9 一致（3 个 High 不同），同档批处理仍为 12/12。其 decode 混档命中
O/down 的 High K64、split=16，而单请求走默认 K128、split=1。
关闭表恢复受控 9/9；在仓库外诊断脚本中让单请求 decode 也使用相同
配置，受控对照恢复 9/9（最大所选 token logprob 差 0.011152）。
这支持差异来自配置相关的浮点累加次序；不等价于模型质量无损证明。
该诊断改变单请求配置后，同档批处理对照为 9/12，因为两侧又使用了
不同配置。未把诊断中的配置借用加入生产代码，精确匹配规则保持不变。

原始长短请求混合对照中，默认与带表均只有 10/12 token 全序列一致；
更改 batch 形状后的严格生成一致性目前不受保证。本轮没有重新跑
WikiText PPL，不应沿用旧 fingerprint 的 PPL 为新调优表背书。
正常默认与匹配配置诊断均成功退出；最初带表脚本因受控一致性断言
失败而非正常结束，日志保留，未删除或放宽该断言。
`model-default.json`、`model-with-database.json`、`model-matched-config.json`
及对应日志保存完整证据。离线表是可选产物，未改为全局默认启用。

## 单次启动的混档激活量化（2026-10-23）

### 实现与官方参考

`vllm/model_executor/kernels/linear/bitblend/sm120/activation.py` 实现一个
CuTe DSL kernel，混档线性层不再逐 span 启动量化。GEMM 仍独立执行。
重点参考 FlashInfer 0.7.2 的 `mxfp4_quantize.py`、`mxfp8_quantize.py`，
直接导入 `quantization_cute_dsl_utils.py` 和 `fp4_common.py` 中的官方
向量加载、BF16 max-abs、线程归约、UE8M0 ceil 缩放、FP4/FP8 转换及
128x4 scale 索引函数，没有重新实现舍入规则。

每个 CTA 根据 descriptor 选择一种格式：Low/Mid 为 1 thread / 32
元素；High 默认 4 threads / 32 元素，全部 High span 的有效 scale
块数达到 65,536 时改为 2 threads。这是官方 MXFP8 大输入合作方式的
应用，数学输出不变。padding rows 与有效 rows 一起并行分配，避免
短 decode 时少数 CTA 串行清理大量 scale。每个 span 保留独立的
payload、scale 原点和 128-row padding，重复档位和任意顺序都合法。
任务采用有界网格和 grid-stride 循环，无 shared memory 或 block 同步。

`prepare_activation` 验证输入、分配输出并上传 6-int64/span descriptor；
`ActivationCall.run` 只执行一次量化 launch，支持 Graph replay 和输入
原地更新。BF16 输入要求连续且 16 字节对齐，K 是 128 的倍数；scale
索引受 int32 上限检查，输入地址使用 int64。K/High 合作数选择编译
实例，M 和 span 数动态。两种实例在模型加载期编译。固定档位量化
仍调用官方 public API，混档统一使用新实现，不按形状回退逐 span。

### 测量结果

使用真实 Qwen3-0.6B layer-zero 的 QKV、O、gate/up、down 权重；
对应 K 为 1024、2048、1024、3072。官方基线直接调用相同的官方
CuTe DSL 编译实例，保留 PDL；输出预分配，避免把分配差异计入
kernel 测量。融合量化未启用 PDL。各 Graph 包含 100 次操作，
7 轮交替正反测量顺序；CUDA Event 耗时除以 100 后取中位数。
完整线性使用相同的默认 grouped GEMM，包含量化，未使用离线表。

| span 组合 | 量化加速比（四种投影范围） | 量化+GEMM 加速比 |
| --- | ---: | ---: |
| low/1, mid/1, high/1 | 1.27–1.60x | 1.061–1.089x |
| low/256, mid/256, high/256 | 1.58–1.74x | 1.027–1.065x |
| low/3, mid/9, high/129, mid/17 | 2.15–2.20x | 1.068–1.093x |
| low/2048, mid/2048, high/2048 | 1.07–1.21x | 0.995–1.017x |

例如 QKV decode 的量化为 1.975 → 1.518 us，完整线性为
8.653 → 8.073 us；QKV prefill 的量化为 3.570 → 2.051 us，
完整线性为 29.785 → 27.980 us。大 batch 完整线性基本持平，
gate/up 中位数约退化 0.48%；不能将量化本身的加速比直接解释为
模型或服务加速比。原始大 batch 样本仍有时钟漂移，保留全部样本。

另测 eager 完整线性，包括分配、descriptor 构建/上传和 CPU 提交间隙；
具体数据保存在 `activation.json`。这类 Event 时间不是纯 kernel 耗时。
所有结果为固定权重复用的微基准，不是 HTTP 吞吐或冷 HBM 性能。
可通过 `benchmark_bitblend.py --compare-activation` 复现。

### 正确性和模型回归

- 全部 BitBlend 测试：136 passed、26 skipped。
- 新量化覆盖 K=128/640/1024/3072、小/大 M、单档、缺档、重复 High/Mid、
  两种 High 合作大小、零、正负数和小量级输入。Graph 多次更新输入、
  将输出预填为 255 后 replay，payload 和完整 scale（包括 padding）
  均与官方 CuTe DSL 逐字节一致。
- 与官方逐 span 量化使用相同 GEMM 配置，四类投影输出逐位一致。
- Compute Sanitizer memcheck、synccheck：各 9 项测试通过，均 0 errors；
  memcheck 关闭 CUDA API 错误报告，kernel 内存检查保持开启。
- Ruff check/format 和 `git diff --check` 通过。
- 真实模型：Qwen3-0.6B，eager 混档、async scheduling、chunked prefill、
  decode/prefill 同批、32 KV blocks 强制抢占；164 次混档 forward、
  4 次抢占，受控对照 9/9、同档批处理 12/12 token 一致，正常退出。
  未加载离线表。12 个原始混合请求以及对应单请求的生成 token，
  与本次改动前记录完全相同。

历史长短混合/单请求对照仍为 10/12 一致，最大已选 logprob 差
0.295357；本次没有解决跨 batch 形状或 split-K 配置的数值一致性。
未重新跑完整 WikiText PPL、HTTP 基准，也未实现混档整模型 Graph；
只验证预准备量化及 GEMM 的 Graph replay。

产物目录：`/root/quant/validation/bitblend-mixed-activation/`。
四个形状子目录中的 `activation.json` 保存 fingerprint 和全部计时；
`tests.log`、`memcheck.log`、`synccheck.log`、`model.log` 和
`offline-async-preemption.json` 保存验证证据。

此前 12 项 grouped 配置已在当前 fingerprint 下重新独立复验，
可加载表为本目录的 `verified-database.json`。只迁移历史配置值，
所有旧计时从中间配置文件移除；新输入 oracle 检查与计时均重新执行。
各形状的 `grouped-configs.json` 保存来源 fingerprint 和历史报告 SHA256，
`grouped-verified.json` 保存新验证结果。未替换或全局启用该表，使用时
将 `VLLM_BITBLEND_SM120_TACTICS` 指向新文件。带 split-K 表的跨配置
生成差异仍适用上节限制，不将本轮无表模型回归视为该问题已解决。

## 混档整模型 CUDA Graph（2026-10-23）

### 设计与启用

复用原生 `CUDAGraphWrapper`、token 桶、attention metadata、warmup、图内存池和
重放调度，不按每一种三档比例建立额外的模型图。`BitBlendBackend` 提供 workspace
创建接口，runner 不直接依赖 SM120 布局。SM120 的 `graph.py` 为每个不同的 K
准备可复用的量化 payload/scale arena，为每个 K/N 准备地址固定的描述符。
每次 forward 前合并成一次 pinned H2D 元数据上传；整模型重放期间不执行逐层
Python 描述符构造或 H2D 上传。

量化与 grouped GEMM 继续分开，各一次 launch。捕获的是任务容量，实际工作量
读取设备描述符尾部；未使用的区间保持零工作量。每个 CTA 的档位分支仍统一。
同一个图可覆盖缺失档位、单档、重复档位和三档比例变化。最多十二个 span
对应三档乘以原生四个 attention 区域。padding 延长最后一个 span，实际请求
和 KV/sample 映射不变。量化 arena 跨层顺序复用，GEMM 输出仍由图内存池分别
分配，避免覆盖残差。异步调度沿用原生 compute stream 的串行模型执行。

在 `mixed_batch=true`、`backend=sm120`、`VLLM_USE_V2_MODEL_RUNNER=0` 下，
删除 `--enforce-eager` 并使用：

```bash
--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'
```

`FULL` 覆盖满足原生图条件的 prefill 和 decode；也验证了 `FULL_DECODE_ONLY`。
超出捕获范围的批次使用 eager 回退。Dynamo/piecewise、其他硬件、TP/PP、
LoRA、MoE、推测解码等原有不支持的组合仍被明确拒绝。

Graph 以 padded token 数选择同一 grouped 模板中的默认 tile，使用一波
persistent CTA、无 Split-K；三档比例改变不会改变已捕获的 specialization。
图模式不套用 eager 的精确 span 调优表。图模式中的 eager 回退也使用同一
workspace 和 tile 策略；`--enforce-eager` 仍保留原来的精确 span 查表路径。

### 正确性、整模型与 HTTP

产物目录：`/root/quant/validation/bitblend-mixed-graph/`。

- 完整 BitBlend 测试：140 passed、26 skipped；runner 重排和 padding 测试
  2 passed。Ruff check/format 与 `git diff --check` 通过。
- 两层共用 K 的真实量化权重测试：同一个图从捕获时的全 Low 切换到全 High、
  全 Mid、不均衡三档和十二个重复区间；更新输入并污染 workspace 后，仍与
  相同 tile 的官方量化加 grouped GEMM 逐位一致。覆盖 High 的 2/4 线程协作。
- Compute Sanitizer：混档量化/线性/Graph 共 11 项 memcheck、synccheck 均通过，
  各零错误；收紧 launch 容量后，两个 Graph 测试再次各零错误。
  memcheck 关闭 CUDA API 错误报告，kernel 内存检查开启。
- 真实 Qwen3-0.6B、TP1、Triton attention、原生 async scheduling、prefix cache、
  32 KV blocks 强制抢占。每种运行配置均覆盖 164 次混档 forward、4 次抢占、
  chunked prefill 和 prefill/decode 同批；请求档位全程保持正确。

| 配置 | FULL 图重放 | eager forward | 区间组合数 |
| --- | ---: | ---: | ---: |
| FULL，捕获至 96 tokens | 1,300 | 0 | 41 |
| FULL，捕获至 32 tokens | 1,270 | 30 | 44 |
| FULL_DECODE_ONLY | 1,233 | 67 | 47 |

FULL 主验证使用 1/2/4/8/16/32/64/96 八个原生 token 桶，实际运行均命中
已捕获图，没有因档位比例变化重新捕获。上述三种配置的受控生成对照均为
9/9 一致，同档批处理均为 12/12 一致。FULL 主验证的十二个混档请求及其
单请求对照，token 数组与本次修改前的量化融合基线全部一致。
原始长短请求混合/单请求对照仍为 10/12，保留已有 batch 数值差异边界，
不承诺任意批次形状下严格生成一致。本轮未重新测 WikiText PPL。

标准多进程 HTTP 服务另行验证：12 个并发三档 completion、三档并发流式
chat、非法档位拒绝均通过，共 15 个成功响应。服务退出码为 0。
此轮为图执行功能和回归验证，未将结果描述为吞吐或尾延迟性能评估。

复现脚本、完整 graph/区间 trace 和原始日志位于产物目录；`final.sh`
运行完整测试、最终 Graph sanitizer 和真实模型检查，`check_http.py` 验证
标准 HTTP 服务。源码指纹变化后通过 `reverify.py` 重新验证 eager 调优配置；
旧表不能通过修改 fingerprint 直接沿用。

本轮 decode、prefill、uneven 的 12 项 eager grouped 配置已全部完成当前源码下的
独立正确性与计时复验，可加载表为产物目录的 `verified-database.json`。
该表仅服务原有 eager 精确区间查表路径，不改变上述 Graph tile 策略。

## 混档整模型 Graph 离线调优（2026-10-23）

本节替代上一节的默认 tile 限制。沿用同一 grouped kernel 模板和原生模型图，
新增独立的 `graph_selected` 查表区，按十二个 span 的容量和精确 padded M/N/K
选择配置；同一捕获桶内改变请求比例仍然重放原图。表外形状保留默认配置。
每档可独立选择 tile 和 Split-K，支持 waves、M/N 遍历顺序、heavy-first 和
CTA residency；workspace、任务容量和图内归约均按所选配置准备。
全部 specialization 在模型加载阶段预准备，未增加逐层 replay Python 调度。

### 搜索与独立复验

产物目录：`/root/quant/validation/bitblend-graph-tune/`。
复现命令和参数说明见 [数据库文档](bitblend-sm120-database.md#whole-model-graph-tuning)。

- 使用 Qwen3-0.6B layer-zero 真实权重，覆盖 QKV、attention output、gate/up、down
  四种几何和 M=`1,2,4,8,16,32,64,96,128,256,512`，共 44 项。
- 测量 2,487 个候选配置/形状组合。四种 CTA 线程数种子均先测量，再对一个家族
  做一轮、每档最多十二个邻居的坐标搜索，随后搜索 residency 和任务排序。
  参数轴包含全部已实现选择，但本次预算没有穷举完整笛卡尔积，不声称全局最优。
- 计时覆盖实际容量路径上的融合激活量化、grouped GEMM 和可选归约；描述符
  上传在计时外。每个候选捕获一次，同一图依次验证单档、缺档、偏斜比例和
  最多十二段重复档位。搜索为五轮、图内五十次；独立复验为七轮、图内一百次，
  更换输入种子并加入额外随机分区。每次测量先重放一百次预热。
- 目标是各分区延迟比的等权几何平均，至少改善 1%，且任何已测分区退化不超过
  10%；否则保留默认配置。复验后 33 项采用新配置，11 项保留默认。
  实际最差分区延迟比为 1.09786，因此平均提升不代表所有比例均变快。
- 最终所选配置相对独立 FP32 解码计算的最大 relative L2 为 `8.48e-5`
  （阈值 `0.001`）。所有测量候选也接受独立算术检查，数值失败会终止搜索。

下表是每种线性层在十一种 M 上的加速比几何平均，基线为原 Graph 默认配置。
包括量化与归约，不是纯 GEMM 或整模型吞吐；权重重复使用，也不是冷 HBM 测量。

| 线性层 | 平均加速比 | 单个 M 的最大加速比 | 保留默认的 M |
| --- | ---: | ---: | --- |
| QKV | 1.130× | 1.438× | 128, 256 |
| Attention output | 1.267× | 1.812× | 8, 128 |
| Gate/up | 1.100× | 1.457× | 4, 8, 32, 128 |
| Down | 1.309× | 1.856× | 4, 8, 128 |

可加载文件为产物目录的 `verified/mixed-graph-verified.json`，SHA-256：
`11c8891ef9fce0ae85bbc30aab81c41dc78ffebf1835a737b68423fde5ed61d6`。
未全局启用该表，使用时设置：

```bash
export VLLM_BITBLEND_SM120_TACTICS=/root/quant/validation/bitblend-graph-tune/verified/mixed-graph-verified.json
```

`search/mixed-graph.json` 保存完整搜索；verified 文件保存新的输入、计时、
额外分区、重选结果、来源报告 hash 和环境/源码 fingerprint。
该文件只包含 Graph 配置，未把旧 fingerprint 的 eager 表直接拼入。

### 回归与模型验证

- 完整 BitBlend pytest：142 passed、26 skipped。
- 四个两层 Graph replay 用例覆盖默认/调优配置和两种容量，调优路径包含不同档位
  tile、不同 Split-K、排序和 residency；同一图切换到全 High、全 Mid、不均衡
  和十二段后，与相同配置的官方量化加 grouped GEMM 逐位一致。
- 上述四例在 Compute Sanitizer memcheck、synccheck 下各零错误；memcheck
  关闭 CUDA API 错误报告，kernel 内存检查开启。
- Qwen3-0.6B、TP1、FULL、Triton attention、async scheduling、prefix cache、
  32 KV blocks 强制抢占：1,300 次实际图重放、164 次混档 forward、4 次抢占，
  chunked prefill 和 prefill/decode 同批均覆盖。八个捕获桶的 32 个几何配置
  全部命中 verified 表。
- 相同输入、padding 和调优配置下，保留原生捕获但绕过图重放，43 次生成调用的
  全部 token 数组与 Graph 完全相同。两个进程记录的配置也相同。
- 捕获范围扩大到 512 tokens、最大序列长度 512、40 KV blocks 后，实际执行
  1,246 次 FULL 图重放（含 512-token 桶）、137 次混档 forward、3 次抢占，
  全部 44 项几何配置命中 verified 表。该组没有触发 chunked prefill，相关覆盖
  来自前述 96-token 组。初始脚本错误地用 384 的最大序列长度捕获 512 桶，
  在原生 Triton attention 的 dummy warmup 中越界；同步定位后调整验证配置，
  保留原始失败日志，未修改 kernel 或调优表。
- 跨 batch/桶对照中，长短混合/单请求为 8/12、受控三档/单请求为 6/9、
  同档批处理/单请求为 10/12 token 一致；绕过图重放时结果相同。
  不同桶的 tile/Split-K 会改变浮点归约顺序，本表不能承诺任意 batch 下生成
  严格相同。此次未重跑完整 WikiText PPL，未将历史 PPL 作为新表质量结论。

`model-checks.sh`、`model.py`、`offline-*.json` 保存复现步骤和全部 token/配置/
重放 trace；`tests.log`、`memcheck.log`、`synccheck.log` 保存回归证据。
本轮未重测 HTTP 吞吐或端到端尾延迟。

## 最新 HTTP 端到端评估：固定 2048 输入 / 512 输出

产物目录：`/root/quant/validation/bitblend-e2e-2048-512/`。
`summary.md` 为完整报告，`benchmark.py` 保存复现命令，`verification.json`
保存独立校验；原生客户端详细结果、服务日志、Prometheus 快照和显存采样均保留。
本轮只增加测量与文档，未修改 kernel 或运行时。

Qwen3-0.6B、RTX 5090、TP1、并发 32，原生 `vllm serve` 加
`vllm.benchmarks.serve`。每组预热 33 请求，测两轮、每轮 96 请求，固定
2048-token prompt、512-token 输出、忽略 EOS，随机 token 负载不用于质量评估。
混档按 Low/Mid/High 轮转分配，每档共 64 个测量请求；实时活动比例由原生调度决定。
下表为两轮算术平均。

| 档位 | Prefill 单请求等效 token/s | Decode 单请求等效 token/s | 整体输出 token/s | 加载模型 GiB | 服务峰值 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 34,526 | 137.24 | 4,165.3 | 1.12 | 14.623 |
| MXFP4 | 60,260 | 141.96 | 4,386.3 | 0.54 | 14.047 |
| MXFP8 | 52,783 | 143.41 | 4,413.3 | 0.73 | 14.223 |
| Low | 50,560 | 140.41 | 4,319.1 | 1.22 | 14.657 |
| Mid | 46,586 | 138.06 | 4,236.8 | 1.22 | 14.657 |
| High | 36,880 | 133.21 | 4,056.0 | 1.22 | 14.657 |
| 等比例混档 | 41,544 | 130.90 | 4,009.9 | 1.22 | 14.657 |

Prefill 为 `2048 / mean(request_prefill_time)`，Decode 为
`511 / mean(request_decode_time)`，首 token 归于 prefill。这两个数是并发负载下
单请求等效速度，排除初始服务端排队，包含调度/分块交错影响；请求阶段相互重叠，
不能解释为 GPU 独立阶段总吞吐。整体输出吞吐为客户端输出 token 总数除以墙钟时间，
包含排队与传输；输入加输出总吞吐为其五倍。

服务峰值为测量阶段每 20ms 采样的 NVML 全设备显存，包含统一预分配的 **12 GiB
KV cache**、Graph、context 和 allocator。加载模型显存来自服务加载日志。
BitBlend 保留共同的多平面权重，单档不释放其他平面，三档显存相同符合当前实现。
最初沿用 6 GiB KV 配置发现抢占后停止，旧数据归档 `kv6-diagnostic/`，不纳入本表。

MXFP4/MXFP8 加载 BF16 模型并在启动时在线量化权重，量化时间不纳入请求计时。
Low/Mid/High 和混档全部设置 `mixed_batch=True`，使用当前统一 grouped GEMM、
融合激活量化与整模型 Graph 路径。关闭 prefix cache，开启 async scheduling，
使用 Triton attention，max_model_len=4096、max_num_seqs=32、
max_num_batched_tokens=2048。
FULL 捕获桶为 `1,2,4,8,16,32,64,96,128,256,512,1024,1536,2048`。
加载上一节 verified Graph 表；该表精确调优至 M=512，三个更大的 prefill 桶
使用当前同一 kernel 模板的默认策略，尚未离线搜索。

独立检查 14 轮、共 1,344 个请求的实际输入/输出长度、token 总量、吞吐公式、
Prometheus 阶段样本数及时间分解：零失败、零抢占、零 prefix cache 命中，
脚本与调优表 hash 匹配。实际执行日志全部记录为 FULL，包含 2048-token prefill 图。
所有服务退出码为 0，无强制退出；BF16、MXFP4、High 仍在关闭触发之后记录
`EngineDeadError`，测量期间未发生，不能据此声称关闭阶段问题已修复。

整体输出吞吐相对 BF16：MXFP4 +5.31%、MXFP8 +5.95%、Low +3.69%、
Mid +1.72%、High -2.62%、混档 -3.73%。这反映当前模型、长度、并发和实现，
不能推广到所有负载；两轮短测的微小差异不代表统计显著。

## 最新融合量化、grouped GEMM 与 FULL Graph 的质量和稳定性

产物目录：`/root/quant/validation/bitblend-latest-quality-stability/`。
`summary.md` 为完整报告；`quality.py`、`stability.py` 为复现入口，
`manifest.json` 保存源码、模型、脚本与调优表 hash，`verification.json`
保存独立交叉校验。本轮未调优或修改 kernel/运行时。

### 完整 WikiText2 质量

使用 lm-evaluation-harness 原有 WikiText 文档预处理、rolling windows、
prompt logprob 解析及 word PPL 聚合，完整 test 62 篇、311 个窗口，
每档计分 284,651 tokens，context=1024。所有档位使用同一 tokenizer、
语料 fingerprint `3b100fc026da3bad` 和计分窗口 hash。
服务模型最大长度 4096、batch 上限 8、每步 2048 tokens，2 GiB KV cache；
TP1、Triton attention、async scheduling、关闭 prefix cache。
BitBlend 使用 `mixed_batch=True`、最新融合激活量化、grouped GEMM、
FULL Graph 和前述 verified Graph 表；捕获桶最大 2048，与端到端性能测试一致。

| 档位 | Word PPL（越低越好） | 相对 BF16 |
| --- | ---: | ---: |
| BF16 | 32.6866 | 基线 |
| MXFP4 | 63.8732 | +95.41% |
| MXFP8 | 33.2757 | +1.80% |
| Low | 63.8732 | +95.41% |
| Mid | 44.8732 | +37.28% |
| High | 33.1370 | +1.38% |

Low 与原生 MXFP4 全部目标 token 对数概率逐位一致。混档让每个窗口分别经过
Low/Mid/High，并逐 forward 检查请求档位与 token span 一致；实际观察到
463 个混档 forward。每档在单档、混档、相同形状绕过 Graph 重放三种方式下，
全部目标 token 对数概率逐位一致。bypass 保留相同捕获模型、workspace 和
tile，仅绕过重放调用，因此不会混入更换配置的浮点舍入差异。

本次观察不到混档或 Graph 额外引入的计分误差，但 Low/Mid 相对 BF16 的质量
损失仍明显。实现一致性不等于量化无损；该结果也不证明其他 batch、模型或
指令/数学/中文任务的质量。`windows-*.json` 保留完整目标 token 概率，
`quality-*.json` 保存逐文档分数、PPL、执行配置和 Graph 桶计数。

### HTTP 稳定性与恢复

三组均使用当前 Graph 调优表、三档混合请求和 FULL prefill/decode Graph。
生产组沿用上节性能配置；压力组开启 prefix cache、限制为 192 KV blocks
（3072 tokens）、最大 16 条活动序列、每步 512 tokens，以触发实际抢占。

| 验证 | 持续时间 | 普通压测完成请求 | KV 抢占 | 预热后显存变化 |
| --- | ---: | ---: | ---: | ---: |
| 生产配置 | 901.9 秒 | 17,344 | 0 | 0 MiB |
| 有限 KV 压力 | 241.3 秒 | 2,976 | 463 | 0 MiB |
| 独立重启 | 30.7 秒 | 608 | 0 | 0 MiB |

生产组并发 32，轮流纯 Low/Mid/High、均衡混档、缺 Mid、偏 Low 六种模式，
输入从 `32,64,128,511,1024,2048`、输出从 `1,16,64,128,512` 中组合；
先用 32 个 2048+512 请求预热，避免把 allocator 首次扩容误判为泄漏。
压力组在较短长度范围内持续运行，463 次抢占后全部请求仍能完成。

每组另覆盖 24 个正常流式请求（completion/chat、三档）、24 个主动断开流、
12 个同时完成的正常请求，三种非法档位返回 400，省略档位与显式 Low 的
默认行为一致。restart 和 pressure 开启请求日志，分别确认 24 次引擎 aborted；
但 Prometheus `finished_reason=abort` 计数仍为零，因此不以该指标作为取消
成功依据，且不声称该指标已修复。取消后 running/waiting 均归零。

prefix-cache 隔离验证：相同 prompt 首次使用各档均为零命中，随后同档重复
分别命中 496 tokens，输出与该档第一次调用一致。三组无运行期 EngineDeadError，
退出码均为 0、无强杀，关闭后显存回到启动前基线；本次三组关闭也未记录该错误，
不代表此前偶发关闭问题已修复。

完整 BitBlend 回归：142 passed、26 skipped。四个 mixed Graph 更新/重放
用例在 Compute Sanitizer memcheck、synccheck 下分别零错误。
上述为当前单模型、单 GPU 的有限稳定性证据，未完成数小时/数天老化或跨设备验证。

## 跨硬件接口整理回归（2026-10-23）

本次仅整理公共接口与打包，未改动 CUDA kernel、量化规则和权重格式。
backend 选择统一检查设备/profile/固定档与混档 Graph 能力；公开
`PreparedLayer.request` 和 `MixedGraphWorkspace` 协议，runner 不再读取设备
私有 state 或使用 backend 名称白名单。共享配置校验移至量化配置模块，
新增架构的 JIT 源码按目录自动打包。接入步骤见
[Adding a hardware backend](features/quantization/bitblend.md#adding-a-hardware-backend)。

验证产物：`/root/quant/validation/bitblend-backend-refactor/`。

- BitBlend 回归：145 passed、26 skipped；原生 batch 重排与 prefix-cache
  隔离测试：3 passed。双 FP8 两档 profile 和 opaque backend state 均覆盖
  公共接口测试，但没有运行 H100/B200 硬件 kernel。
- 实际执行 setup.py 的打包命令：SM120 与第二架构测试目录各 5 份源码均复制
  且 hash 一致，缺少可选源码树也正常；第二目录只为临时测试 fixture。
- 全量 WikiText2 62 篇、311 窗口、每档 284,651 个目标 token。
  Low/Mid/High 的单档、混档、相同形状 Graph bypass 共九组目标 token
  对数概率均与重构前逐位一致。PPL 分别为 63.873199912676846、
  44.8732301625724、33.136959302950764。
- 独立多进程 HTTP 混档 FULL Graph 回归：预热后 31.1 秒完成 640 个普通
  请求，另覆盖 24 个流式请求、24 次取消、取消期间的正常请求和非法档位。
  预热后显存变化 0 MiB，结束后活动请求为零，记录 24 次引擎 aborted，
  服务退出码 0、无 EngineDeadError，显存回到启动前基线。该短测用于接口
  回归，不替代上节更长的稳定性验证。

修改 backend.py 会使历史调优表指纹失效，本次先确认旧表被拒绝，再提取历史
候选配置作为未验证输入，使用原独立 verifier 重新计时和检查新输入/留出混档
比例；未继承旧表的 verified 状态，也未重新执行完整参数搜索。
新表 `verified/mixed-graph-verified.json` 包含 44 条已验证配置，SHA-256 为
`a5a40a13d19eed1febbd4ffcebc347833aa4f53d5ab5b2b4e6128042e27bb3ee`。
启动时应将 `VLLM_BITBLEND_SM120_TACTICS` 指向上述产物目录中的新表。
旧表保留为历史记录，不再适用于当前源码。

## 实现清理回归（2026-10-23）

本轮审查范围为 BitBlend 量化适配、backend、SM120 kernel 调用、原生 vLLM
接入点及其测试、离线工具和文档。删除未使用的 `GraphKey`、`ResourceBudget`、
`WorkspaceSpec`/空 workspace 查询和调用参数占位；固定档覆盖计划只保留实际
使用的 profile/tier。backend 私有状态不再重复保存 `PreparedLayer.request`，
eager 与 Graph 共用 grouped descriptor 排序逻辑，离线 Graph 调优也改用公开
`PreparedLayer` 接口。未改动 CUDA kernel、权重编码或激活量化规则。

固定档 Mid/High 入口、混档 grouped GEMM、融合激活量化和 reference 数值验证
均保留；它们仍有实际调用用途。无同档请求独占 batch 的替代 scheduler。
当前目录职责及跨硬件扩展步骤见
[BitBlend framework](features/quantization/bitblend.md#extension-boundaries)。

验证产物：`/root/quant/validation/bitblend-cleanup/`。

- BitBlend 及原生 batch 重排/prefix-cache 隔离回归：144 passed、26 skipped、
  109 deselected。相关 24 份 Python 文件通过 Ruff 检查与格式检查。
- 对旧表候选重新独立计时、验证后生成 44 条配置；未继承旧 verified 状态，
  未重新执行完整参数搜索。
- Qwen3-0.6B 全量 WikiText2：三档分别测试单档、混档和 Graph bypass，
  共九组、每组 311 窗口/284,651 个目标 token 的对数概率均与清理前逐位一致。
  Low/Mid/High PPL 仍为 63.873199912676846、44.8732301625724、
  33.136959302950764；比对结果保存在 `regression.json`。
- 独立多进程 HTTP 混档 FULL Graph 短测：预热后 30.6 秒完成 608 个普通
  请求，另覆盖 24 个流式请求、24 次取消、取消期间正常请求及非法档位。
  预热后显存变化 0 MiB，结束后活动请求为零；日志记录 24 次 aborted。
  服务退出码 0、无 EngineDeadError、退出后显存回到基线。这是本次清理的
  短期回归证据，不替代长期稳定性或跨设备验证。

当前源码应使用
`/root/quant/validation/bitblend-cleanup/verified/mixed-graph-verified.json`，
SHA-256：`7710b901cdc5e53b8e8ee361355375077afb6efba3528d28f4820c736bf9fe50`。
`VLLM_BITBLEND_SM120_TACTICS` 需指向此表，上一节的表仅保留为历史记录。

## 包内默认调优表（2026-10-23）

默认调优表已随包分发：
`vllm/model_executor/kernels/linear/bitblend/sm120/configs/rtx5090.json`。
未设置 `VLLM_BITBLEND_SM120_TACTICS` 时，仅在完整设备、软件环境和源码指纹
匹配时自动启用；否则使用默认配置。显式设置的外部表优先，仍严格拒绝未验证
或指纹不匹配的表。此表包含 44 条 mixed Graph 配置，未覆盖的形状、固定档和
eager 路径仍使用各自默认配置。

加载代码变化后，先确认旧表被拒绝，再对历史候选重新独立计时、检查新输入及
留出混档比例，生成新的 verified 表；没有直接改写旧表指纹。
外部完整产物位于 `/root/quant/validation/bitblend-bundled-table/`。
包内文件与新 verifier 输出逐字节一致，SHA-256：
`47a6200903d0e117885b90c6e5ab3722d0963e43669825aa0078701606df1b6f`。
前述外部旧表不再适用于当前源码；当前环境正常启动无需设置表路径。

- 实际执行 setuptools 的 package-data 复制，包内目标存在且 SHA-256 一致；
  无需完整重编译 CUDA 扩展。
- 完整 BitBlend 及原生重排/cache 隔离回归：146 passed、26 skipped、
  109 deselected。新增用例检查默认表精确匹配、环境不匹配回退、外部路径优先
  和缺少包内表时的行为。
- 清除表路径环境变量，先断言默认解析指向包内表且含 44 条 Graph 配置，再跑
  全量 WikiText2。三档单档、混档和 Graph bypass 共九组、每组 311 窗口/
  284,651 个目标 token 的对数概率与清理前逐位一致；PPL 不变。
- 无表路径环境变量的独立多进程 HTTP 混档 FULL Graph 短测通过：30.6 秒
  完成 608 个普通请求，另覆盖流式请求、24 次取消及非法档位；预热后显存
  变化 0 MiB，服务退出码 0，无 EngineDeadError，退出后显存回到基线。
