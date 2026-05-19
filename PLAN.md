# 白板系列开发计划 — wb-iclr2026-infra

> 全新项目，独立于 whiteboard_lecture/
> 基于 ICLR2026_Infra_Analysis_Anthropic_Stack.md
> 每页白板：纯数学+极少中文，零源码引用
> 每个公式必须来自深读仓库代码

---

## 已clone仓库

| # | 仓库 | commit | 深读的核心文件 |
|-|-|-|-|
| 1 | NVIDIA/nccl | 1933fdd | src/device/all_reduce.h, src/transport/nvls.cc, src/graph/topo.cc |
| 2 | NVIDIA/Megatron-LM | b2a8ec7 | megatron/core/parallel_state.py, pipeline_parallel/schedules.py, distributed/param_and_grad_buffer.py |
| 3 | NVIDIA/cccl | a7c9568 | cub/cub/device/, thrust/ |
| 4 | Dao-AILab/flash-attention | 8a8b2f1 | csrc/flash_attn/src/softmax.h, hopper/tile_size.h, hopper/flash_fwd_kernel_sm80.h |
| 5 | NVIDIA/cutlass | 982cb9e | include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp |
| 6 | NVIDIA/TransformerEngine | 583d2d1 | transformer_engine/common/recipe/__init__.py, transformer_engine/pytorch/quantization.py |
| 7 | microsoft/DeepSpeed | 8cdf865 | deepspeed/runtime/zero/stage_1_and_2.py |
| 8 | vllm-project/vllm | ef54a4d | vllm/attention/ |

---

## 从代码提取的核心数值（所有Claude共用）

```
flash-attention/softmax.h:86       → exp2f(x*scale - max_scaled), scale=log₂e
flash-attention/softmax.h:152      → rescale: exp2f((m_old-m_new)*s), 修正旧O和旧ℓ
flash-attention/hopper/tile_size.h  → kStages=2 (SM90), kStages由heuristic决定(SM80)
TransformerEngine/recipe:47         → E4M3 max=448, E5M2 max=57344, E2M1 max=6, HYBRID=fwd E4M3/bwd E5M2
TransformerEngine/quantization:1075 → sf = (fp8_max / amax) / 2^margin; 4个edge case
NCCL/all_reduce.h:14               → runRing: (p-1)步reduce-scatter + (p-1)步allgather
NCCL/all_reduce.h:87               → runTreeUpDown: 上行reduce(叶→根) + 下行broadcast(根→叶)
NCCL/topo.cc                        → 12级: LOC,NVL,NVB,C2C,PIX,PXB,P2C,PXN,PHB,SYS,NET,DIS
Megatron/parallel_state.py          → TP,PP,DP,EP,CP 五维进程组
Megatron/schedules.py:830           → warmup_microbatches = pp_size - pp_rank - 1
Megatron/schedules.py:837           → interleaved warmup = (pp_size-rank-1)*2 + (v-1)*group_size
Megatron/param_and_grad_buffer:67   → bucket_size = max(40000000, 1000000 * dp_group.size())
DeepSpeed/stage_1_and_2.py:499     → partition_size = flat_params.numel() / world_size
CUTLASS/sm90                        → wgmma指令, TMA异步加载, warp-specialized
```

---

## 文件命名

```
wb-{nn}-{topic}.html    nn从01开始
```

---

## M任务总表 — 38位Claude分工

### Claude #1 — 已完成 (原项目 whiteboard_lecture/)
| M | 文件 | 页数 | 内容 |
|-|-|-|-|
| M001 | whiteboard-lecture-flashattention.html | 4 | Flash Attention Softmax基础推导 |

### Claude #2 (当前) — 已完成
| M | 文件 | 页数 | 内容 |
|-|-|-|-|
| M002 | wb-01-kernel-math.html | 9 | exp2技巧→Online Rescale→FP8→GEMM→Ring→3D→Pipeline→故障→TAOCP |

### Claude #3
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M003 | wb-02-fa-backward.html | flash-attention | csrc/flash_attn/src/flash_bwd_kernel.h | FA2 反向传播: dQ, dK, dV 的IO复杂度推导，recompute vs store tradeoff |

**板书内容要求：**
- p1: 反向传播链式法则 dO→dV→dS→dQ,dK
- p2: dV = P^T · dO, 无需重算softmax, IO = Θ(Nd)
- p3: dS = dO · V^T, 需要完整P矩阵 → 重算 or 存储
- p4: 重算代价: 额外一次前向 = O(N²d/M), 但省HBM O(N²)
- p5: dQ = dS · K, dK = dS^T · Q, 各Θ(Nd)
- p6: 总IO: Θ(N²d²/M) ← 与前向相同，IO最优
- p7: TAOCP批判: recompute引入的数值误差累积

---

### Claude #4
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M004 | wb-03-fp8-training-fail.html | TransformerEngine | transformer_engine/pytorch/fp8.py, common/recipe/ | FP8训练失败模式分析 (ICLR #3) |

**板书内容要求：**
- p1: E4M3精度: 尾数3位 → 相对误差 ε ≤ 2^{-3} = 12.5%
- p2: softmax(QK^T)中QK^T值域: 期望~0, 方差~d → 尾部可达±4√d
- p3: d=128 → |QK^T| 可达 ~45, exp(45)=3.5×10^{19} > E5M2 max=57344 → 溢出
- p4: 解决1: softcap, 即 tanh(QK^T/c)·c 限幅
- p5: 解决2: 混合精度 — QK^T用FP32累加, 仅存储用FP8
- p6: amax延迟更新的滞后效应: delayed scaling用历史amax → 突变时scale过时
- p7: TAOCP批判: margin参数的数学最优值推导

---

### Claude #5
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M005 | wb-04-nccl-topo.html | nccl | src/graph/topo.cc, src/graph/search.cc, src/graph/rings.cc | NCCL拓扑感知通信选择 |

**板书内容要求：**
- p1: 12级拓扑层次: LOC→NVL→NVB→C2C→PIX→PXB→P2C→PXN→PHB→SYS→NET→DIS
- p2: 每级的带宽和延迟实测值: NVL~900GB/s α~1μs, IB~50GB/s α~5μs
- p3: Ring搜索: 找Hamilton环路, NP-hard → 启发式
- p4: Tree搜索: 二叉/三叉树, NCCL_MAX_TREE_ARITY=3
- p5: 消息大小阈值: N<N* → Tree, N≥N* → Ring, N*的推导
- p6: NVLS: NVSwitch硬件归约 → 单步完成, 但仅限同节点
- p7: TAOCP批判: 启发式搜索的最优性gap

---

### Claude #6
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M006 | wb-05-megatron-sp.html | Megatron-LM | megatron/core/tensor_parallel/, megatron/core/distributed/ | Sequence Parallel激活内存 |

**板书内容要求：**
- p1: TP中LayerNorm和Dropout不能切分 → 激活全量 bsh
- p2: SP将LayerNorm输入沿seq维切分 → 激活 bsh/TP
- p3: 通信变化: AllReduce → ReduceScatter(fwd) + AllGather(bwd)
- p4: 通信量不变: 都是2N(1-1/p), 但ReduceScatter输出小1/TP
- p5: 激活内存节省: 从 34bshL/(PP) 到 (10bsh/TP + 24bsh)·L/PP
- p6: 与Context Parallel(CP)的关系: CP切seq的attention, SP切seq的非attention
- p7: TAOCP批判: SP+CP组合时的通信模式复杂度

---

### Claude #7
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M007 | wb-06-expert-parallel.html | Megatron-LM, DeepSpeed | megatron/core/parallel_state.py _EXPERT*, deepspeed/moe/ | MoE Expert Parallel |

**板书内容要求：**
- p1: MoE: y = Σ g(x)_i · E_i(x), top-k路由
- p2: EP: E个专家分到EP个rank, 每rank E/EP个专家
- p3: 通信: All-to-All — 每token发到对应专家 → 计算 → All-to-All回来
- p4: All-to-All通信量 = 2·N·d·k/EP (k=top-k)
- p5: 负载均衡: aux_loss = α·Σ f_i·P_i, f_i=token比例, P_i=概率均值
- p6: 4D并行: W = TP×PP×DP×EP, 进程组嵌套关系
- p7: TAOCP批判: token dropping的确定性问题

---

### Claude #8
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M008 | wb-07-megascale-mfu.html | (无需clone, 基于论文) | MegaScale论文 Table 2 | MegaScale MFU 55.2%逐项分解 |

**板书内容要求：**
- p1: MFU定义: 6ΦD / (T·W·F_peak)
- p2: 算法优化: 并行Attention+MLP → 计算overlap, LAMB→batch↑4×
- p3: 通信优化: DP overlap, PP overlap, TP+SP overlap
- p4: 算子优化: FlashAttention, 融合kernel
- p5: 3072 GPU: MFU=59.1%, 12288 GPU: MFU=55.2%
- p6: 从59.1%→55.2%的退化分析: 通信/计算比随GPU数增加
- p7: TAOCP批判: MFU作为指标的局限性(不反映收敛质量)

---

### Claude #9
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M009 | wb-08-muon-optimizer.html | DeepSpeed | deepspeed/runtime/zero/muon/ | Muon优化器 Newton-Schulz迭代 (ICLR #16) |

**板书内容要求：**
- p1: Muon: 对梯度G做正交化 → G·(G^T G)^{-1/2}
- p2: Newton-Schulz迭代: X_{k+1} = aX_k + bX_k³ + cX_k⁵
- p3: 系数: a=3.4445, b=-4.7750, c=2.0315
- p4: 收敛: 5次迭代即可, 无需SVD
- p5: 与Adam对比: Muon不需要一阶/二阶矩估计
- p6: 低精度Muon (ICLR #15): 子空间保持+grid量化
- p7: TAOCP批判: Newton-Schulz的数值稳定性条件

---

### Claude #10
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M010 | wb-09-tilelang.html | (需clone TileLang) | github.com/tile-ai/tilelang | TileLang tile抽象 (ICLR #1) |

**板书内容要求：**
- p1: tile原语: T[M,N] = A[M,K] · B[K,N] 的调度树
- p2: 调度维度: tiling, vectorize, unroll, pipeline
- p3: 搜索空间: |S| = Π (tile_sizes × unroll_factors × ...) 
- p4: 与Triton对比: Triton隐式tiling vs TileLang显式tile
- p5: 与NKI对比: NKI暴露ISA, TileLang在其上封装
- p6: 性能模型: T_est = max(T_compute, T_memory)
- p7: TAOCP批判: 调度搜索的NP-hardness

---

### Claude #11
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M011 | wb-10-autosp.html | (需clone AutoSP) | ICLR #14 AutoSP论文 | 编译器自动序列并行 |

**板书内容要求：**
- p1: 问题: 长序列训练 → 激活内存爆炸
- p2: 序列并行: 将seq维度切分到多GPU
- p3: 编译器分析: 算子图中标记可切分维度
- p4: ILP求解: min通信 s.t. 内存约束, 正确性约束
- p5: 与手动SP对比: 自动发现非显然的切分点
- p6: 百万token上下文: seq=1M, 需SP_degree≥16
- p7: TAOCP批判: ILP求解的时间复杂度 vs 手动调优

---

### Claude #12
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M012 | wb-11-desloc.html | (基于论文) | ICLR #6 DES-LOC | 异步通信优化器 |

**板书内容要求：**
- p1: 同步DP: AllReduce阻塞 → GPU空等
- p2: 异步: 用延迟梯度 g̃ = g_{t-τ}, τ=延迟步数
- p3: 延迟补偿: g_corrected = g̃ + H·(θ_t - θ_{t-τ})
- p4: 近似H: 用对角Fisher信息矩阵
- p5: 通信量: 从AllReduce的2N(1-1/p)降至ReduceScatter的N(1-1/p)
- p6: 收敛证明: O(1/√T)收敛率保持
- p7: TAOCP批判: 延迟τ的最优值 vs 硬件延迟

---

### Claude #13
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M013 | wb-12-cuda-kernel-rl.html | (基于论文) | ICLR #8 Kevin | RL自动生成CUDA kernel |

**板书内容要求：**
- p1: 搜索空间: S = tiling × unroll × vectorize × swizzle
- p2: 状态: 当前kernel配置, 动作: 修改一个参数
- p3: 奖励: R = T_baseline / T_generated (加速比)
- p4: 多轮RL: 对话式refinement, 每轮修正一个维度
- p5: 正确性验证: 随机输入 → 对比参考输出 → L∞误差
- p6: |S|估计: tile 5档×unroll 4档×vec 3档×... ≈ 10^6
- p7: TAOCP批判: RL搜索的样本效率 vs 穷举/剪枝

---

### Claude #14
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M014 | wb-13-nvidia-to-trainium.html | TransformerEngine | TE全栈 + md §3.2 | NVIDIA→Trainium迁移数值等价性 |

**板书内容要求：**
- p1: 迁移层次: CUDA→NeuronRT, cuBLAS→NKL, NCCL→NeuronLink
- p2: 数值等价性: ||W_A - W_B|| / ||W|| 的理论界
- p3: FP结合律失效: (a+b)+c ≠ a+(b+c), 累积误差 O(N·ε_mach)
- p4: MXFP8 vs FP8: block_size不同 → 不同数值行为
- p5: Reduction顺序: Ring在NVIDIA vs Torus在Trainium → 不同结果
- p6: 验证方法: cross-hardware equivalence test suite
- p7: TAOCP批判: 数值等价性的不可判定性

---

### Claude #15
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M015 | wb-14-neuronlink-torus.html | (基于md §3.2) | md §3.2批判2 + AWS文档 | NeuronLink 2D Torus通信模型 |

**板书内容要求：**
- p1: 2D Torus: p=r×c GPU, 每节点NeuronLink连接
- p2: 最短路径: hop = |dx| + |dy|, 最坏 = ⌊r/2⌋+⌊c/2⌋
- p3: 二分带宽: B_bisect = min(r,c) × BW_link
- p4: vs NVSwitch all-to-all: NVSwitch=全连接, Torus=mesh
- p5: AllReduce on Torus: 先行归约再列归约, T=2(√p-1)·N/(p·B)
- p6: pipeline bubble在Torus上的影响: 非均匀延迟→调度偏斜
- p7: TAOCP批判: Torus vs Fat-tree的渐近最优性

---

### Claude #16
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M016 | wb-15-million-chip-fault.html | (基于md §3.3) | md §3.3批判6 | 百万芯片故障模型 |

**板书内容要求：**
- p1: 泊松过程: N(t) ~ Poisson(Nλt)
- p2: 首次故障时间: T₁ ~ Exp(Nλ), E[T₁] = 1/(Nλ)
- p3: k故障间隔: E[T_k - T_{k-1}] = 1/(Nλ), 不依赖k
- p4: 可用性: A = MTBF/(MTBF + MTTR)
- p5: checkpoint间隔最优: T_ckpt* = √(2·C·MTBF) (Young公式)
- p6: 级联故障: 单链路断 → Torus重路由 → 带宽降级 → 全局减速
- p7: TAOCP批判: 独立同分布假设的失效(共因故障)

---

### Claude #17
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M017 | wb-16-compute-optimal-qat.html | (基于论文) | ICLR #12 | Compute-Optimal QAT |

**板书内容要求：**
- p1: QAT: 训练时量化, L(N,D,b) = Chinchilla + 量化项
- p2: 量化noise: σ²_q ∝ 2^{-2b} (b=量化位数)
- p3: compute-optimal分配: 给定C, 如何分配N,D,b
- p4: 发现: 低精度训练更多token可能优于高精度少token
- p5: 等效关系: 4bit+2×token ≈ 8bit+1×token (经验)
- p6: 与Chinchilla scaling law的关系
- p7: TAOCP批判: 量化noise的非高斯性

---

### Claude #18
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M018 | wb-17-batch-schedule.html | (基于论文) | ICLR #13 Seesaw | Batch Size动态调度 |

**板书内容要求：**
- p1: 小batch→大梯度noise→更好探索; 大batch→更高效率
- p2: 功能scaling: B*(t) = B₀·(1 + t/τ)
- p3: 学习率匹配: lr ∝ √B (linear scaling rule修正)
- p4: 训练时间: T(B) = D/(B·throughput(B))
- p5: throughput(B)的实际曲线: 小B时GPU利用率低
- p6: 最优切换点: 求解 dT/dB = 0
- p7: TAOCP批判: 连续近似 vs 离散batch的gap

---

### Claude #19
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M019 | wb-18-flashrnn.html | (需clone) | github.com/NX-AI/flashrnn, ICLR #2 | FlashRNN并行训练 |

**板书内容要求：**
- p1: RNN: h_t = f(h_{t-1}, x_t), 顺序依赖无法并行
- p2: 关键洞察: 如果f是线性的 → 前缀和(parallel scan)
- p3: 非线性RNN: 将f局部线性化 → 分段parallel scan
- p4: tiling: 块内顺序计算, 块间并行归约
- p5: IO分析: 类似FlashAttention, 块内SRAM驻留
- p6: 与Mamba/S4的关系: 结构化状态空间 → 隐式并行
- p7: TAOCP批判: 局部线性化的近似误差

---

### Claude #20
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M020 | wb-19-cub-reduce.html | cccl | cub/cub/device/device_reduce.cuh | CUB DeviceReduce原语 |

**板书内容要求：**
- p1: GPU归约: N元素 → 1个结果
- p2: warp内归约: __shfl_down_sync, log₂(32)=5步
- p3: block内归约: shared memory, log₂(blockDim)步
- p4: grid级归约: 两轮kernel — 第一轮每block一个结果, 第二轮归约
- p5: 工作量: O(N), 步骤: O(logN), 并行度: O(N/logN)
- p6: 实际瓶颈: 全局内存带宽, 非计算
- p7: TAOCP批判: GPU归约的cache-oblivious分析

---

### Claude #21
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M021 | wb-20-blelloch-scan.html | cccl | thrust/ | Blelloch Parallel Scan |

**板书内容要求：**
- p1: 前缀和: y_i = Σ_{j≤i} x_j
- p2: 朴素: O(N) 顺序, 无法并行
- p3: Blelloch up-sweep: 二叉树从叶到根 → O(N)工作, O(logN)步
- p4: Blelloch down-sweep: 根到叶分发 → O(N)工作, O(logN)步
- p5: 总: O(N)工作, O(logN)步, work-efficient
- p6: 应用: cumsum, sort(radix), stream compaction
- p7: TAOCP批判: Blelloch vs Hillis-Steele的trade-off

---

### Claude #22
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M022 | wb-21-roofline-model.html | (基于架构手册) | A100/H100 spec | Roofline模型完整推导 |

**板书内容要求：**
- p1: 算力-带宽平面: Y轴=FLOPS, X轴=算术强度
- p2: A100: 312T/2.0T=156, H100: 990T/3.35T=295
- p3: 三种kernel的定位: GEMM(计算密集), Softmax(带宽), Elementwise(带宽)
- p4: FlashAttention将Softmax从带宽受限→计算受限
- p5: mixed precision的影响: FP8算力=FP16×2 → 平衡点×2
- p6: 多级roofline: L1/L2/HBM各有自己的屋顶线
- p7: TAOCP批判: Roofline忽略的因素(bank conflict, warp divergence)

---

### Claude #23
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M023 | wb-22-dp-to-gpu-kernel.html | (基于论文) | ICLR #17 | 动态规划→GPU Kernel并行化 |

**板书内容要求：**
- p1: DP递推: dp[i] = f(dp[i-1], x_i), 顺序依赖
- p2: 如果f满足结合律 → parallel scan变换
- p3: 矩阵形式: [dp[i], 1]^T = M_i · [dp[i-1], 1]^T
- p4: 前缀矩阵乘: 可用Blelloch scan并行化
- p5: 例: Viterbi解码 → 每步max+add → 半环parallel scan
- p6: 例: CTC loss → 每步logadd → log-semiring scan
- p7: TAOCP批判: 不满足结合律的DP如何处理

---

### Claude #24
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M024 | wb-23-rlhf-pipeline.html | (基于论文) | ICLR OPPO | RLHF Pipeline Overlap |

**板书内容要求：**
- p1: RLHF四模型: Actor, Critic, Reference, Reward
- p2: PPO一步: generate→score→advantage→update
- p3: 朴素: 串行 → GPU利用率低(大量空闲)
- p4: OPPO: pipeline actor生成和critic评估
- p5: 时间线: generate batch k 的同时 score batch k-1
- p6: 加速比 = 1 + min(T_gen, T_score) / max(T_gen, T_score)
- p7: TAOCP批判: pipeline引入的on-policy偏差

---

### Claude #25
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M025 | wb-24-dash-deterministic.html | (基于论文) | ICLR #1 DASH | 确定性Attention调度 |

**板书内容要求：**
- p1: 问题: 不同GPU计算顺序不同 → reduce顺序不同 → 结果不同
- p2: 浮点非结合性: |((a+b)+c) - (a+(b+c))| ≤ ε·(|a|+|b|+|c|)
- p3: DASH: 固定所有GPU的计算和通信调度
- p4: 调度约束: 同一token在所有GPU上的softmax计算顺序一致
- p5: 代价: 灵活性降低, 可能非最优调度
- p6: 可复现训练: 同seed+同schedule → bit-exact相同loss曲线
- p7: TAOCP批判: 确定性 vs 性能的Pareto前沿

---

### Claude #26
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M026 | wb-25-moss-fp8.html | TransformerEngine | TE FP8 + ICLR #7 MOSS | MOSS: microscaling FP8训练 |

**板书内容要求：**
- p1: MXFP8: 每组k个元素共享1个scale (k=32或128)
- p2: group scale = max(|x_i|, i∈group) → 量化
- p3: vs per-tensor scaling: 更细粒度 → 更小量化误差
- p4: 误差界: ε_group ≤ ε_tensor / √k (统计平均)
- p5: 自动scaling: 运行时动态选择k
- p6: 硬件支持: Hopper MXFP8 native, Trainium2 MXFP8 native
- p7: TAOCP批判: group size k的计算/精度trade-off最优解

---

### Claude #27
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M027 | wb-26-metis-fp4.html | (基于论文) | ICLR #4 Metis | FP4量化训练 |

**板书内容要求：**
- p1: E2M1: 仅16个可表示值, max=6
- p2: 量化误差: ε ≤ amax/4 → 相对误差~25%
- p3: Metis方法: 梯度用FP4, 参数用FP8, master weight用FP32
- p4: 梯度压缩: 通信量降4×, 但噪声增大
- p5: 补偿: error feedback — 累积量化误差到下一步
- p6: 收敛: O(1/√T + σ_q/√T) 其中σ_q是量化noise
- p7: TAOCP批判: FP4的信息论下界

---

### Claude #28
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M028 | wb-27-grad-overlap.html | Megatron-LM | megatron/core/distributed/param_and_grad_buffer.py | 梯度通信Overlap |

**板书内容要求：**
- p1: 朴素: 全部backward完成 → AllReduce全部梯度
- p2: Bucketing: 参数分组, 每组backward完成就立即通信
- p3: bucket_size: max(40M, 1M × dp_size) — 自动调整
- p4: overlap: bucket_i通信 与 bucket_{i+1}计算 并行
- p5: 时间线: T_total = T_compute + T_comm_exposed (仅最后一个bucket)
- p6: 最优bucket数: 太多→延迟开销, 太少→overlap不充分
- p7: TAOCP批判: bucket_size公式对万卡dp_size=1024的安全性

---

### Claude #29
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M029 | wb-28-checkpoint-async.html | (基于MegaScale) | MegaScale论文 | 异步Checkpoint数学 |

**板书内容要求：**
- p1: 同步ckpt: 训练暂停 → 写入 → 恢复, 浪费T_ckpt
- p2: 异步: 后台线程写入, 训练继续
- p3: 一致性: snapshot必须是某一步的完整状态
- p4: Copy-on-Write: 快照参数到CPU pinned memory → 后台写NVMe
- p5: 时间: T_snapshot = Φ×2/BW_PCIe ≈ 350GB/32GB·s ≈ 11s
- p6: Young公式最优间隔: T* = √(2·T_snapshot·MTBF)
- p7: TAOCP批判: 异步ckpt期间参数被修改的一致性保证

---

### Claude #30
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M030 | wb-29-elastic-training.html | DeepSpeed | deepspeed/elasticity/ | 弹性训练数学 |

**板书内容要求：**
- p1: GPU数变化: W₁ → W₂, 训练不中断
- p2: LR调整: lr₂ = lr₁ × √(B₂/B₁), B∝W (linear scaling)
- p3: DP rebalance: 重新分配数据, AllReduce group变化
- p4: PP rebalance: 重新切分层, 需要checkpoint+resume
- p5: 最小扰动: 只调DP维度 → 无需重新切层
- p6: 收敛影响: lr跳变 → loss spike, 需warmup
- p7: TAOCP批判: 弹性伸缩的最小中断时间下界

---

### Claude #31
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M031 | wb-30-memory-barrier.html | (基于论文) | ICLR #6 Out of Memory Barrier | 百万token上下文内存优化 |

**板书内容要求：**
- p1: 长序列问题: 激活∝N², N=1M → 内存爆炸
- p2: Ring Attention: 将KV切分到多GPU, 每GPU只存N/p
- p3: 通信: KV block在GPU间轮转, 与计算overlap
- p4: 内存: O(N²/p) → 可行
- p5: 与SP/CP的组合: Ring Attention处理attention, SP处理非attention
- p6: 极限: p=1024, N=1M → 每GPU N/p=1K token的KV → 0.5MB/layer
- p7: TAOCP批判: Ring Attention的通信瓶颈分析

---

### Claude #32
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M032 | wb-31-three-forward.html | (基于论文) | ICLR #11 | 三次前向一次反向 |

**板书内容要求：**
- p1: 标准: 1F1B, 存激活 → 内存大
- p2: Recompute: 不存激活, backward时重算 → 2F1B
- p3: 新方法: 3F1B — 第三次前向用来计算精确梯度
- p4: 内存: 只存参数+优化器, 不存任何激活
- p5: 计算代价: 3× forward + 1× backward = 5× vs 标准3×
- p6: 内存节省: 从O(bshL)激活 → O(1)
- p7: TAOCP批判: 5/3×的计算开销 vs 内存节省的Pareto分析

---

### Claude #33
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M033 | wb-32-cuda-to-nki.html | TransformerEngine | TE + ICLR #9 | CUDA→NKI kernel迁移 |

**板书内容要求：**
- p1: CUDA抽象层: thread→warp→block→grid
- p2: NKI抽象层: tensor_engine→partition→tile
- p3: 关键差异: CUDA是SIMT(线程级), NKI是SIMD(tile级)
- p4: GEMM映射: CUDA用wmma/mma, NKI用tensor_engine
- p5: 内存映射: CUDA SMEM↔NKI SBUF, CUDA L2↔NKI PSUM
- p6: 推理图迁移: 找到CUDA pattern → 映射到NKI等价
- p7: TAOCP批判: 抽象层差异导致的性能不可移植性

---

### Claude #34
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M034 | wb-33-cross-hw-test.html | (基于md §3.3) | md §3.3批判4 | 跨硬件数值一致性测试 |

**板书内容要求：**
- p1: 三种硬件: NVIDIA GPU, Trainium2, TPU
- p2: 浮点差异来源: rounding mode, reduction order, fma availability
- p3: 度量: ||W_A - W_B||₂ / ||W||₂ (相对L2距离)
- p4: 可接受阈值: < 10^{-4} for BF16, < 10^{-2} for FP8
- p5: checkpoint迁移: A硬件训练 → B硬件resume → loss不应spike
- p6: A/B测试陷阱: baseline在A, experiment在B → 虚假显著性
- p7: TAOCP批判: 数值等价性测试的完备性不可能性

---

### Claude #35
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M035 | wb-34-e2e-benchmark.html | (基于MegaScale) | MegaScale + ICLR | 端到端训练Benchmark |

**板书内容要求：**
- p1: 指标: MFU, samples/sec, time-to-convergence
- p2: MFU的局限: 只衡量硬件利用率, 不衡量算法质量
- p3: 通信/计算比: r = T_comm / T_compute, 随p增加
- p4: 强scaling效率: E(p) = T(1)/(p·T(p))
- p5: 弱scaling效率: E(p) = T(1)/T(p) (固定每GPU工作量)
- p6: Amdahl定律: 串行比例s → speedup ≤ 1/s
- p7: TAOCP批判: benchmark的可比性(不同硬件、不同模型)

---

### Claude #36
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M036 | wb-35-fault-injection.html | (基于md §3.3) | md §3.3 | 故障注入测试 |

**板书内容要求：**
- p1: 故障类型: bit-flip, stuck-at, link-down, node-crash
- p2: silent data corruption: 概率 ~10^{-15}/bit/hr
- p3: 检测: ECC correctable → log + continue; uncorrectable → halt
- p4: 注入方法: 随机flip梯度中的1bit → 观察loss影响
- p5: 敏感度分析: 高层梯度flip比低层影响大~100×
- p6: 防护: gradient clipping + NaN检测 + loss spike rollback
- p7: TAOCP批判: bit-flip概率模型在HBM中的适用性

---

### Claude #37
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M037 | wb-36-anthropic-stack.html | (基于md §3.4) | md全文 | Anthropic垂直整合栈分析 |

**板书内容要求：**
- p1: 6层栈: L0硬件→L1 ISA→L2编译器→L3训练框架→L4算法→L5 Post-training
- p2: vs NVIDIA栈: CUDA→cuBLAS→NCCL→Megatron→...
- p3: 优势: 全栈协同设计, 无中间层损耗
- p4: 风险: 每层都是single-point-of-failure
- p5: ISA暴露的双刃剑: 性能↑ 但硬件迭代时需重写
- p6: 50万→100万芯片集群的工程挑战
- p7: TAOCP批判: 垂直整合 vs 生态系统 的历史规律

---

### Claude #38
| M | 文件 | 需clone | 需深读 | 内容 |
|-|-|-|-|-|
| M038 | wb-37-knuth-final.html | (综合) | 所有仓库+论文+md | TAOCP总结: 从算法到万卡 |

**板书内容要求：**
- p1: Knuth五性质: 贯穿从softmax到AllReduce到Pipeline
- p2: §1.3.1 "运行时间取决于实现": FA exp2f优化(FLOPs同,指令↓)
- p3: IO复杂度才是真正瓶颈: FA2 Θ(N²d²/M) 是IO最优
- p4: 常数因子是胜负手: MFU 55% vs 40%, 等效多25%硬件
- p5: 数值精度不是可选项: FP8 training failure modes
- p6: 规模改变一切: 100GPU的优化在10000GPU上可能失效
- p7: 最终结论: 先正确, 再最优, 然后才是快

---

## 每位Claude的工作流程

```
1. git clone 指定仓库 (--depth 1)
2. tree -L 2 -d 查看架构
3. git branch -a 查看分支
4. 按"需深读"列读取核心代码
5. 提取数值和公式
6. 写白板HTML (零源码引用, 纯数学)
7. 验证: grep确认无代码泄露
8. git commit到 wb-iclr2026-infra/
9. diff对比
10. TAOCP双角度批判 (用户+系统)
```

---

## 当前进度

```
Claude #1:  M001 ✅ (原项目 whiteboard_lecture/)
Claude #2:  M002 ✅ wb-01-kernel-math.html (9页, 334行, 20KB)
Claude #3:  M003 ⏳ wb-02-fa-backward.html
Claude #4:  M004 ⏳ wb-03-fp8-training-fail.html
...
Claude #38: M038 ⏳ wb-37-knuth-final.html
```
