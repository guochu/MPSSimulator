# 正则形式保持验证（canonical form check）

对应脚本：[`debug/canonical_form.jl`](canonical_form.jl)

## 目的

验证 MPSSimulator 的核心承诺：**酉门演化全程保持混合正则形式**（站点张量
右正则 + `svectors` 键谱一致），且模拟结果与稠密后端 VQC 逐步一致。同时
验证非幺正信道对正则性的破坏及 `canonicalize!` 的恢复能力。

## 方法

1. **随机线路**：以固定种子（`MersenneTwister(42)`）生成 5 比特、25 指令的
   线路——每层含随机单比特门（`H` / `RX`）与随机两比特门（`CZ` / `RZZ`），
   两比特位置允许**非相邻**（触发 SWAP 聚集 → Hastings 截断 SVD 路径）。
2. **纯态 MPS**：`zero_state_mps(5)` 出发，逐门 `apply!`，每步后检查
   `iscanonical`。
3. **混合态 DMPS**：`densitymatrix_mps(5)` 出发，对同一酉线路重复上述检查，
   并验证 `tr ρ = 1` 保持。
4. **非幺正信道**：在线路中段插入振幅阻尼信道（`KrausOp`，γ = 0.3），
   检查正则性破坏与 `canonicalize!` 恢复。
5. **稠密对照**：终态与 VQC（`StateVector` / `DensityMatrix`）逐步对照。

## `iscanonical` 的判定语义

状态处于混合正则形式当且仅当：

* 站点 `2…L` 的张量右正则（行正交）；
* `svectors` 已初始化，且前缀约化密度矩阵 = `diag(Λ²)`（逐键一致）。

站点 1 **允许携带整体标度**：`canonicalize!(; normalize=false)`（DMPS 默认，
保持迹）的输出中站点 1 携带 HS 范数权重 `sqrt(tr ρ²)`——对迹为 1 的混合态
该值 ≠ 1，与"迹归一"不能同时成立，故此处的正则性检查不对站点 1 施加
单位范数约束。

## 结果

最近一次运行（seed `20260909`，n = 5，25 指令）：

| 检查项 | 结果 |
|---|---|
| 纯态 MPS：25 步逐门 `iscanonical` | 全部 `true`（键维 1 → 4） |
| 纯态终态 vs VQC | maxdiff = 3.6e-16 |
| 混合态 DMPS：25 步逐门 `iscanonical` | 全部 `true`（键维 1 → 16） |
| 混合态 `tr ρ` | 1（偏差 < 3e-16） |
| 混合态终态 vs VQC | maxdiff = 5.0e-16 |
| Kraus 信道后 `iscanonical` | `false`（预期：非幺正破坏正则性） |
| `canonicalize!` 后 `iscanonical` | `true`（迹保持 1，状态精确） |
| 带信道终态 vs VQC | maxdiff = 2.5e-16 |

## 结论

* 酉门（含非相邻 SWAP 聚集 + Hastings 更新）在纯态与混合态上均**严格保持
  混合正则形式**，键谱同步更新；
* 非幺正 Kraus 信道破坏正则性但**状态保持精确**（迹不变），
  `canonicalize!` 可完整恢复正则形式；
* MPS 后端与 VQC 稠密后端在全部场景下机器精度一致。

## 运行方式

```sh
JULIA_PROBE_LIBSTDCXX=0 JULIA_DEPOT_PATH=/tmp/jdepot: julia --startup-file=no \
  debug/canonical_form.jl
```

（脚本通过 `@__DIR__` 自动定位同级的 `QuantumCircuits` / `VQC` / 本包。）
