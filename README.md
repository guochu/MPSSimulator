# MPSSimulator.jl

基于**矩阵乘积态（MPS）**的量子线路模拟后端，是全栈量子计算软件的线路模拟环节之一。
作为 [`QuantumCircuits`](../QuantumCircuits) IR 的后端实现，与基于态矢量的稠密后端
[`VQC`](../VQC) 平行：二者共享同一线路 IR 与接口协议（`simulate` / `nqubits` /
`measure`），可互换使用并交叉验证。

> 状态：开发阶段。纯态使用局域维 2 的 MPS，混合态使用局域维 4 的
> DensityOperatorMPS；支持任意拓扑门作用（非相邻比特自动 SWAP 聚集）、
> 噪声信道、中途测量与经典反馈，以及强模拟 / 采样后端。

## 特性

- **纯态**：`MPS`（局域维 2），**混合态**：`DensityOperatorMPS`（局域维 4，
  ket/bra 融合指标 `p = i + 2j`）；二者继承同一抽象类型 `AbstractMPS`，
  共用数据布局与绝大多数操作
- **基于 `FiniteMPSAlgorithms.CanonicalMPS` 的薄封装**：站点张量保持右正则，
  键 Schmidt 谱记录于 `s`；正则化 / 相邻交换 / 两体门与张量操作层均复用
  [`FiniteMPSAlgorithms`](../FiniteMPSAlgorithms) 的现成实现；幺正演化全程
  保持正则形式
- **Hastings 式门更新**：相邻块截断 SVD 扫描，长程 / 多体门自动以相邻 SWAP
  聚集后作用再散开；**所有作用型接口接受 `trunc` 关键字**做逐键截断
- **噪声即指令**：`KrausOp` / `Depolarizing` / `AmplitudeDamping` 等信道
  直接进线路 IR，纯态遇信道自动提升为混合态
- **完整可观测量接口**：局域算符链、Pauli / 自旋算符代数、GateOp 期望、
  保真度、约化密度矩阵——均不构造满算符矩阵
- **QuantumCircuits.Interface 后端**：`MatrixProductBackend` 支持强模拟、
  逐 shot 采样、中途测量与经典条件反馈（`IfOp`）
- 基本张量操作（截断 SVD / 正交分解）与 TEMPO 的 `tensorops` 接口一致，
  底层由 [MatrixAlgebraKit](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl) 驱动

## 环境要求

- Julia ≥ 1.10
- 本地包依赖：[`QuantumCircuits`](../QuantumCircuits)（IR 与接口协议）、
  [`FiniteMPSAlgorithms`](../FiniteMPSAlgorithms)（CanonicalMPS 与张量操作）
- 测试额外依赖：[`VQC`](../VQC)（稠密后端，用于交叉验证）

## 安装

本仓库为本地开发布局（与 `QuantumCircuits` 平级放置），两种接入方式：

```julia
# 方式一：开发环境经 LOAD_PATH 解析（无 Manifest，最轻量）
# 顺序很重要：依赖包必须在 MPSSimulator 之前入栈
push!(LOAD_PATH, ".../QuantumCircuits")
push!(LOAD_PATH, ".../FiniteMPSAlgorithms")
push!(LOAD_PATH, ".../MPSSimulator")

# 方式二：Pkg 开发挂接
using Pkg
Pkg.develop(path = ".../QuantumCircuits")
Pkg.develop(path = ".../FiniteMPSAlgorithms")
Pkg.develop(path = ".../MPSSimulator")
```

## 快速上手

```julia
using QuantumCircuits      # IR + simulate 接口
using MPSSimulator
using LinearAlgebra        # tr / diag 等
import MPSSimulator: expectation   # 两包均导出 expectation，显式消歧

# ── 纯态线路模拟 ──────────────────────────────────────────────
c = Circuit(4)
push!(c, H(1))
push!(c, CX(1, 2))
push!(c, RY(0.5, 4))
psi = simulate(c, zero_state_mps(4))          # 非就地；simulate! 为就地
statevector(psi)                              # 稠密振幅（小系统验证用）

# ── 截断：所有作用型接口均有 trunc 关键字（默认 truncdim(50)）──
psi = simulate(c, zero_state_mps(4); trunc = truncdim(16))

# ── 混合态与噪声信道 ──────────────────────────────────────────
γ = 0.2
ch = KrausChannel([[1 0; 0 sqrt(1 - γ)], [0 sqrt(γ); 0 0]])
cc = Circuit(3)
push!(cc, H(1))
push!(cc, KrausOp(ch, 2))                     # 噪声直接进 IR
rho = simulate(cc, zero_state_mps(3))         # 纯态自动提升 → DensityOperatorMPS
tr(rho)                                       # 1.0

# ── 测量 / 采样 / 后选择 ──────────────────────────────────────
st = copy(psi)
outcome = measure!(st, 1)                     # 就地坍缩，返回 0/1
sample(psi, 1000)                             # Dict{Int,Int} 计数
post_select(psi, 1, 0)                        # => (坍缩态, 概率)

# ── 可观测量（不构造满矩阵）──────────────────────────────────
using QuantumCircuits.Hamiltonian
h = PauliSum([PauliTerm(1.0, 1 => :Z, 2 => :Z), PauliTerm(0.5, 2 => :X)])
expectation(h, psi)                           # ⟨ψ|h|ψ⟩
expectation(h, rho)                           # tr(ρ h)
expectation([1 => [0 1; 1 0], 3 => [1 0; 0 -1]], psi)   # 局域算符链
expectation(RZZ(0.7, 2, 3), psi)              # GateOp 可观测

# ── 纠缠 / 约化密度矩阵 ──────────────────────────────────────
rdm = partial_tr(psi, [2, 3, 4])              # 保留比特 1 的约化密度矩阵
λ² = real.(diag(rdm))                         # 归一化态下 = Schmidt 谱平方
-renyi_entropy(λ²)                            # von Neumann 熵（α = 1）

# ── 强模拟后端（shots 采样 + 经典反馈）────────────────────────
b = MatrixProductBackend()
cb = Circuit(2)
push!(cb, H(1))
push!(cb, CX(1, 2))
push!(cb, measure(1, cb.cregs[1][1]))
push!(cb, measure(2, cb.cregs[1][2]))
rs = simulate(cb, b; shots = 1024, seed = 7)
rs.counts                                     # Dict{String,Int}，键如 "c:00"
```

## 约定

与 `QuantumCircuits` / `VQC` 一致：

| 约定 | 说明 |
|---|---|
| 比特索引 | 1-based、小端序（qubit 1 = 最低有效位；站点 `k` ↔ qubit `k-1`） |
| 门矩阵 | `positions` 中**第一个比特为矩阵最高位**（MSB-first） |
| 站点张量 | 轴序 = (左键, 物理维, 右键)，开放边界 |
| 正则形式 | 站点张量右正则 + 键谱 `s`；幺正演化严格保持，投影 / Kraus 可能破坏（状态精确，`canonicalize!` 可恢复） |
| 混合态融合指标 | 局域维 4，`p = i + 2j`（`i` = ket 行指标为低位） |

## API 总览

**态与初始化**（`zero_state_mps` / `onehot_encoding_mps` / `qubit_encoding_mps` /
`amplitude_encoding_mps` / `rand_state_mps` / `densitymatrix_mps`）；
查询（`bonddim` / `bonddims` / `phydim` / `phydims` /
`iscanonical` / `isrightcanonical`）。

**核心操作**（均带 `trunc` 关键字）：
`apply!` / `apply`（局域矩阵 + 任意比特位置）、`apply_kraus!`（Kraus 算子集）、
`reset_qubit_zero!`（相干重置）。

**测量与采样**：`probabilities` / `marginal_probabilities` / `measure!` /
`sample` / `post_select` / `post_select!`。

**可观测量**：`expectation`（局域算符链 / `PauliTerm` / `PauliSum` /
`SpinOpTerm` / `SpinOpSum` / `GateOp`）、`fidelity`、`overlap`、
`partial_tr`、`amplitude`、`statevector`、`densitymatrix`（混合态 → 稠密矩阵，
小系统验证用）、`dot`（HS 内积）、`tr`。

**正则形式**：`canonicalize!`（QR + 截断 SVD 扫描）、`normalize!`（2-范数）、
`normalize_tr!`（迹归一，混合态专用）。

**QuantumCircuits 桥接层**：`simulate` / `simulate!`（线路演化，支持符号参数、
中途测量、`IfOp` 经典反馈、`BlockOp`）、`ClassicalStore`（经典寄存器运行时）、
`MatrixProductBackend`（Interface 后端）。

**张量操作层**（re-export 自 `FiniteMPSAlgorithms`，与 TEMPO `tensorops`
接口一致）：截断方案（`NoTruncation` / `truncdim` / `truncrelerr` /
`truncdimcutoff`）、正交分解（`leftorth!` / `rightorth!`，算法 `QR` / `QRpos` /
`LQ` / `LQpos` / `SVD` / `SDD` / `Polar`）、带截断 SVD（`tsvd!`）、`renyi_entropy`。

各导出项的完整签名与语义请查阅 docstring（REPL 中 `?函数名`）。

## 默认参数

集中定义于 [`src/defaults.jl`](src/defaults.jl)：所有作用型接口的 `trunc`
默认为 `DefaultMPSTruncation = truncdim(50)`。

## 测试

测试与 VQC 稠密后端逐项交叉验证（165 项，约 45 秒）：

```sh
cd MPSSimulator
JULIA_PROBE_LIBSTDCXX=0 JULIA_DEPOT_PATH=/tmp/jdepot: julia --startup-file=no -e '
  push!(LOAD_PATH, "../QuantumCircuits"); push!(LOAD_PATH, "../VQC");
  push!(LOAD_PATH, ".");
  include("test/runtests.jl")'
```

（`JULIA_PROBE_LIBSTDCXX=0` 用于跳过沙箱环境下的 libstdc++ 探测；`DEPOT_PATH`
尾随冒号追加默认 depot。）

## 许可证

[GPL-3.0](LICENSE)
