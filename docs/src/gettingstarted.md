# 快速开始

本节给出与 QuantumCircuits 协同做量子线路模拟的基本流程：
构造态 → 构造线路 → 演化 → 测量。各核心原语的系统列表见
[态](states.md)、[核心操作](coreops.md)、[可观测量与测量](observables.md)。

## 比特与索引约定

MPSSimulator 与 QuantumCircuits / VQC 共用同一套索引约定：

* **比特编号**：1-based、**小端序**——qubit 1 是最低有效位（LSB），
  qubit `i` 的权重为 `2^(i-1)`；MPS 站点 `k` ↔ qubit `k-1`。
* **振幅下标**：计算基矢 `|qₙ…q₂q₁⟩` 的振幅位于
  `statevector(ψ)[1 + q₁ + 2·q₂ + … + 2^(n-1)·qₙ]`。
* **门矩阵的位序**：`positions` / `qubits` 列表中**第一个比特是矩阵的
  最高位**（MSB-first）：`apply!(ψ, M, (2, 1))` 中 `M` 的行列索引按
  `idx = 2·q₂ + q₁` 解释。
* **混合态融合指标**：`DensityOperatorMPS` 每个站点携带局域维 4 的
  融合指标 `p = i + 2j`（`i` 为 ket 行指标、`j` 为 bra 列指标）。

## 安装

本仓库为本地开发布局（与 `QuantumCircuits` 平级放置）：

```julia
# 方式一：开发环境经 LOAD_PATH 解析（顺序很重要，QuantumCircuits 在前）
push!(LOAD_PATH, ".../QuantumCircuits")
push!(LOAD_PATH, ".../MPSSimulator")

# 方式二：Pkg 开发挂接
using Pkg
Pkg.develop(path = ".../QuantumCircuits")
Pkg.develop(path = ".../MPSSimulator")
```

注意：`QuantumCircuits` 与 `MPSSimulator` 都导出了 `expectation`，
同时 `using` 两个包时请用 `import MPSSimulator: expectation` 显式消歧。

## 基本流程

```@example
using QuantumCircuits, MPSSimulator
import MPSSimulator: expectation
using LinearAlgebra
using QuantumCircuits.Hamiltonian

# 1) 构造态
ψ = zero_state_mps(3)                       # |0⟩⊗³

# 2) 构造线路（支持符号参数 :θ）
c = Circuit(3)
push!(c, RY(:θ, 1))
push!(c, CX(1, 2))
push!(c, CZ(2, 3))                          # 非相邻门自动 SWAP 聚集

# 3) 演化（params 绑定符号参数；trunc 控制逐键截断）
ψ = simulate(c, ψ; params = [0.4])

# 4) 测量与可观测量
println("probabilities = ", round.(probabilities(ψ); digits = 4))
println("⟨Z₁Z₂⟩ = ", expectation(PauliTerm(1.0, 1 => :Z, 2 => :Z), ψ))

# 5) 纠缠：约化密度矩阵与 von Neumann 熵
rdm = partial_tr(ψ, [2, 3])
println("entanglement entropy = ", -renyi_entropy(real.(diag(rdm))))
```

## 采样后端

`MatrixProductBackend` 实现了 QuantumCircuits.Interface 的后端协议，
支持强模拟、逐 shot 采样与经典反馈：

```@example
using QuantumCircuits, MPSSimulator

b = MatrixProductBackend()
c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))
push!(c, measure(1, c.cregs[1][1]))
push!(c, measure(2, c.cregs[1][2]))

rs = simulate(c, b; shots = 512, seed = 7)
println(rs.counts)
```
