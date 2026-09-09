# 可观测量与测量

所有可观测量均**不构造满算符矩阵**：局域算符链直接在转移（MPO-less）
收缩中插入算符；`GateOp` 经重排后作用到 ket 副本（纯态）或仅作用 ket 侧
的超算符（混合态）；Pauli / 自旋代数逐项化为局域算符链。

支持的可观测量类型：

* **局域算符链**：`Vector{Pair{Int,<:AbstractMatrix}}`（qubit → 2×2 矩阵，
  任意顺序，缺失比特为恒等）；
* **Pauli 代数**：`PauliTerm` / `PauliSum`（`QuantumCircuits.Hamiltonian`）；
* **自旋算符**：`SpinOpTerm` / `SpinOpSum`（支持 `:I/:X/:Y/:Z/:P/:M` 与
  任意 2×2 矩阵因子，同位多因子按乘积序合并）；
* **`GateOp`**：酉门作为可观测量，`⟨ψ|U|ψ⟩` 或 `tr(ρ U)`。

注意：本包不提供稠密（满）矩阵算符的期望值接口；`QuantumCircuits` 的
`mat(t::SpinOpTerm, n)` 等稠密展开仅供小规模验证 / 对拍。

```@docs
MPSSimulator.expectation
fidelity
overlap
partial_tr
densitymatrix(rho::DensityOperatorMPS)
amplitude
statevector
```

## 测量与采样

```@docs
probabilities
marginal_probabilities
measure!
sample
post_select
post_select!
```

`measure!` **就地坍缩**并归一化（纯态保持 2-范数，混合态保持迹 1）；
`sample` 为非破坏性采样（按边缘概率分布，指数级于子集大小）。

## 纯态-混合态互操作

```@example
using QuantumCircuits, MPSSimulator
import MPSSimulator: expectation
using QuantumCircuits.Hamiltonian
using LinearAlgebra

ψ = simulate(Circuit([H(1), CX(1, 2)]), zero_state_mps(2))
ρ = densitymatrix(ψ)                        # 纯态 → 混合态

println("⟨Z₁⟩  = ", expectation(PauliTerm(1.0, 1 => :Z), ψ))
println("tr(ρZ₁) = ", expectation(PauliTerm(1.0, 1 => :Z), ρ))
println("fidelity = ", fidelity(ρ, ψ))     # ⟶ 1
```
