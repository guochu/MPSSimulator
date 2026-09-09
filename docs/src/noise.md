# 混合态与噪声信道

混合态由 `DensityOperatorMPS` 表示：局域维 4 的 MPS，携带密度矩阵的
ket/bra 融合指标。酉门在融合指标上以局域超算符作用；Kraus 信道先
组合为单个局域超算符 `M = Σₖ conj(Kₖ) ⊗ Kₖ`（行/列按逐站点
(i,j) 交织）再作用。

## 噪声即指令

噪声信道由 QuantumCircuits IR 直接描述（信道构造器来自
`QuantumCircuits`）：`KrausChannel`（通用 CPTP）、`Depolarizing`、
`AmplitudeDamping` 等，用 `KrausOp(ch, qubits)` 定位到比特后
`push!` 进线路即可：

```@example
using QuantumCircuits, MPSSimulator
using LinearAlgebra

# 显式 Kraus 算子集（须满足 Σ K†K = I）
γ = 0.3
ch = KrausChannel([[1 0; 0 sqrt(1 - γ)], [0 sqrt(γ); 0 0]])

c = Circuit(3)
push!(c, H(1))
push!(c, CX(1, 2))
push!(c, QuantumCircuits.KrausOp(ch, 1)) # 振幅阻尼 qubit 1
push!(c, Depolarizing([2], 0.05))        # 去极化 qubit 2

ρ = simulate(c, zero_state_mps(3))       # 纯态遇信道自动提升
println("trace = ", tr(ρ))
println("purity = ", real(dot(ρ, ρ)))
```

混合态初态可直接用 `densitymatrix_mps(n)`（`|0…0⟩⟨0…0|`）或
`densitymatrix(psi)`（纯态精确转换）。

## 正则形式的保持

* 酉门作用严格保持混合正则形式（站点右正则 + 键谱同步更新）；
* Kraus / 投影等非幺正算子可能破坏正则性——**状态始终精确**，
  需要时 `canonicalize!(ρ)` 恢复（默认不归一，保持迹不变）；
* 迹归一用 `normalize_tr!`；`LinearAlgebra.normalize!` 归一到
  融合指标空间的 2-范数（= `sqrt(tr ρ²)`），与迹归一不同。

## 混合态专属操作

```@docs
LinearAlgebra.tr(rho::DensityOperatorMPS)
LinearAlgebra.dot(rho::DensityOperatorMPS, sigma::DensityOperatorMPS)
```

`tr(ρ)` 为 δ 插入收缩；`dot(ρ, σ)` 为 Hilbert-Schmidt 内积
`tr(ρσ)`（直接双网络收缩，不转稠密）。迹归一 `normalize_tr!` 与
2-范数归一的区别见[态](states.md)；信道作用的底层原语 `apply_kraus!`
见[核心操作](coreops.md)。
