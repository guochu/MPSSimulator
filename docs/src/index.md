# MPSSimulator.jl

MPSSimulator 是 [QuantumCircuits](https://github.com/) IR 的**矩阵乘积态（MPS）模拟后端**：
把 `Circuit` 中的指令（`GateOp` / `ChannelOp` / `MeasOp` / `ReinitOp` / `IfOp` / …）
高效地作用到矩阵乘积态上，并提供测量、偏迹、期望值等原语。纯态使用局域维 2 的
`MPS`，混合态使用局域维 4 的 `DensityOperatorMPS`；与基于态矢量的稠密后端 VQC
共享同一线路 IR 与接口协议（`simulate` / `nqubits` / `measure`），可互换使用并
交叉验证。

约定（与 QuantumCircuits 一致）：比特索引 **1-based**、小端序（qubit 1 = 最低
有效位）；门矩阵 `positions` 的第一个比特为矩阵最高位。

## 一个简单的例子：Bell 态

```@example
using QuantumCircuits, MPSSimulator

c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))

ψ = simulate(c, zero_state_mps(2))
println("probabilities = ", probabilities(ψ))

outcome = measure!(ψ, 1)
println("outcome = $outcome")
```

## 长程门与截断

非相邻比特的多体门自动经相邻 SWAP 聚集后作用再散开；所有作用型接口均接受
`trunc` 关键字做逐键截断 SVD（默认 `truncdim(50)`）：

```@example
using QuantumCircuits, MPSSimulator

c = Circuit(4)
push!(c, CZ(1, 4))                 # 非相邻两比特门
ψ = simulate(c, zero_state_mps(4); trunc = truncdim(16))
println("bond dimensions = ", bonddims(ψ))
```

## 噪声与混合态

噪声信道直接进线路 IR（`KrausOp` / `Depolarizing` / `AmplitudeDamping`）；
纯态遇信道自动提升为 `DensityOperatorMPS`：

```@example
using QuantumCircuits, MPSSimulator, LinearAlgebra

c = Circuit(2)
push!(c, H(1))
push!(c, AmplitudeDamping(1, 0.3))

ρ = simulate(c, zero_state_mps(2))      # → DensityOperatorMPS
println("trace = ", tr(ρ))
```

## 目录

```@contents
Depth = 2
```
