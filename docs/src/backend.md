# 线路桥接层与后端

`src/quantumcircuits` 桥接层把 QuantumCircuits IR 指令映射到核心原语：

| IR 指令 | 核心操作 |
|---|---|
| `GateOp`（含 `CtrlGate` / `InvGate` / `PowGate` 修饰符） | `mat(op)` 重排后 `apply!`（非相邻自动 SWAP 聚集） |
| `ChannelOp`（`KrausOp` / `Depolarizing` / …） | `apply_kraus!`（纯态自动提升） |
| `MeasOp` | `measure!` 坍缩 + 经典寄存器写回 |
| `ReinitOp` | `reset_qubit_zero!` |
| `IfOp` | 经典条件分支（需 `simulate`） |
| `BlockOp` | `unroll` 展开逐条执行 |
| `BarrierOp` | 无操作 |

## simulate / simulate!

```@docs
simulate(c::Circuit, s::Union{MPS,DensityOperatorMPS})
simulate!(c::Circuit, s::Union{MPS,DensityOperatorMPS})
```

* `params`：`Vector{<:Real}` 按 `parameters(c)` 顺序绑定；`Dict` 参数表
  （键可为 `Param` / `Symbol` / `ParamVector`）；同名参数 = 权重共享；
* 测量结果写回经典寄存器（`ClassicalStore`），供 `IfOp` 条件分支使用；
* `trunc`：每个门作用时的截断方案。

```@example
using QuantumCircuits, MPSSimulator

c = Circuit(2)
push!(c, RX(:θ, 1))
push!(c, RY(:θ, 2))
push!(c, CX(1, 2))

ψ = simulate(c, zero_state_mps(2); params = [0.4])       # 位置绑定
ψ = simulate(c, zero_state_mps(2); params = Dict(:θ => 0.4))
c2 = QuantumCircuits.assign(copy(c), Dict(:θ => 0.4))    # 或先绑参再演化
println("consistent = ", statevector(ψ) ≈ statevector(simulate(c2, zero_state_mps(2))))
```

## MatrixProductBackend

```@docs
MatrixProductBackend
```

实现 QuantumCircuits.Interface 的后端协议（能力：`:statevector`、
`:mid_measure`、`:noise`）。`shots > 0` 时逐 shot 采样，测量结果按
`counts_key` 约定聚合到 `SimResult.counts`（键只覆盖真正被测量的寄存器）。
## 经典寄存器运行时

```@docs
ClassicalStore
```

## 线路构造

变分线路构造器 `variational_circuit_1d` / `real_variational_circuit_1d`
由 QuantumCircuits 提供（支持符号参数与数值参数两种模式）：

```julia
using QuantumCircuits

c = variational_circuit_1d(4, 2; θs = rand(36) .* 2π)   # 数值参数
c = variational_circuit_1d(4, 2)                        # 符号参数 θ[1]…θ[36]
```

其余复合线路（如 QFT、量子相位估计）可由基本门与 `ctrl` / `pow` /
`dagger` 修饰符按需组合——参见 QuantumCircuits 的线路构造文档。

## 与 VQC 的对应关系

两后端共享 IR 与接口协议，可互换验证：

| 任务 | MPSSimulator（MPS） | VQC（稠密） |
|---|---|---|
| 纯态模拟 | `simulate(c, zero_state_mps(n))` | `simulate(c, zero_state(n))` |
| 混合态模拟 | `simulate(c, densitymatrix_mps(n))` | `simulate(c, DensityMatrix(...))` |
| 采样后端 | `MatrixProductBackend()` | `StateVectorBackend()` |

适用规模：态矢量后端适合 `n ≲ 25`；MPS 后端适合低纠缠深度线路的
更大 `n`，纠缠增长时经 `trunc` 以受控精度压缩键维。
