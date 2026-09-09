# 核心操作

核心层原语不依赖任何 IR 类型：直接把局域矩阵 / Kraus 算子集作用到
MPS 态上。**所有作用型接口均接受 `trunc` 关键字**（`TruncationScheme`），
门作用为 Hastings 式更新（相邻块截断 SVD 扫描）；非相邻比特的
多体算子自动经相邻 SWAP 聚集后作用再散开。

## 局域矩阵作用

```@docs
apply!
apply
```

约定：`positions[1]` 为矩阵最高位（MSB-first）；矩阵维度须匹配
`N` 个比特的局域维（纯态 `d = 2`、混合态 `d = 4`）。
若矩阵是 `d²×d²` 的局域超算符（如 `KrausOp` 的展开），纯态会自动
提升为 `DensityOperatorMPS`（返回值类型改变）。

```@example
using QuantumCircuits, MPSSimulator

ψ = zero_state_mps(2)
apply!(ψ, [0 1; 1 0], (1,))            # X 作用在 qubit 1
println("probabilities = ", probabilities(ψ))
```

## Kraus 算子集

```@docs
apply_kraus!
```

`ks` 须满足完全正迹条件 `Σₖ Kₖ†Kₖ = I`。纯态输入自动提升为
`DensityOperatorMPS`。非幺正演化可能破坏正则形式（状态精确，
可随时 `canonicalize!` 恢复）。

## 相干重置

```@docs
reset_qubit_zero!
```

## 截断方案

默认截断为 `DefaultMPSTruncation = truncdim(50)`（集中定义于
`src/defaults.jl`）。可选方案：

```@docs
NoTruncation
truncdim
trunccutoff
truncdimcutoff
DefaultMPSTruncation
```

例：压缩键维到 16、或按相对误差 `1e-8` 截断——

```julia
apply!(ψ, M, (1, 2); trunc = truncdim(16))
apply!(ψ, M, (1, 2); trunc = trunccutoff(ϵ = 1e-8))
```
