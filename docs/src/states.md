# 态与正则形式

MPSSimulator 的量子态为矩阵乘积态（MPS）。纯态 [`MPS`](@ref) 局域物理维为 2，
混合态 [`DensityOperatorMPS`](@ref) 局域物理维为 4；二者继承同一抽象类型
[`AbstractMPS`](@ref)，共用数据布局与绝大多数操作。

## 数据布局（混合正则形式）

纯态与混合态都是 `FiniteMPSAlgorithms.CanonicalMPS` 的**薄封装**（字段
`core`）：存储、键谱与正交化复用 FiniteMPSAlgorithms，封装层额外保持
`scaling == 1`（范数始终存放在站点数据中）。底层布局为：

* **站点张量**：`N` 个 `A::Array{T,3}`，轴序 = (左键, 物理维, 右键)，
  开放边界（`size(ψ[1],1) == 1`、`size(ψ[end],3) == 1`）。
  **量子态就是张量网络的普通收缩**——网络中不含权重；
* **键谱 `s`**：长度 `N+1` 的 Schmidt 谱数组，`s[b+1]` 为第 `b` 个键
  （站点 `b` 与 `b+1` 之间）的谱；`missing` 表示正则信息未知。

当所有站点张量右正则（行正交）且键谱正确时，态处于**混合正则形式**
（`iscanonical == true`）：站点张量右正则，键谱作为 Schmidt 系数存放，
前缀约化密度矩阵 = `diag(Λ²)`。幺正演化严格保持该形式（测试中逐步验证）；
非幺正算子（投影 / Kraus）可能破坏正则性——状态始终精确，
需要时用 `canonicalize!` 恢复。

`DensityOperatorMPS` 的局域张量携带 ket/bra 融合指标 `p = i + 2j`
（`i` 为行指标、`j` 为列指标），`N` 体门的局域超算符作用在该融合指标上。

## 初始化

```@docs
zero_state_mps
onehot_encoding_mps
qubit_encoding_mps
amplitude_encoding_mps
rand_state_mps
densitymatrix_mps
statevector_mps
```

纯态 → 混合态的精确转换：

```@docs
densitymatrix(psi::MPS; trunc=DefaultMPSTruncation)
```

## 查询与规范型

```@docs
phydim
phydims
bonddim
bonddims
svectors_uninitialized
isrightcanonical
iscanonical
canonicalize!(psi::AbstractMPS; trunc=NoTruncation(), normalize=true)
normalize!(psi::AbstractMPS)
normalize_tr!(rho::DensityOperatorMPS)
```

## 类型

```@docs
AbstractMPS
MPS
DensityOperatorMPS
```
