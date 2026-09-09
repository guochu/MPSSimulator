"""
    MPSSimulator

QuantumCircuits IR 的 **矩阵乘积态（MPS）模拟后端**（与态矢量后端 VQC 平行）。

分层（QuantumCircuits 桥接层与核心层在物理上分离；核心层 `src/states` /
`src/kernels` / `src/coreops` / `src/ops` / `src/expecs` 只依赖通用张量操作，
不引用任何 IR 类型）：

* **张量操作层**（`src/tensorops`）：截断方案、正交 / 奇异值分解，
  与 TEMPO 的 `tensorops` 接口一致，由 MatrixAlgebraKit 驱动；
* **态层**（`src/states`）：抽象基类 `AbstractMPS`、纯态 `MPS`（局域维 2）与
  混合态 `DensityOperatorMPS`（局域维 4，融合指标 `p = i + 2j`）——两者共用
  同一数据布局；存储采用混合正则形式——站点张量右正则，`svectors` 记录
  各键 Schmidt 谱；
* **核心层**（`src/kernels` / `src/coreops` / `src/ops` / `src/expecs` /
  `src/initializers`）：局域矩阵 / Kraus 原语（`apply!` / `apply_kraus!`，
  支持任意比特位置——非相邻位置自动经 SWAP 聚集）、测量 / 后选择 /
  偏迹、期望值、态初始化；**所有作用型接口均接受 `trunc` 关键字**
  （`TruncationScheme`，默认 `DefaultMPSTruncation`）；
* **QuantumCircuits 桥接层**（`src/quantumcircuits`）：把 IR 指令
  （`GateOp` / `ChannelOp` / `MeasOp` / `ReinitOp` / `BarrierOp` / `IfOp` /
  `BlockOp`）与 Pauli 代数桥接到核心原语，提供 `simulate` / 经典寄存器
  运行时 / `nqubits` 与 `measure` 协议扩展，以及
  Interface 后端 `MatrixProductBackend`
  （`QuantumCircuits.simulate(c, backend; shots, seed)`）。

约定（与 QuantumCircuits / VQC 一致）：

* 比特索引 **1-based**，小端序（qubit 1 = 最低有效位）；
* 门矩阵约定：`positions` 列表中**第一个比特为矩阵最高位**；
* 站点张量轴序 = (左键, 物理维, 右键)，开放边界；
* 幺正演化严格保持正则形式；非幺正算子（投影 / Kraus）可能破坏正则性
  （状态始终精确，需要时可用 `canonicalize!` 恢复）。

核心接口：

* `apply!` / `apply` / `apply_kraus!` / `reset_qubit_zero!`：局域算子作用；
* `simulate` / `simulate!`（`QuantumCircuits.Interface`）：演化整条线路；
* `expectation` / `probabilities` / `marginal_probabilities` / `measure!` /
  `sample` / `post_select` / `partial_tr` / `fidelity` / `amplitude` /
  `statevector`；
* `canonicalize!` / `normalize!` / `tr`。
"""
module MPSSimulator

using LinearAlgebra
using LinearAlgebra: BlasFloat
using Random
using TensorOperations
using MatrixAlgebraKit

using QuantumCircuits
using QuantumCircuits: GateOp, ChannelOp, MeasOp, ReinitOp, BarrierOp, IfOp, BlockOp,
                       Operation, Circuit, Cond, Param, ClbitRef, CReg, ParamVector,
                       qubits, mat, kraus, parameters, unroll
import QuantumCircuits: nqubits, measure
using QuantumCircuits.Interface: Backend

# ── 张量操作层 ───────────────────────────────────────────────────────────────
include("tensorops/tensorops.jl")

# ── 默认参数 ────────────────────────────────────────────────────────────────
include("defaults.jl")

# ── 态 ──────────────────────────────────────────────────────────────────────
include("states/abstractmps.jl")
include("states/mps.jl")
include("states/densityoperator.jl")

# ── 核心层 ──────────────────────────────────────────────────────────────────
include("kernels.jl")
include("coreops.jl")
include("expecs.jl")
include("ops/ops.jl")
include("initializers.jl")

# ── QuantumCircuits 桥接层 ──────────────────────────────────────────────────
include("quantumcircuits/quantumcircuits.jl")

# ── 导出 ────────────────────────────────────────────────────────────────────
export # 张量操作
    TruncationScheme, NoTruncation, TruncateDim, TruncateCutoff, TruncationDimCutoff,
    truncdim, trunccutoff, truncdimcutoff,
    QR, QRpos, LQ, LQpos, SVD, SDD, Polar,
    tsvd, tsvd!, leftorth, leftorth!, rightorth, rightorth!,
    permute, tie, distance, distance2, renyi_entropy,
    # 态
    AbstractMPS, MPS, DensityOperatorMPS,
    zero_state_mps, rand_state_mps, qubit_encoding_mps, onehot_encoding_mps,
    amplitude_encoding_mps, statevector_mps, densitymatrix_mps, densitymatrix,
    bond_dimensions, max_bond_dimension, physical_dimension, physical_dimensions,
    iscanonical, isrightcanonical, svectors_uninitialized,
    # 核心操作
    apply, apply!, apply_kraus!, reset_qubit_zero!,
    probabilities, marginal_probabilities, measure!, sample, post_select, post_select!,
    partial_tr, expectation, fidelity, overlap, amplitude, statevector,
    canonicalize!, normalize!, normalize_tr!,
    # QuantumCircuits.Interface 再导出
    simulate, simulate!,
    # QuantumCircuits 桥接层
    ClassicalStore, MatrixProductBackend

end # module
