# ops.jl — IR 指令 → 核心原语

# ── 内部工具 ─────────────────────────────────────────────────────────────────

function _check_gateop(op::Operation, n::Int)
    qs = qubits(op)
    length(unique(qs)) == length(qs) || throw(ArgumentError("duplicate qubits in $op"))
    all(q -> 1 <= q <= n, qs) || throw(ArgumentError("qubit index out of range [1, $n] in $op"))
    return qs
end

# ── GateOp ────────────────────────────────────────────────────────────────────

"""
    apply!(state, op, [table]; trunc=DefaultMPSTruncation) -> state

把 IR 指令 `op` 就地作用到 MPS 态上，返回（可能被提升类型的）态。

支持：`GateOp`（酉门）、`ChannelOp`（噪声信道，纯态自动提升为
`DensityOperatorMPS`）、`MeasOp`（坍缩，测量结果丢弃）、`ResetOp`、
`BarrierOp`（无操作）、`BlockOp`（展开执行）。`IfOp` 需要经典寄存器
状态，请通过 `simulate` 执行。

`table`：可选的 `Dict{Param/Symbol => Real}` 参数表，绑定 `GateOp`
中的符号参数。`trunc`：截断方案（透传给核心层）。
"""
function apply!(s::Union{MPS,DensityOperatorMPS}, op::GateOp,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    n = length(s)
    _check_gateop(op, n)
    return apply!(s, mat(op, table), Tuple(qubits(op)); trunc=trunc)
end

# ── ChannelOp ────────────────────────────────────────────────────────────────

"""
    apply!(state, op::ChannelOp; trunc=DefaultMPSTruncation) -> DensityOperatorMPS

把 Kraus 信道作用到态上：`ρ ← Σₖ Kₖ ρ Kₖ†`。纯态输入自动提升为
`DensityOperatorMPS`（返回值随之改变类型）。
"""
function apply!(s::Union{MPS,DensityOperatorMPS}, op::ChannelOp,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    n = length(s)
    qs = qubits(op)
    length(unique(qs)) == length(qs) || throw(ArgumentError("duplicate qubits in $op"))
    all(q -> 1 <= q <= n, qs) || throw(ArgumentError("qubit index out of range [1, $n] in $op"))
    return apply_kraus!(s, kraus(op.channel), Tuple(qs); trunc=trunc)
end

# ── ResetOp / BarrierOp / MeasOp ────────────────────────────────────────────

"""
    apply!(state, op::ResetOp; trunc=DefaultMPSTruncation) -> state

把 `op.qubits` 中的每个比特相干重置到 `|0⟩`。
"""
function apply!(s::Union{MPS,DensityOperatorMPS}, op::ResetOp,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    n = length(s)
    _check_gateop(op, n)
    for q in qubits(op)
        s = reset_qubit_zero!(s, q; trunc=trunc)
    end
    return s
end

"""
    apply!(state, op::BarrierOp) -> state

屏障：无量子语义，直接返回。
"""
apply!(s::Union{MPS,DensityOperatorMPS}, ::BarrierOp,
       table::Union{Nothing,AbstractDict}=nothing;
       trunc::TruncationScheme=DefaultMPSTruncation) = s

"""
    apply!(state, op::MeasOp; trunc=DefaultMPSTruncation) -> state

测量 `op.qubits` 并坍缩（结果丢弃；需要写回经典位时请用 `simulate`）。
"""
function apply!(s::Union{MPS,DensityOperatorMPS}, op::MeasOp,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    n = length(s)
    _check_gateop(op, n)
    for q in qubits(op)
        measure!(s, q; trunc=trunc)
    end
    return s
end

# ── BlockOp ──────────────────────────────────────────────────────────────────

function apply!(s::Union{MPS,DensityOperatorMPS}, op::BlockOp,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    for mapped in unroll(op)
        s = apply!(s, mapped, table; trunc=trunc)
    end
    return s
end
