# classical.jl — IfOp 执行（经典寄存器运行时已上移至 QuantumCircuits）
#
# ClassicalStore / reg_value / set_clbit! / eval_cond 由
# QuantumCircuits 统一提供（2026-10 上移，三后端共享）；本文件只保留
# MPS 态上的 IfOp 分支执行与 QuantumCircuits 协议扩展。

using QuantumCircuits: eval_cond

"""
    apply!(state, op::IfOp, store::ClassicalStore; trunc=DefaultMPSTruncation) -> state

执行经典条件分支：按 `store` 求值 `op.cond`，执行 `op.then` 或
`op.otherwise` 子线路。需直接调用时请使用 `simulate`。
"""
function apply!(s::Union{MPS,DensityOperatorMPS}, op::IfOp, store::ClassicalStore,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    branch = eval_cond(op.cond, store) ? op.then : op.otherwise
    branch === nothing && return s
    for o in branch.ops
        s = apply!(s, o, table; trunc=trunc)
    end
    return s
end

function apply!(s::Union{MPS,DensityOperatorMPS}, op::IfOp,
                table::Union{Nothing,AbstractDict}=nothing;
                trunc::TruncationScheme=DefaultMPSTruncation)
    throw(ArgumentError("`apply!` on IfOp requires classical register values; use `simulate` instead"))
end

# ── QuantumCircuits 协议扩展：nqubits / measure ─────────────────────────────

"状态类型接入 QuantumCircuits 的 `nqubits` 协议。"
nqubits(x::MPS) = length(x)
nqubits(x::DensityOperatorMPS) = length(x)

"""
    measure(s::Union{MPS,DensityOperatorMPS}, q::Integer; trunc=DefaultMPSTruncation)
    -> (s', outcome, p)

QuantumCircuits `measure` 协议的态方法：非就地测量，返回坍缩后的
新态、结果与概率。
"""
function measure(s::Union{MPS,DensityOperatorMPS}, q::Integer;
                 trunc::TruncationScheme=DefaultMPSTruncation)
    out = copy(s)
    p0, p1 = marginal_probabilities(out, q)
    outcome = measure!(out, q; trunc=trunc)
    return out, outcome, (outcome == 0 ? p0 : p1)
end
