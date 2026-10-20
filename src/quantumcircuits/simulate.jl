# simulate.jl — 线路演化（与 VQC 扩展层一致，附加 trunc 关键字）

# ── 参数环境 ─────────────────────────────────────────────────────────────────

"""
    _ParamEnv <: AbstractDict{Param,Float64}

模拟期的参数环境：按 `parameters(c)` 顺序存放绑定值 `vec`，并提供
`Param → 序号` 的 `index`。继承 `AbstractDict` 使其可直接供
`QuantumCircuits.mat(op, table)` 查表；`vec` 的向量索引对自动微分友好。
"""
struct _ParamEnv <: AbstractDict{Param,Float64}
    vec::Vector{Float64}
    index::Dict{Param,Int}

    function _ParamEnv(vec::Vector{Float64}, index::Dict{Param,Int})
        length(vec) == length(index) ||
            throw(ArgumentError("parameter values and index length mismatch"))
        new(vec, index)
    end
end

_ParamEnv() = _ParamEnv(Float64[], Dict{Param,Int}())

Base.getindex(e::_ParamEnv, p::Param) = e.vec[e.index[p]]
Base.haskey(e::_ParamEnv, p::Param) = haskey(e.index, p)
Base.keys(e::_ParamEnv) = keys(e.index)
Base.length(e::_ParamEnv) = length(e.vec)
Base.iterate(e::_ParamEnv, state...) = iterate(pairs(e.index), state...)

"""
    _param_env(c, params) -> _ParamEnv

* `params::Vector{<:Real}`：按 `parameters(c)` 顺序绑定；
* `params::Dict`：参数表（键可为 `Param` / `Symbol` / `ParamVector`）；
* `nothing`：不绑定。
"""
function _param_env(c::Circuit, params::AbstractVector{<:Real})
    ps = parameters(c)
    length(params) == length(ps) ||
        throw(ArgumentError("circuit has $(length(ps)) symbolic parameter(s), got $(length(params)) values"))
    return _ParamEnv(Float64.(params), Dict{Param,Int}(p => i for (i, p) in enumerate(ps)))
end

function _param_env(c::Circuit, table::AbstractDict)
    ps = parameters(c)
    t = Dict{Param,Float64}()
    for (k, v) in table
        if k isa ParamVector
            for (i, vv) in enumerate(v)
                t[k[i]] = Float64(vv)
            end
        elseif k isa Param
            t[k] = Float64(v)
        elseif k isa Symbol
            t[Param(k)] = Float64(v)
        else
            throw(ArgumentError("invalid parameter key $(k)"))
        end
    end
    vec = Float64[get(t, p) do
                      throw(ArgumentError("parameter $(p.name) is not bound"))
                  end for p in ps]
    return _ParamEnv(vec, Dict{Param,Int}(p => i for (i, p) in enumerate(ps)))
end

_param_env(::Circuit, ::Nothing) = _ParamEnv()

# ── 演化 ─────────────────────────────────────────────────────────────────────

# simulate / simulate! 挂到 QuantumCircuits.Interface 的同名泛型上
# （`using QuantumCircuits` 时即可直接调用，无需限定）。
import QuantumCircuits.Interface: simulate, simulate!

"""
    simulate(c::Circuit, state; params=nothing, trunc=DefaultMPSTruncation) -> state'

演化整条线路（`simulate` 非就地；对应的 `simulate!` 为就地版本）。
测量结果写回经典寄存器，供 `IfOp` 条件分支使用。

`params`：

* `Vector{<:Real}`：按 `parameters(c)` 顺序绑定符号参数；
* `Dict`：参数表（键可为 `Param` / `Symbol` / `ParamVector`）；
* `nothing`：不绑定（线路须无未绑定符号参数）。

`trunc`：每个门作用时的 MPS 截断方案。
"""
function simulate(c::Circuit, s::Union{MPS,DensityOperatorMPS};
                  params=nothing, trunc::TruncationScheme=DefaultMPSTruncation)
    return _simulate_bound(c, copy(s), _param_env(c, params), trunc)
end

"""
    simulate!(c::Circuit, state; params=nothing, trunc=DefaultMPSTruncation) -> state

`simulate` 的就地版本：把整条线路作用到 `state` 上（就地修改）。
参数与截断语义见 [`simulate`](@ref)。
"""
function simulate!(c::Circuit, s::Union{MPS,DensityOperatorMPS};
                   params=nothing, trunc::TruncationScheme=DefaultMPSTruncation)
    return _simulate_bound!(c, s, _param_env(c, params), trunc)
end

function _simulate_bound(c::Circuit, s::Union{MPS,DensityOperatorMPS},
                         env::_ParamEnv, trunc::TruncationScheme)
    return _simulate_bound!(c, copy(s), env, trunc)
end

function _simulate_bound!(c::Circuit, s::Union{MPS,DensityOperatorMPS}, env::_ParamEnv,
                          trunc::TruncationScheme)
    return _simulate_bound!(c, s, env, trunc, ClassicalStore(c))
end

# 带 ClassicalStore 的变体：后端可在每 shot 结束后读取经典寄存器值
function _simulate_bound!(c::Circuit, s::Union{MPS,DensityOperatorMPS}, env::_ParamEnv,
                          trunc::TruncationScheme, store::ClassicalStore)
    for op in c.ops
        s = _apply_with_store!(s, op, store, env, trunc)
    end
    return s
end

function _apply_with_store!(s, op::Operation, store::ClassicalStore, env::_ParamEnv,
                            trunc::TruncationScheme)
    return apply!(s, op, env; trunc=trunc)
end

function _apply_with_store!(s, op::MeasOp, store::ClassicalStore, env::_ParamEnv,
                            trunc::TruncationScheme)
    for (q, cb) in zip(op.qubits, op.clbits)
        outcome = measure!(s, q; trunc=trunc)
        set_clbit!(store, cb, outcome)
    end
    return s
end

function _apply_with_store!(s, op::IfOp, store::ClassicalStore, env::_ParamEnv,
                            trunc::TruncationScheme)
    return apply!(s, op, store, env; trunc=trunc)
end
