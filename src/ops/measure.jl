# measure.jl — 计算基测量 / 采样（接口与 VQC 一致）

"""
    marginal_probabilities(s::Union{MPS,DensityOperatorMPS}, qs::AbstractVector{Int}) -> Vector{Float64}
    marginal_probabilities(s, q::Int) -> Vector{Float64}

指定比特子集 `qs` 的边缘概率分布：长度 `2^k`（`k = length(qs)`），
输出的第 `j` 位（LSB-first）对应 `qs[j]` 的取值。
"""
function marginal_probabilities(s::Union{MPS,DensityOperatorMPS}, qs::AbstractVector{Int})
    n = length(s)
    k = length(qs)
    all(q -> 1 <= q <= n, qs) || throw(ArgumentError("qubit index out of range [1, $n]"))
    length(unique(qs)) == k || throw(ArgumentError("duplicate qubits"))
    srt = sort!(collect(Int, qs))                  # 计算时按升序
    rdm = partial_tr(s, setdiff(1:n, srt))
    ps = real.(diag(rdm))
    k == n && return ps
    # 重排到用户给定顺序：out[j] 对应 qs[j]
    pos = [findfirst(==(q), srt) for q in qs]
    out = zeros(Float64, 1 << k)
    for idx in 0:(1 << k)-1
        j = 0
        for t in 1:k
            j |= ((idx >> (pos[t] - 1)) & 1) << (t - 1)
        end
        out[j+1] = ps[idx+1]
    end
    return out
end

marginal_probabilities(s::Union{MPS,DensityOperatorMPS}, q::Int) =
    marginal_probabilities(s, [q])

"全部计算基概率（长度 `2^n`）。"
probabilities(s::Union{MPS,DensityOperatorMPS}) = marginal_probabilities(s, collect(1:length(s)))

"""
    measure!(s, q::Integer; trunc=DefaultMPSTruncation) -> Int
    measure!(s, qs::AbstractVector{Int}; trunc=DefaultMPSTruncation) -> Vector{Int}
    measure!(s; trunc=DefaultMPSTruncation) -> Int

测量并**就地坍缩**：

* `measure!(s, q)`：测量 qubit `q`（1-based），返回结果（0/1）；
* `measure!(s, qs)`：依次测量一组比特，返回结果向量；
* `measure!(s)`：测量全部比特，返回所有结果按小端序组成的整数。
"""
function measure!(s::Union{MPS,DensityOperatorMPS}, q::Integer;
                  trunc::TruncationScheme=DefaultMPSTruncation)
    q = Int(q)
    n = length(s)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    p0, p1 = marginal_probabilities(s, q)
    outcome = rand() * (p0 + p1) < p0 ? 0 : 1
    p = outcome == 0 ? p0 : p1
    p > 0 || throw(ArgumentError("cannot collapse onto zero-probability outcome"))
    if s isa MPS
        apply!(s, _projector(outcome, eltype(s)) ./ sqrt(p), (q,); trunc=trunc)
    else
        P = _projector(outcome, eltype(s))
        apply!(s, P, (q,); trunc=trunc)            # ρ ← P ρ P
        normalize_tr!(s)                           # tr(P ρ P) = p
    end
    return outcome
end

function measure!(s::Union{MPS,DensityOperatorMPS}, qs::AbstractVector{Int};
                  trunc::TruncationScheme=DefaultMPSTruncation)
    return Int[measure!(s, q; trunc=trunc) for q in qs]
end

function measure!(s::Union{MPS,DensityOperatorMPS};
                  trunc::TruncationScheme=DefaultMPSTruncation)
    outcome = 0
    for q in 1:length(s)
        outcome |= measure!(s, q; trunc=trunc) << (q - 1)
    end
    return outcome
end

"""
    sample(s::Union{MPS,DensityOperatorMPS}, qs::AbstractVector{Int}, shots::Int) -> Dict{Int,Int}
    sample(s, shots::Int) -> Dict{Int,Int}

按计算基采样 `shots` 次，返回 `Dict{Int,Int}`：键为（子集）比特串的
整数值（`qubits[1]` = 最低位；全集采样时 = 基矢索引），值为出现次数。
（非破坏性：经边缘概率分布实现，指数级于子集大小。）
"""
function sample(s::Union{MPS,DensityOperatorMPS}, qs::AbstractVector{Int}, shots::Int)
    shots >= 0 || throw(ArgumentError("shots must be non-negative"))
    ps = marginal_probabilities(s, qs)
    return _sample_counts(ps, shots)
end

sample(s::Union{MPS,DensityOperatorMPS}, shots::Int) = sample(s, collect(1:length(s)), shots)

function _sample_counts(ps::AbstractVector{Float64}, shots::Int)
    cum = cumsum(ps)
    counts = Dict{Int,Int}()
    for _ in 1:shots
        i = searchsortedfirst(cum, rand())
        i = min(i, length(cum))
        counts[i-1] = get(counts, i-1, 0) + 1
    end
    return counts
end

# ── 投影（测量坍缩与后选择共用） ─────────────────────────────────────────────

"单比特投影算子 `|outcome⟩⟨outcome|`。"
function _projector(outcome::Int, T::Type{<:Number})
    P = zeros(T, 2, 2)
    P[outcome+1, outcome+1] = one(T)
    return P
end
