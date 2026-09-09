# postselect.jl — 后选择

"""
    post_select!(s::Union{MPS,DensityOperatorMPS}, q::Integer, val::Integer=0;
                 trunc=DefaultMPSTruncation) -> Float64

把 qubit `q` 后选择到 `val`（就地投影并归一化），返回发生概率。
"""
function post_select!(s::Union{MPS,DensityOperatorMPS}, q::Integer, val::Integer=0;
                      trunc::TruncationScheme=DefaultMPSTruncation)
    q = Int(q)
    val = Int(val)
    n = length(s)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    val in (0, 1) || throw(ArgumentError("post-selection value must be 0 or 1"))
    p0, p1 = marginal_probabilities(s, q)
    p = val == 0 ? p0 : p1
    p > 0 || throw(ArgumentError("post-selection probability is zero"))
    if s isa MPS
        apply!(s, _projector(val, eltype(s)) ./ sqrt(p), (q,); trunc=trunc)
    else
        P = _projector(val, eltype(s))
        apply!(s, P, (q,); trunc=trunc)
        normalize_tr!(s)
    end
    return p
end

"""
    post_select(s, q::Integer, val::Integer=0; trunc=DefaultMPSTruncation) -> (s', p)

非就地后选择：返回坍缩后的新态与发生概率。
"""
function post_select(s::Union{MPS,DensityOperatorMPS}, q::Integer, val::Integer=0;
                     trunc::TruncationScheme=DefaultMPSTruncation)
    out = copy(s)
    p = post_select!(out, q, val; trunc=trunc)
    return out, p
end
