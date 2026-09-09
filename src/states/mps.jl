# mps.jl — 纯态矩阵乘积态（MPS）
#
# 数据布局与公共约定见 `abstractmps.jl`；本文件只补充纯态专属内容：
# 物理维恒为 2，`dot` / `distance` 等按纯态重叠 `⟨a|b⟩` 计算。

"""
    MPS(data, [svectors]) <: AbstractMPS
    MPS{T,R}(data, svectors)

纯态矩阵乘积态（局域物理维 2）。数据布局与正则形式约定见
[`AbstractMPS`](@ref)；构造时校验键维匹配与开放边界。
"""
mutable struct MPS{T<:Number,R<:Real} <: AbstractMPS{T}
    data::Vector{Array{T,3}}
    svectors::Union{Nothing,Vector{Vector{R}}}

    function MPS{T,R}(data::Vector{<:AbstractArray{<:Number,3}},
                      svectors::Union{Nothing,Vector{Vector{R}}}) where {T<:Number,R<:Real}
        R == real(T) || throw(ArgumentError("scalar type for singular vectors must be real(T)"))
        _check_mps_data(data, svectors; d=2)
        new{T,R}(convert(Vector{Array{T,3}}, data), svectors)
    end
end

MPS(data::Vector{<:AbstractArray{T,3}}, svectors::Union{Nothing,Vector{Vector{R}}}=nothing) where {T<:Number,R<:Real} =
    MPS{T,real(T)}(data, svectors)

Base.eltype(::Type{MPS{T,R}}) where {T,R} = T

"""
    LinearAlgebra.dot(a::MPS, b::MPS)

重叠 `⟨a|b⟩`（全收缩，无截断）。
"""
function LinearAlgebra.dot(a::MPS, b::MPS)
    length(a) == length(b) || throw(DimensionMismatch("MPS length mismatch"))
    E = _ones_env(a, b)
    for k in 1:length(a)
        E = _transfer2(E, a.data[k], b.data[k])
    end
    return E[1, 1]
end

distance2(a::MPS, b::MPS) = _distance2(a, b)
distance(a::MPS, b::MPS) = _distance(a, b)

Base.isapprox(a::MPS, b::MPS; kwargs...) =
    length(a) == length(b) && distance2(a, b) <= (get(kwargs, :atol, 0) + get(kwargs, :rtol, 0) * max(norm(a), norm(b)))^2
