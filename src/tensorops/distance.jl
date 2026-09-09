# distance.jl — 数组距离度量（与 TEMPO 的 `src/tensorops/distance.jl` 一致）

function _distance2(x, y)
    sA = real(dot(x, x))
    sB = real(dot(y, y))
    c = dot(x, y)
    r = sA + sB - 2 * real(c)
    return abs(r)
end

_distance(x, y) = sqrt(_distance2(x, y))

"""
    distance2(x, y)

同形状数组（或张量网络）的平方距离：
`‖x - y‖² = ‖x‖² + ‖y‖² - 2 Re⟨x, y⟩`；返回绝对值以避免浮点舍入导致
的负数。参见 [`distance`](@ref)。
"""
distance2(x::AbstractArray{<:Number,N}, y::AbstractArray{<:Number,N}) where N = _distance2(x, y)

"""
    distance(x, y)

同形状数组的 2-范数距离，等于 `sqrt(distance2(x, y))`。
"""
distance(x::AbstractArray{<:Number,N}, y::AbstractArray{<:Number,N}) where N = _distance(x, y)
