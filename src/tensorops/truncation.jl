# truncation.jl — 奇异值截断方案（与 TEMPO 的 `src/tensorops/truncation.jl` 一致）

"""
    TruncationScheme

张量截断方案抽象类型：规定奇异值按维数、按误差或两者结合的方式截断。
"""
abstract type TruncationScheme end

"""
    NoTruncation()

不截断，保留全部奇异值。
"""
struct NoTruncation <: TruncationScheme end

"按维数截断：只保留最大的 `D` 个奇异值。"
struct TruncateDim <: TruncationScheme
    D::Int
end
TruncateDim(; D::Int) = TruncateDim(D)
"""
    truncdim(d::Int)
    truncdim(; D::Int)

按维数截断：只保留最大的 `d` 个奇异值。
"""
truncdim(d::Int) = TruncateDim(d)
truncdim(; D::Int) = truncdim(D)

"按相对误差截断：丢弃相对范数低于 `ϵ` 的全部奇异值。"
struct TruncateCutoff <: TruncationScheme
    ϵ::Float64
end
TruncateCutoff(; ϵ::Real) = TruncateCutoff(convert(Float64, ϵ))
"""
    trunccutoff(; ϵ::Real)

按相对误差截断：丢弃相对范数低于 `ϵ` 的奇异值。
"""
trunccutoff(; ϵ::Real) = TruncateCutoff(ϵ)

"""
    TruncationDimCutoff(D, ϵ, add_back=0)
    TruncationDimCutoff(; D, ϵ, add_back=0)

维数 + 误差组合截断：先按相对范数 `ϵ` 确定截断点，再封顶 `D` 个奇异值，
且至少保留 `add_back` 个。
"""
struct TruncationDimCutoff <: TruncationScheme
    D::Int
    ϵ::Float64
    add_back::Int
end
TruncationDimCutoff(; D::Int, ϵ::Real, add_back::Int=0) =
    TruncationDimCutoff(D, float(ϵ), min(add_back, D))
"""
    truncdimcutoff(D, ϵ, add_back=0)
    truncdimcutoff(; D, ϵ, add_back=0)

`TruncationDimCutoff` 的位置参数 / 关键字便捷构造。
"""
truncdimcutoff(D::Int, epsilon::Real; add_back::Int=0) =
    TruncationDimCutoff(D, epsilon, min(add_back, D))
truncdimcutoff(; D::Int, ϵ::Real, add_back::Int=0) =
    TruncationDimCutoff(D, float(ϵ), min(add_back, D))

function _truncate!(v::AbstractVector{<:Real}, trunc::NoTruncation, p::Real=2)
    return v, 0.0
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncateDim, p::Real=2)
    dtrunc = min(length(v), trunc.D)
    truncerr = norm(view(v, dtrunc+1:length(v)), p)
    resize!(v, dtrunc)
    return v, truncerr
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncateCutoff, p::Real=2)
    sca = norm(v, p)
    dtrunc = findlast(Base.Fix2(>, sca * trunc.ϵ), v)
    dtrunc = isnothing(dtrunc) ? 0 : dtrunc
    return _truncate!(v, TruncateDim(dtrunc), p)
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncationDimCutoff, p::Real=2)
    sca = norm(v, p)
    dtrunc = findlast(Base.Fix2(>, sca * trunc.ϵ), v)
    dtrunc = isnothing(dtrunc) ? 0 : dtrunc
    dtrunc = max(dtrunc, trunc.add_back)   # 至少保留 add_back 个奇异值
    dtrunc = min(dtrunc, trunc.D)          # 但不超过 D 个
    v, err = _truncate!(v, TruncateDim(dtrunc), p)
    return v, err / sca
end
