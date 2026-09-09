# tensorfactorizations.jl — 张量级分解原语（与 TEMPO 的
# `src/tensorops/tensorfactorizations.jl` 一致）

scalar(x::AbstractArray{T}) where {T<:Number} = only(x)

"""
    permute(m::AbstractArray, perm)

按 `perm` 置换 `m` 的维度，返回视图（`PermutedDimsArray`），不复制数据。
"""
permute(m::AbstractArray, perm) = PermutedDimsArray(m, perm)
"""
    permute(m::AbstractArray, left, right)

把维度分组为 `left` 与 `right` 并按此顺序排列，等价于 `permute(m, (left..., right...))`。
"""
permute(m::AbstractArray, left, right) = permute(m, (left..., right...))

function random_hermitian(::Type{T}, n::Int) where {T<:Number}
    m = randn(T, n, n)
    return m + m'
end

random_unitary(::Type{T}, n::Int) where {T<:Number} = exp(im .* random_hermitian(T, n))

function isometry(::Type{T}, m::Int, n::Int) where {T<:Number}
    r = zeros(T, m, n)
    for i in 1:min(m, n)
        r[i, i] = 1
    end
    return r
end
isometry(::Type{T}, d::Int) where {T<:Number} = isometry(T, d, d)
isometry(m::Int, n::Int) = isometry(Float64, m, n)
isometry(d::Int) = isometry(Float64, d)

function _group_extent(extent::NTuple{N,Int}, idx::NTuple{N1,Int}) where {N,N1}
    ext = Vector{Int}(undef, N1)
    l = 0
    for i = 1:N1
        ext[i] = prod(extent[(l+1):(l+idx[i])])
        l += idx[i]
    end
    return NTuple{N1,Int}(ext)
end

"""
    tie(a::AbstractArray, axs::NTuple{N1,Int})

把 `a` 的 `N` 个轴按 `axs` 分组合并（reshape），`sum(axs) == ndims(a)`。
"""
function tie(a::AbstractArray{T,N}, axs::NTuple{N1,Int}) where {T,N,N1}
    (sum(axs) != N) && error("total number of axes should equal to tensor rank.")
    return reshape(a, _group_extent(size(a), axs))
end

function Base.kron(a::AbstractArray{Ta,N}, b::AbstractArray{Tb,N}) where {Ta<:Number,Tb<:Number,N}
    N == 0 && error("empty tensors.")
    sa = size(a)
    sb = size(b)
    sc = ntuple(i -> sa[i] * sb[i], Val(N))
    c = Array{promote_type(Ta,Tb),N}(undef, sc)
    for index in CartesianIndices(a)
        ranges = ntuple(j -> ((index[j]-1)*sb[j]+1):(index[j]*sb[j]), Val(N))
        c[ranges...] = a[index] * b
    end
    return c
end

"""
    tsvd!(a; trunc=NoTruncation(), alg=SDD()) -> (u, s, v, err)

对矩阵 `a` 做带截断的奇异值分解，返回 `(u, s, v, err)`；`err` 为被丢弃
奇异值的 2-范数，`trunc` 为截断方案，`alg` 选择 SVD 驱动
（`SDD()` 分治 / `SVD()` QR 迭代）。

注意：输入矩阵 `a` 被当作工作区使用，可能被**覆盖销毁**；需保留输入请先拷贝。
"""
function tsvd!(a::StridedMatrix; trunc::TruncationScheme=NoTruncation(), alg::Union{SVD,SDD}=SDD())
    u, S, v = svd_compact!(a)
    s = Vector(diagview(S))
    s, err = _truncate!(s, trunc)
    d = length(s)
    if d == length(diagview(S))
        return u, s, v, err
    else
        return u[:, 1:d], s, v[1:d, :], err
    end
end

"""
    tsvd(a; trunc=NoTruncation(), alg=SDD())

`tsvd!` 的非就地版本：先拷贝输入再做分解。
"""
tsvd(a::AbstractMatrix; kwargs...) = tsvd!(copy(a); kwargs...)

"""
    tsvd!(a, left, right; trunc=NoTruncation(), alg=SDD())

把张量 `a` 的维度分组为 `left` / `right` 后做带截断的奇异值分解，
返回 `(u, s, v, err)`。

注意：输入张量 `a` 本身不被修改（内部分组置换会拷贝工作区）；
`!` 表示该变体复用该拷贝作为工作区。
"""
function tsvd!(a::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int};
               trunc::TruncationScheme=NoTruncation(), alg::Union{SVD,SDD}=SDD()) where {T<:Number,N,N1,N2}
    b, ushape, vshape = _tomat(a, left, right)
    isa(b, StridedMatrix) || (b = copy(b))
    u, s, v, err = tsvd!(b; trunc=trunc, alg=alg)
    md = length(s)
    return reshape(u, (ushape..., md)), s, reshape(v, (md, vshape...)), err
end

tsvd(a::AbstractArray, left::NTuple{N1,Int}, right::NTuple{N2,Int}; kwargs...) where {N1,N2} =
    tsvd!(copy(a), left, right; kwargs...)

"""
    leftorth!(A; alg=QRpos(), atol=0)

矩阵版 `leftorth!` 的关键字形式，等价于 `leftorth!(A, alg, atol)`。
"""
leftorth!(A::StridedMatrix; alg::Union{QR,QRpos,SVD,SDD,Polar}=QRpos(),
          atol::Real=zero(float(real(scalartype(A))))) = leftorth!(A, alg, atol)

"""
    leftorth!(A, left, right; alg=QRpos(), atol=0)

把张量 `A` 的维度分组为 `left` / `right` 后做左正交分解，返回 `(u, v)`：
`u` 的维度为 `(left..., s)`，`v` 的维度为 `(s, right...)`。

注意：输入张量 `A` 本身不被修改；`!` 表示复用内部拷贝作为工作区。
"""
function leftorth!(A::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int};
                   alg::Union{QR,QRpos,SVD,SDD,Polar}=QRpos(),
                   atol::Real=zero(float(real(scalartype(A))))) where {T,N,N1,N2}
    A2, dimu, dimv = _tomat(A, left, right)
    isa(A2, StridedMatrix) || (A2 = copy(A2))
    u, v = leftorth!(A2, alg, atol)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

"""
    leftorth(A; alg=QRpos(), atol=0)
    leftorth(A, left, right; alg=QRpos(), atol=0)

`leftorth!` 的非就地版本：先拷贝输入再分解。
"""
leftorth(A::AbstractMatrix; kwargs...) = leftorth!(copy(A); kwargs...)
leftorth(A::AbstractArray, left::NTuple{N1,Int}, right::NTuple{N2,Int}; kwargs...) where {N1,N2} =
    leftorth!(copy(A), left, right; kwargs...)

"""
    rightorth!(A; alg=LQpos(), atol=0)

矩阵版 `rightorth!` 的关键字形式，等价于 `rightorth!(A, alg, atol)`。
"""
rightorth!(A::StridedMatrix; alg::Union{LQ,LQpos,SVD,SDD,Polar}=LQpos(),
           atol::Real=zero(float(real(scalartype(A))))) = rightorth!(A, alg, atol)

"""
    rightorth!(A, left, right; alg=LQpos(), atol=0)

把张量 `A` 的维度分组为 `left` / `right` 后做右正交分解，返回 `(u, v)`。
"""
function rightorth!(A::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int};
                    alg::Union{LQ,LQpos,SVD,SDD,Polar}=LQpos(),
                    atol::Real=zero(float(real(scalartype(A))))) where {T,N,N1,N2}
    A2, dimu, dimv = _tomat(A, left, right)
    isa(A2, StridedMatrix) || (A2 = copy(A2))
    u, v = rightorth!(A2, alg, atol)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

rightorth(A::AbstractMatrix; kwargs...) = rightorth!(copy(A); kwargs...)
rightorth(A::AbstractArray, left::NTuple{N1,Int}, right::NTuple{N2,Int}; kwargs...) where {N1,N2} =
    rightorth!(copy(A), left, right; kwargs...)

function _tomat(a::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int}) where {T<:Number,N,N1,N2}
    (N == N1 + N2) || throw(DimensionMismatch())
    newindex = (left..., right...)
    a1 = permute(a, newindex)
    shape_a = size(a1)
    dimu = ntuple(i -> shape_a[i], N1)
    dimv = ntuple(i -> shape_a[N1+i], N2)
    return reshape(a1, prod(dimu), prod(dimv)), dimu, dimv
end

"""
    renyi_entropy(v::AbstractVector{<:Real}; α::Real=1)

向量 `v` 的 Rényi 熵：`α=1` 时为 Shannon 熵 `-Σᵢ vᵢ log(vᵢ)`，
否则为 `1/(1-α) * log(Σᵢ vᵢ^α)`。`v` 须非负且和为 1（如归一化的
奇异值平方）。
"""
function renyi_entropy(v::AbstractVector{<:Real}; α::Real=1)
    α = convert(eltype(v), α)
    a = _check_and_filter(v)
    if α == one(α)
        return -dot(a, log.(a))
    else
        a = a .^ (α)
        return (1 / (1 - α)) * log(sum(a))
    end
end

function _check_and_filter(v::AbstractVector{<:Real}; tol::Real=1.0e-12)
    (abs(sum(v) - 1) <= tol) || throw(ArgumentError("sum of singular values not equal to 1"))
    oo = zero(eltype(v))
    for item in v
        ((item < oo) && (-item > tol)) && throw(ArgumentError("negative singular values"))
    end
    return [item for item in v if abs(item) > tol]
end
