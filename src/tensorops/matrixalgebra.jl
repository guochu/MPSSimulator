# matrixalgebra.jl — 矩阵正交 / 极分解（与 TEMPO 的 `src/tensorops/matrixalgebra.jl` 接口一致，
# 底层适配 MatrixAlgebraKit 0.6+）

using LinearAlgebra: BlasFloat
using MatrixAlgebraKit: MatrixAlgebraKit, left_orth!, right_orth!, left_polar!,
                        svd_compact!, svd_trunc!, trunctol, diagview

"数组/类型的标量算术类型。"
scalartype(::Type{T}) where {T<:Number} = T <: Real ? float(T) : complex(float(real(T)))
scalartype(::Type{T}) where {T} = Float64
scalartype(A::AbstractArray) = scalartype(eltype(A))

abstract type FactorizationAlgorithm end

"""
    OrthogonalFactorizationAlgorithm

正交分解算法抽象类型；具体算法包括 `QR` / `QRpos` / `LQ` / `LQpos` / `SVD` /
`SDD` / `Polar`，由 [MatrixAlgebraKit](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl) 驱动。
"""
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

"QR 分解（对偶为 `LQ`）；对角元符号不保证为正。"
struct QR <: OrthogonalFactorizationAlgorithm end
"对角元为正的 QR 分解（对偶为 `LQpos`）。"
struct QRpos <: OrthogonalFactorizationAlgorithm end
"LQ 分解（`QR` 的转置对偶）。"
struct LQ <: OrthogonalFactorizationAlgorithm end
"对角元为正的 LQ 分解。"
struct LQpos <: OrthogonalFactorizationAlgorithm end
"SVD 分解（LAPACK 分治驱动）。"
struct SDD <: OrthogonalFactorizationAlgorithm end
"SVD 分解（LAPACK QR 迭代驱动）。"
struct SVD <: OrthogonalFactorizationAlgorithm end
"极分解。"
struct Polar <: OrthogonalFactorizationAlgorithm end

Base.adjoint(::QRpos) = LQpos()
Base.adjoint(::QR) = LQ()
Base.adjoint(::LQpos) = QRpos()
Base.adjoint(::LQ) = QR()
Base.adjoint(alg::Union{SVD,SDD,Polar}) = alg

const OFA = OrthogonalFactorizationAlgorithm

"""
    leftorth!(A, alg=QRpos(), atol=0) -> (Q, R)

把矩阵 `A` 左正交分解为 `A = Q * R`（`Q` 列正交）。`alg` 可取
`QR` / `QRpos`（QR 分解，`atol` 须为 0）、`SVD` / `SDD`（SVD，
绝对值低于 `atol` 的奇异值被截断）或 `Polar`（极分解）。

注意：`A` 被当作工作区使用，可能被**覆盖销毁**；需保留输入请先拷贝。
"""
function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos},
                   atol::Real=zero(float(real(scalartype(A)))))
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return left_orth!(A; alg = :qr, positive = alg isa QRpos)
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD},
                   atol::Real=zero(float(real(scalartype(A)))))
    atol > 0 || return leftorth!(A, QR(), atol)
    u, s, v, _ = svd_trunc!(A; trunc = trunctol(atol = atol))
    return u, s * v
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar,
                   atol::Real=zero(float(real(scalartype(A)))))
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return left_polar!(A)
end

"""
    rightorth!(A, alg=LQpos(), atol=0) -> (L, Q)

把矩阵 `A` 右正交分解为 `A = L * Q`（`Q` 行正交）。`alg` 可取
`LQ` / `LQpos`、`SVD` / `SDD` 或 `Polar`。

注意：`A` 被当作工作区使用，可能被**覆盖销毁**；需保留输入请先拷贝。
"""
function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos},
                    atol::Real=zero(float(real(scalartype(A)))))
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return right_orth!(A; alg = :lq, positive = alg isa LQpos)
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD},
                    atol::Real=zero(float(real(scalartype(A)))))
    atol > 0 || return rightorth!(A, LQ(), atol)
    u, s, v, _ = svd_trunc!(A; trunc = trunctol(atol = atol))
    return u * s, v
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar,
                    atol::Real=zero(float(real(scalartype(A)))))
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    l, q = right_orth!(A; alg = :lq)
    return l, q
end
