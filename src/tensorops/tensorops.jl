# tensorops.jl — 基础张量操作（与 TEMPO 的 `src/tensorops` 保持一致）
#
# 提供：截断方案（truncation.jl）、正交分解 / SVD（matrixalgebra.jl）、
# 张量级分解原语（tensorfactorizations.jl）与距离度量（distance.jl）。
# 底层由 MatrixAlgebraKit 驱动。

include("truncation.jl")
include("matrixalgebra.jl")
include("tensorfactorizations.jl")
include("distance.jl")
