# test_tensorops.jl — 张量操作层（与 TEMPO 一致的接口语义）

@testset "tensorops" begin
    A = randn(ComplexF64, 4, 3, 4)

    # tsvd（张量版）：无截断时精确重构
    u, s, v, err = tsvd(A, (1, 2), (3,))
    @test err ≈ 0 atol = 1e-14
    @test size(u) == (4, 3, length(s))
    um = reshape(u, :, length(s))
    @test norm(um * Diagonal(s) * v - reshape(A, 12, 4)) ≈ 0 atol = 1e-12
    @test all(s .>= 0)

    # 按维数截断
    u2, s2, v2, err2 = tsvd(A, (1, 2), (3,); trunc=truncdim(D=3))
    @test length(s2) == 3
    @test err2 ≈ norm(s[4:end]) atol = 1e-12

    # 按相对误差截断
    ϵ = 1e-8
    u3, s3, v3, err3 = tsvd(A, (1, 2), (3,); trunc=trunccutoff(ϵ=ϵ))
    @test all(s3 .> ϵ * s[1])
    @test err3 / s[1] <= ϵ + 1e-12

    # 维数 + 误差组合截断（add_back 保底）
    u4, s4, v4, err4 = tsvd(A, (1, 2), (3,); trunc=truncdimcutoff(D=2, ϵ=1e-12, add_back=2))
    @test length(s4) == 2

    # 矩阵版
    M = randn(ComplexF64, 16, 9)
    u5, s5, v5, err5 = tsvd(M)
    @test norm(u5 * Diagonal(s5) * v5 - M) ≈ 0 atol = 1e-12

    # leftorth / rightorth（张量版：分解 + 矩阵化重构）
    Q, R = leftorth(A, (1, 2), (3,); alg=QRpos())
    @test norm(reshape(Q, :, size(Q, 3))' * reshape(Q, :, size(Q, 3)) - I(size(Q, 3))) < 1e-12
    @test norm(reshape(Q, :, size(Q, 3)) * R - reshape(A, 12, 4)) < 1e-12
    L, Q2 = rightorth(A, (1, 2), (3,); alg=LQpos())
    @test norm(reshape(Q2, size(Q2, 1), :)' * reshape(Q2, size(Q2, 1), :) - I(size(Q2, 1))) < 1e-12
    @test norm(reshape(L, :, size(Q2, 1)) * Q2 - reshape(A, 12, 4)) < 1e-12

    # adjoint 对偶
    @test QRpos()' == LQpos()
    @test SDD()' == SDD()

    # distance
    x = randn(8); y = randn(8)
    @test distance(x, y) ≈ norm(x - y) atol = 1e-12
end
