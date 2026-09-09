# test_states.jl — 态初始化与稠密转换

@testset "states" begin
    # 零态 / 基态编码
    psi = zero_state_mps(3)
    @test length(psi) == 3
    @test statevector(psi) ≈ VQC.storage(VQC.zero_state(3))
    @test statevector(onehot_encoding_mps([1, 0])) ≈ VQC.storage(VQC.onehot_encoding([1, 0]))
    @test statevector(statevector_mps(Float64, [1, 0])) ≈ [0, 1, 0, 0]

    # qubit 编码（直积态）
    θ = [0.3, 1.1, 0.7]
    @test statevector(qubit_encoding_mps(θ)) ≈ VQC.storage(VQC.qubit_encoding(θ))

    # 振幅编码
    v = randn(ComplexF64, 8)
    @test statevector(amplitude_encoding_mps(v)) ≈ VQC.storage(VQC.amplitude_encoding(v))
    # 截断振幅编码：键维受限
    vt = amplitude_encoding_mps(v; trunc=truncdim(D=2))
    @test max_bond_dimension(vt) <= 2

    # 随机态：归一化、右正则
    r = rand_state_mps(5; D=4)
    @test norm(r) ≈ 1 atol = 1e-12
    @test iscanonical(r)
    for k in 1:4
        M = reshape(r[k], size(r[k], 1), :)
        @test norm(M * M' - I(size(M, 1))) < 1e-10
    end
    @test VQC.probabilities(VQC.StateVector(statevector(r))) ≈ probabilities(r) atol = 1e-10

    # amplitude
    for bits in ([0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 1, 0, 1])
        @test amplitude(r, bits) ≈ VQC.amplitude(VQC.StateVector(statevector(r)), bits) atol = 1e-10
    end

    # dot / fidelity / distance 与稠密一致
    r2 = rand_state_mps(5; D=4)
    sv1 = statevector(r); sv2 = statevector(r2)
    @test dot(r, r2) ≈ dot(sv1, sv2) atol = 1e-10
    @test fidelity(r, r2) ≈ abs2(dot(sv1, sv2)) atol = 1e-10
    @test distance(r, r2) ≈ norm(sv1 - sv2) atol = 1e-10
end
