# test_measure.jl — 测量 / 采样 / 后选择 / 偏迹

@testset "measure" begin
    n = 4
    # Bell 态
    c = Circuit(2)
    push!(c, H(1))
    push!(c, CX(1, 2))
    bell = simulate(c, zero_state_mps(2))
    bell_ref = simulate(c, VQC.zero_state(2))

    @test probabilities(bell) ≈ VQC.probabilities(bell_ref) atol = 1e-12
    @test marginal_probabilities(bell, 1) ≈ [0.5, 0.5] atol = 1e-12
    @test marginal_probabilities(bell, [2, 1]) ≈ [0.5, 0, 0, 0.5] atol = 1e-12

    # sample：统计一致
    counts = sample(bell, 2000)
    @test abs(get(counts, 0, 0) - get(counts, 3, 0)) < 150
    counts2 = sample(bell, [2], 2000)
    @test get(counts2, 0, 0) + get(counts2, 1, 0) == 2000
    @test abs(get(counts2, 0, 0) - get(counts2, 1, 0)) < 150

    # measure! 坍缩：结果与概率
    m = copy(bell)
    outcome = measure!(m, 1)
    @test outcome in (0, 1)
    @test probabilities(m) ≈ (outcome == 0 ? [1.0, 0, 0, 0] : [0, 0, 0, 1.0]) atol = 1e-10

    # post_select
    ps, p0 = post_select(bell, 1, 0)
    @test p0 ≈ 0.5 atol = 1e-12
    @test probabilities(ps) ≈ [1, 0, 0, 0] atol = 1e-10

    # partial_tr 与 VQC 一致（随机电路后的纠缠态）
    Random.seed!(7)
    cv = variational_circuit_1d(n, 2; θs = rand(36) .* 2π)
    state = simulate(cv, zero_state_mps(n))
    ref = simulate(cv, VQC.zero_state(n))
    for sites in ([1], [2, 3], [1, 4], [1, 2, 3])
        rdm = partial_tr(state, sites)
        ref_dm = VQC.partial_tr(ref, sites)
        ref_mat = length(sites) == n ? VQC.tr(ref_dm) :
                  VQC.storage(ref_dm)
        @test rdm ≈ ref_mat atol = 1e-8
    end
    @test partial_tr(state, Int[]) ≈ VQC.storage(VQC.partial_tr(ref, Int[])) atol = 1e-8

    # 混合态测量 / 偏迹
    rho = densitymatrix(state)
    @test marginal_probabilities(rho, 1) ≈ VQC.marginal_probabilities(ref, 1) atol = 1e-8
    @test partial_tr(rho, [1, 2]) ≈ VQC.storage(VQC.partial_tr(ref, [1, 2])) atol = 1e-8
    rm = copy(rho)
    o = measure!(rm, 1)
    @test o in (0, 1)
    @test tr(rm) ≈ 1 atol = 1e-8
end
