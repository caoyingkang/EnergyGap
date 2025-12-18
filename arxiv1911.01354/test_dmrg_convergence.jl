using Test
using DataFrames
using CSV
using HDF5
using ITensors, ITensorMPS

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "DMRG_energy_gap_vs_k.csv")
h5_path = joinpath(results_dir, "DMRG_mps.h5")

k_values = let df = CSV.read(csv_path, DataFrame)
    df.k
end

@testset "DMRG convergence tests" begin
    f = h5open(h5_path, "r")
    for k in k_values
        println("Testing k = $k...")
        psi0 = read(f, "k=$k/psi0", MPS)
        psi1 = read(f, "k=$k/psi1", MPS)

        @test inner(psi0, psi0) ≈ 1.0
        @test inner(psi1, psi1) ≈ 1.0
        @test isapprox(inner(psi0, psi1), 0.0, atol=1e-6)

        sites = siteinds(psi0)
        for j in 1:(length(sites)÷3)
            os = OpSum()
            os += "Z", 3 * j - 2, "Z", 3 * j - 1
            zz = MPO(os, sites)
            @test inner(psi0', zz, psi0) ≈ 1.0
            @test inner(psi1', zz, psi1) ≈ 1.0
        end
    end
    close(f)
end
