using Test
using DataFrames
using CSV
using HDF5
using ITensors, ITensorMPS

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "DMRG_energy_gap_vs_N.csv")
h5_path = joinpath(results_dir, "DMRG_mps.h5")

N_values = let df = CSV.read(csv_path, DataFrame)
    df.N
end

@testset "DMRG convergence tests" begin
    f = h5open(h5_path, "r")
    for N in N_values
        println("Testing N = $N...")
        psi0 = read(f, "N=$N/psi0", MPS)
        psi1 = read(f, "N=$N/psi1", MPS)

        @test inner(psi0, psi0) ≈ 1.0
        @test inner(psi1, psi1) ≈ 1.0
        @test isapprox(inner(psi0, psi1), 0.0, atol=1e-6)

        sites = siteinds(psi0)
        for i in 1:N
            os = OpSum()
            os += "X", 2i, "X", 2i + 1
            xx = MPO(os, sites)
            @test inner(psi0', xx, psi0) ≈ 1.0
            @test inner(psi1', xx, psi1) ≈ 1.0
        end
    end
    close(f)
end
