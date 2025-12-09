## --------------------------- Set up ----------------------------------------
using DataFrames
using CSV
using KrylovKit: eigsolve

include("ed_utils.jl")


results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "ED_energy_gap_vs_k.csv")


## --------------------------- Run ED calculation ------------------------------------

df = DataFrame(k=Int[], E0=Float64[], E1=Float64[], ΔE=Float64[], log10_ΔE=Float64[])
for k in 1:3
    println("\nComputing for k = $k (total sites: $(6*k))...")
    H = build_hamiltonian(Val(k))
    Es, ψs, info = eigsolve(H, 2, :SR; ishermitian=true, verbosity=3)
    ΔE = Es[2] - Es[1]
    push!(df, (k=k, E0=Es[1], E1=Es[2], ΔE=ΔE, log10_ΔE=log10(ΔE)))
end
CSV.write(csv_path, df)
