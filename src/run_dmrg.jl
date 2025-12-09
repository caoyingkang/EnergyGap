## --------------------------- Set up ----------------------------------------
using CairoMakie
using DataFrames
using CSV
using HDF5

include("dmrg_utils.jl")


k_values = 1:1:30
df = DataFrame(k=k_values, E0=NaN, E1=NaN, ΔE=NaN, log10_ΔE=NaN)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "DMRG_energy_gap_vs_k.csv")
png_path = joinpath(results_dir, "DMRG_energy_gap_vs_k.png")
h5_path = joinpath(results_dir, "DMRG_mps.h5")

# Load existing data points if they exist
if isfile(csv_path)
    let saved_df = CSV.read(csv_path, DataFrame)
        for r in eachrow(saved_df)
            idx = findfirst(==(r.k), df.k)
            df[idx, :] = r
        end
    end
end


## --------------------------- Run DMRG ----------------------------------------
for r in eachrow(df)
    if !isnan(r.ΔE)
        continue
    end
    k = r.k
    println("\nComputing for k = $k (total sites: $(6*k))...")
    sites, H = build_hamiltonian(k)
    E0, ψ0, E1, ψ1 = compute_ground_and_first_excited_states(
        sites, H,
        nsweeps=100,
        maxdim=[10, 20, 100, 100, 200],
        cutoff=1e-10,
        outputlevel=1,
        conv_check_length=4,
        conv_tol=1e-8
    )
    r.E0 = E0
    r.E1 = E1
    r.ΔE = E1 - E0
    r.log10_ΔE = log10(E1 - E0)
    h5write(h5_path, "k=$k/psi0", ψ0)
    h5write(h5_path, "k=$k/psi1", ψ1)
    println(r)
end

## --------------------------- Plot results ------------------------------------
fig = Figure()
ax = Axis(fig[1, 1], xlabel="k", ylabel="ΔE", yscale=log10, title="Energy Gap vs k")
lines!(ax, df.k, df.ΔE)
scatter!(ax, df.k, df.ΔE)

# display(fig)
save(png_path, fig)

## --------------------------- Save results ------------------------------------
df_to_save = df[1:5, :]
CSV.write(csv_path, df_to_save)

## --------------------------- Sanity checks on MPS ----------------------------------------
f = h5open(h5_path, "r")
for k in 1:5
    psi0 = read(f, "k=$k/psi0", MPS)
    psi1 = read(f, "k=$k/psi1", MPS)
    @show inner(psi0, psi1)
end
close(f)