## --------------------------- Set up ----------------------------------------
using DataFrames
using CSV
using KrylovKit: eigsolve
using QuantumToolbox: sigmax, sigmaz, multisite_operator

const X = sigmax()
const Z = sigmaz()


"""
Build the reduced Hamiltonian for the [[6k, 2k, 2]] code described in https://arxiv.org/abs/1911.01354.

The reduced Hamiltonian is obtained by projecting the original Hamiltonian onto the simultaneous +1 
eigenspace of the bare logical Z operators. It has the following form:

X        X                 X
|        |                 |
1---ZZ---3---ZZ--- ... ---2⋅(2k)-1
|        |                 |
ZZ       ZZ                ZZ
|        |                 |
2---XX---4---XX--- ... ---2⋅(2k)
"""
function build_reduced_hamiltonian(::Val{K}) where {K}
    m = 2 * K  # number of rungs
    # single-site terms
    H = (-1) * sum(
        multisite_operator(Val(4K), 2j - 1 => X)
        for j in 1:m
    )
    # horizontal couplings
    H += (-1) * sum(
        multisite_operator(Val(4K), 2j - 1 => Z, 2(j + 1) - 1 => Z) +
        multisite_operator(Val(4K), 2j => X, 2(j + 1) => X)
        for j in 1:(m-1)
    )
    # vertical couplings
    H += (-1) * sum(
        multisite_operator(Val(4K), 2j - 1 => Z, 2j => Z)
        for j in 1:m
    )
    return H.data
end

k_values = 1:6
df = DataFrame(k=k_values, E0=NaN, E1=NaN, ΔE=NaN, log10_ΔE=NaN)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "ED_reduced_ham_energy_gap_vs_k.csv")

# Load existing data points if they exist
if isfile(csv_path)
    let saved_df = CSV.read(csv_path, DataFrame)
        for r in eachrow(saved_df)
            idx = findfirst(==(r.k), df.k)
            df[idx, :] = r
        end
    end
end


## --------------------------- Run ED calculation ------------------------------------
for r in eachrow(df)
    if !isnan(r.ΔE)
        continue
    end
    k = r.k
    println("\nComputing for k = $k (total sites: $(6*k))...")
    H = build_reduced_hamiltonian(Val(k))
    # @show size(H), Base.format_bytes(Base.summarysize(H)), typeof(H)
    Es, ψs, info = eigsolve(H, 2, :SR; ishermitian=true, verbosity=3)
    r.E0 = Es[1]
    r.E1 = Es[2]
    r.ΔE = Es[2] - Es[1]
    r.log10_ΔE = log10(Es[2] - Es[1])
    println(r)
end


## --------------------------- Save results ------------------------------------
CSV.write(csv_path, df)