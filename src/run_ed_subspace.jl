## --------------------------- Set up ----------------------------------------
using DataFrames
using CSV
using SparseArrays
using KrylovKit: eigsolve
using QuantumToolbox: sigmax, sigmaz, multisite_operator, ⊗

const X = sigmax()
const Z = sigmaz()


"""
Build the penalty Hamiltonian for the [[6k, 2k, 2]] code described in https://arxiv.org/abs/1911.01354.

The sites of the lattice are indexed in this way:

1---ZZ---4---ZZ--- ... ---3⋅(2k)-2
|        |                 |
XX       XX                XX
|        |                 |
2        5         ...    3⋅(2k)-1
|        |                 |
ZZ       ZZ                ZZ
|        |                 |
3---XX---6---XX--- ... ---3⋅(2k)
"""
function build_Hpen(::Val{K}) where {K}
    m = 2 * K  # number of rungs
    # horizontal couplings
    Hpen = (-1) * sum(
        multisite_operator(Val(6K), 3j - 2 => Z, 3(j + 1) - 2 => Z) +
        multisite_operator(Val(6K), 3j => X, 3(j + 1) => X)
        for j in 1:(m-1)
    )
    # vertical couplings
    Hpen += (-1) * sum(
        multisite_operator(Val(6K), 3j - 2 => X, 3j - 1 => X) +
        multisite_operator(Val(6K), 3j - 1 => Z, 3j => Z)
        for j in 1:m
    )
    return Hpen
end


function build_projector(::Val{K}) where {K}
    m = 2 * K  # number of rungs
    # Bare logical Z operators
    LZs = [multisite_operator(Val(6K), 3j - 2 => Z, 3j - 1 => Z) for j in 1:m]
    # Projector onto their simultaneous +1 eigenspace
    P = foldl(*, (1 + lz) / 2 for lz in LZs)
    return P
end


function build_reduced_Hpen(::Val{K}) where {K}
    P = build_projector(Val(K))
    Hpen = build_Hpen(Val(K))
    H = (P * Hpen * P).data
    @assert size(H) == (2^(6K), 2^(6K))
    rows, _, _ = findnz(H)
    rows = sort(unique(rows))
    H_reduced = H[rows, rows]
    @assert size(H_reduced) == (2^(4K), 2^(4K))
    return H_reduced
end

k_values = 1:4
df = DataFrame(k=k_values, E0=NaN, E1=NaN, ΔE=NaN, log10_ΔE=NaN)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "ED_subspace_energy_gap_vs_k.csv")

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
    H = build_reduced_Hpen(Val(k))
    # @show size(H), typeof(H), Base.format_bytes(Base.summarysize(H))
    Es, ψs, info = eigsolve(H, 2, :SR; ishermitian=true, verbosity=3)
    r.E0 = Es[1]
    r.E1 = Es[2]
    r.ΔE = Es[2] - Es[1]
    r.log10_ΔE = log10(Es[2] - Es[1])
    println(r)
end


## --------------------------- Save results ------------------------------------
CSV.write(csv_path, df)
