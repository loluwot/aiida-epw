#!/usr/bin/env -S julia --project=/home/aiida/workspace/aiida-epw-workflows/utils
# chk2ukk.jl

using WannierIO


function build_cu(chk)
    n_kpts = length(chk.kpoints)
    n_wann = chk.n_wann

    nbndep_per_k = [count(chk.dis_bands[ik]) for ik in 1:n_kpts]
    nbndep = nbndep_per_k[1]

    if !all(==(nbndep), nbndep_per_k)
        @warn "Disentanglement band count is not uniform across k-points. " *
              "Using value at k=1 ($(nbndep)). Results may be incorrect."
    end

    cu = zeros(ComplexF64, nbndep, n_wann, n_kpts)
    for ik in 1:n_kpts
        mask   = chk.dis_bands[ik]
        Udis_k = chk.Udis[ik][mask, :]   # (nbndep × n_wann)
        Uml_k  = chk.Uml[ik]              # (n_wann × n_wann)
        cu[:, :, ik] = Udis_k * Uml_k
    end

    return cu, nbndep
end

"""
Compute nbndskip: the number of contiguous excluded bands starting from
band index 1. EPW uses this as a simple bottom offset.
"""
function compute_nbndskip(exclude_bands, n_bands_full)
    isempty(exclude_bands) && return 0

    sorted = sort(collect(exclude_bands))
    nbndskip = 0
    for (i, b) in enumerate(sorted)
        b == i ? (nbndskip += 1) : break
    end

    non_bottom = filter(b -> b > nbndskip, sorted)
    if !isempty(non_bottom)
        @warn "Excluded bands outside the contiguous bottom block: " *
              "$(non_bottom). nbndskip=$(nbndskip). EPW may not handle " *
              "these correctly via nbndskip alone."
    end

    return nbndskip
end

"""
Build exband flag vector of length n_bands_full.
exband[i] = true if KS band i (1-based, full set) is excluded.
"""
function build_exband(exclude_bands, n_bands_full)
    exband = falses(n_bands_full)
    for b in exclude_bands
        1 <= b <= n_bands_full && (exband[b] = true)
    end
    return exband
end

"""
Build ibndkept: full KS indices (1-based) of bands inside the outer
disentanglement window.

WannierIO band indices in dis_bands are relative to the post-exclusion set.
This function maps them back to the full KS ordering.
"""
function build_ibndkept(chk, nbndep, n_bands_full)
    exclude_set  = Set(chk.exclude_bands)
    post_to_full = [i for i in 1:n_bands_full if i ∉ exclude_set]

    in_window = findall(chk.dis_bands[1])   # post-exclusion indices at k=1

    length(in_window) == nbndep || error(
        "Mismatch: nbndep=$(nbndep) but found $(length(in_window)) " *
        "bands in dis_bands at k=1.")

    return [post_to_full[j] for j in in_window]
end

function write_ukk(filename::String, chk, alat_ang::Float64)
    n_kpts       = length(chk.kpoints)
    n_wann       = chk.n_wann
    n_bands      = chk.n_bands
    n_excl       = length(chk.exclude_bands)
    n_bands_full = n_bands + n_excl

    cu, nbndep = build_cu(chk)
    nbndskip   = compute_nbndskip(chk.exclude_bands, n_bands_full)
    ibndkept   = build_ibndkept(chk, nbndep, n_bands_full)
    exband     = build_exband(chk.exclude_bands, n_bands_full)

    # lwin: all true — the window selection is already encoded in cu rows.
    lwin = trues(nbndep, n_kpts)

    # Wannier centers: Angstrom → dimensionless (divide by alat).
    # chk.r is a length-n_wann vector of Vec3{Float64} in Angstrom.
    w_centers = hcat([collect(chk.r[iw]) ./ alat_ang for iw in 1:n_wann]...)
    # Result: (3 × n_wann), dimensionless Cartesian.

    open(filename, "w") do io

        println(io, "$(nbndep) $(nbndskip)")

        for ibnd in 1:nbndep
            println(io, "$(ibndkept[ibnd])")
        end

        for ik in 1:n_kpts
            for ibnd in 1:nbndep
                for jbnd in 1:n_wann
                    c = cu[ibnd, jbnd, ik]
                    println(io, "($(real(c)),$(imag(c)))")
                end
            end
        end

        for ik in 1:n_kpts
            for ibnd in 1:nbndep
                println(io, lwin[ibnd, ik] ? "T" : "F")
            end
        end

        for ibnd in 1:n_bands_full
            println(io, exband[ibnd] ? "T" : "F")
        end

        for iw in 1:n_wann
            println(io,
                "$(w_centers[1,iw])  $(w_centers[2,iw])  $(w_centers[3,iw])")
        end
    end
end

function main()
    if length(ARGS) < 3
        exit(1)
    end
    chk_file = ARGS[1]
    alat_arg = ARGS[2]
    ukk_file = ARGS[3]
    # mmn_file = ARGS[4]
    # mmn_out = ARGS[5]

    qe = WannierIO.read_qe_xml(alat_arg)
    alat_ang = qe.alat
    chk = read_chk(chk_file)
    n_kpts  = length(chk.kpoints)
    n_wann  = chk.n_wann
    n_bands = chk.n_bands
    n_excl  = length(chk.exclude_bands)
    write_ukk(ukk_file, chk, alat_ang)
end

main()
