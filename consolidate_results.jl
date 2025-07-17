using JSON
using CSV
using DataFrames
using Plots
using Printf
using Statistics
using StatsPlots

include("functions.jl")

all_res_files = filter(s -> !occursin("_fw", s), readdir("results_inexact/", join=true))
all_fw_res_files = filter(s -> occursin("_fw", s), readdir("results_inexact/", join=true))

all_species = ("human", "zebrafish", "salmon", "mouse")

graph_stats = JSON.parsefile("graph_inexact_stats.json")

data_vec = []
for resfile in all_res_files
    for species in all_species
        if !occursin(species, resfile)
            continue
        end
        f_str = open(resfile) do f
            read(f, String)
        end
        if length(f_str) < 2
            continue
        end
        res = JSON.parse(f_str)
        if !haskey(res, "species")
            res["species"] = species
        end
        file_idx = parse(Int, split(split(resfile, "_")[end], ".")[1])
        
        res["idx"] = file_idx
        for key_name in ("nv", "ne", "k", "err0", "err1")
            res[key_name] = graph_stats[species][string(file_idx)][key_name]
        end
        fw_res = JSON.parsefile("results_inexact/result_$(species)_$(file_idx)_fw.json")
        for key_name in ("npaths_fw_poisson", "err_fw_poisson", "time_fw_poisson", "err_fw_flow", "err_fw_poisson_flow", "dualgap_poisson", "time_fw_poisson", "niter_poisson")
            res[key_name] = fw_res[key_name]
        end
        # fix inconsistent naming
        res["err_flow_fw_poisson"] = fw_res["err_fw_poisson_flow"]
        if res["err_fw"] != fw_res["err_fw"]
            @show (res["err_fw"], fw_res["err_fw"])
            @show resfile
        end
        push!(data_vec, res)
    end
end

df = DataFrame(data_vec)

# for col in (:err_fw, :err_fw_cb, :err_scip, :err_scip_rob)
#     @printf(" %.2f \\pm %.2f ", mean(df[:,col]), std(df[:,col]))
#     print(" \\& ")
# end

# for col in (:err_fw, :err_fw_cb, :err_scip, :err_scip_rob)
#     @printf(" %.2f \\pm %2f ", mean(log.(1 .+ df[:,col])), std(log.(1 .+ df[:,col])))
#     print(" \\& ")
# end

function geom_shifted_mean(xs; shift=big"1.0")
    if !isa(shift, BigFloat)
        shift = big(shift)
    end
    n = length(xs)
    r = prod(xi + shift for xi in xs)
    return Float64(r^(1/n) - shift)
end

rows_scip_sol = df[:,:time_scip_sol] .< 1800
rows_scip_rob_sol = df[:,:time_scip_rob_sol] .< 1800
rows_both_sol = rows_scip_sol .& rows_scip_rob_sol

rows_both_opt = (df[:,:time_scip] .< 1800) .& (df[:,:time_scip_rob] .< 1800)

row_indices_filters = [
    (:),
    (:),
    (:),
    (rows_scip_sol),
    (rows_scip_rob_sol),
]

rows_optimal_poisson = df[:,:niter_poisson] .< 5002

minimum_error_path = [min(df[ridx, :err_fw], df[ridx, :err_fw_cb], df[ridx, :err_fw_poisson], df[ridx, :err_scip], df[ridx, :err_scip_rob]) for ridx in 1:size(df, 1)]


minimum_error_flow = [min(df[ridx, :err_flow_fw], df[ridx, :err_flow_fw_cb], df[ridx, :err_flow_fw_poisson], df[ridx, :err_flow_scip], df[ridx, :err_flow_scip_rob]) for ridx in 1:size(df, 1)]

minimum_npaths = [min(df[ridx, :npaths_fw], df[ridx, :npaths_fw_cb], df[ridx, :npaths_fw_poisson], df[ridx, :npaths_scip], df[ridx, :npaths_scip_rob]) for ridx in 1:size(df, 1)]


print("Error & ")
for col in (:err_fw, :err_fw_cb, :err_fw_poisson, :err_scip, :err_scip_rob)
        @printf(" %.2f/%.2f/%i ", mean(df[rows_both_sol,col]), geom_shifted_mean(df[rows_both_sol,col]), count(df[:,col] .== minimum_error_path))
    print(" & ")
end
println()
print("Flow error & ")
for col in (:err_flow_fw, :err_flow_fw_cb, :err_flow_fw_poisson, :err_scip, :err_scip_rob)
    @printf(" %.2e/%.2e/%i ", mean(df[rows_both_sol,col]), geom_shifted_mean(df[rows_both_sol,col]), count(df[:,col] .== minimum_error_flow))
    print(" & ")
end
println()
print("Flow error (opt) & ")
for col in (:err_flow_fw, :err_flow_fw_cb, :err_flow_fw_poisson, :err_scip, :err_scip_rob)
    @printf(" %.2e/%.2e ", mean(df[rows_both_opt,col]), geom_shifted_mean(df[rows_both_opt,col]))
    print(" & ")
end
println()
print("Flow rel.~err. & ")
for col in (:err_flow_fw, :err_flow_fw_cb, :err_flow_fw_poisson, :err_scip, :err_scip_rob)
    @printf(" %.2e/%.2e ", mean(df[rows_both_sol,col] ./ df[rows_both_sol,:ne] ), geom_shifted_mean(df[rows_both_sol,col] ./ df[rows_both_sol,:ne]))
    print(" & ")
end
println()
print("\\# paths & ")
for (col, rows_idx) in zip((:npaths_fw, :npaths_fw_cb, :npaths_fw_poisson, :npaths_scip, :npaths_scip_rob), row_indices_filters)
    @printf(" %.2f/%.2f/%i ", mean(df[rows_both_sol,col]), geom_shifted_mean(df[rows_both_sol,col]), count(df[:,col] .== minimum_npaths))
    print(" & ")
end
println()
print("Time (s) & ")
for col in (:time_fw, :time_fw_cb, :time_fw_poisson, :time_scip, :time_scip_rob)
    @printf(" %.2e ", geom_shifted_mean(df[:,col]))
    print(" & ")
end

@info size(df, 1)

df_by_species = groupby(df, :species)

col_groups = [
    ("\\# paths", (:npaths_fw, :npaths_fw_cb, :npaths_scip, :npaths_scip_rob)),
    ("Error", (:err_fw, :err_fw_cb, :err_scip, :err_scip_rob)),
    ("Time (s)", (:time_fw, :time_fw_cb, :time_scip, :time_scip, :time_scip_rob)),
]

# by species results
print_species = false
for (col_name, cols) in col_groups
    print("$col_name & total & ")
    for col in cols
        @printf(" %.2f / %.2f ", mean(df[:,col]), geom_shifted_mean(df[:,col]))
        print(" & ")
    end
    println("\\\\")
    if !print_species
        continue
    end
    for subdf in df_by_species
        species = subdf[1,:species][1]
        print(" & $species & ")
        for col in cols
            @printf(" %.2f / %.2f ", mean(subdf[:,col]), geom_shifted_mean(subdf[:,col]))
            print(" & ")
        end
        println("\\\\")
    end
end


# julia> mean(df[:,:err_fw_cb])
# 2.2301587301587302

# julia> mean(df[:,:err_fw])
# 2.5674603174603177

# julia> mean(df[:,:err_scip])
# 3.0813492063492065

# julia> mean(df[:,:err_scip_rob])
# 1.632936507936508

# julia> mean(log.(1 .+ df[:,:err_scip_rob]))
# 0.6896399359639055

# julia> mean(log.(1 .+ df[:,:err_scip]))
# 1.3291893020605767

# julia> mean(log.(1 .+ df[:,:err_fw]))
# 1.1050295205918874

# julia> mean(log.(1 .+ df[:,:err_fw_cb]))
# 0.8917473490762234

#### correlations


scatter(df[rows_both_sol,:err_scip] ./ df[rows_both_sol,:ne] , df[rows_both_sol,:err_fw_cb] ./ df[rows_both_sol,:ne])
scatter(df[rows_both_sol,:err_scip_rob] ./ df[rows_both_sol,:ne] , df[rows_both_sol,:err_fw_cb] ./ df[rows_both_sol,:ne])

scatter(df[rows_both_sol,:err_fw], df[rows_both_sol,:err_fw_cb])
scatter(df[rows_both_sol,:err_fw], df[rows_both_sol,:err_scip_rob])
plot!(1:25, 1:25)


scatter(df[rows_both_sol,:err_scip], df[rows_both_sol,:err_scip_rob])
plot!(1:25, 1:25)


scatter(df[:, :err_fw], df[:,:npaths_fw])
scatter(df[:, :err_fw_cb], df[:,:npaths_fw_cb])


scatter(df[:, :err_scip_rob], df[:,:npaths_scip_rob])
scatter(df[:, :err_scip], df[:,:npaths_scip])

scatter(df[:,:err0] ./ df[:,:ne], df[:,:err_fw_cb]  ./ df[:,:ne])

scatter(df[:,:err1] ./ df[:,:ne], df[:,:err_fw_cb]  ./ df[:,:ne], xaxis=:log)
scatter(df[rows_scip_sol,:err1] ./ df[rows_scip_sol,:ne], df[rows_scip_sol,:err_scip]  ./ df[rows_scip_sol,:ne], xaxis=:log)

scatter(df[rows_scip_rob_sol,:err1] ./ df[rows_scip_rob_sol,:k], df[rows_scip_rob_sol,:err_fw]  ./ df[rows_scip_rob_sol,:k])


scatter(df[:,:err1], df[:,:err_fw_cb])
scatter(df[rows_both_sol,:err_fw], df[rows_both_sol,:err_fw_cb])

boxplot(ones(length(rows_both_sol)), df[rows_both_sol,:err_scip] ./ df[rows_both_sol,:ne], legend=nothing)
boxplot!(1 .+ ones(length(rows_both_sol)), df[rows_both_sol,:err_scip_rob] ./ df[rows_both_sol,:ne])
boxplot!(2 .+ ones(length(rows_both_sol)), df[rows_both_sol,:err_fw] ./ df[rows_both_sol,:ne])
boxplot!(3 .+ ones(length(rows_both_sol)), df[rows_both_sol,:err_fw_cb] ./ df[rows_both_sol,:ne])
plot!([], [])
plot!([], [])
xticks!(1:4, ["IP", "IP-R", "FW", "FW-C"])

boxplot(ones(length(rows_both_sol)), df[rows_both_sol,:err_scip] ./ df[rows_both_sol,:k], legend=nothing)
boxplot!(1 .+ ones(length(rows_both_sol)), df[rows_both_sol,:err_scip_rob] ./ df[rows_both_sol,:k])
boxplot!(2 .+ ones(length(rows_both_sol)), df[rows_both_sol,:err_fw] ./ df[rows_both_sol,:k])
boxplot!(3 .+ ones(length(rows_both_sol)), df[rows_both_sol,:err_fw_cb] ./ df[rows_both_sol,:k])
xticks!(1:4, ["IP", "IP-R", "FW", "FW-C"])



@show cor(df[rows_both_sol,:err_scip] ./ df[rows_both_sol,:ne], df[rows_both_sol,:err1] ./ df[rows_both_sol,:ne])
@show cor(df[rows_both_sol,:err_scip_rob] ./ df[rows_both_sol,:ne], df[rows_both_sol,:err1] ./ df[rows_both_sol,:ne])
@show cor(df[rows_both_sol,:err_fw] ./ df[rows_both_sol,:ne], df[rows_both_sol,:err1] ./ df[rows_both_sol,:ne])
@show cor(df[rows_both_sol,:err_fw_cb] ./ df[rows_both_sol,:ne], df[rows_both_sol,:err1] ./ df[rows_both_sol,:ne])

@show cor(df[rows_both_sol,:err_scip] ./ df[rows_both_sol,:ne], log.(df[rows_both_sol,:err0] ./ df[rows_both_sol,:ne]))
@show cor(df[rows_both_sol,:err_scip_rob] ./ df[rows_both_sol,:ne], log.(df[rows_both_sol,:err0] ./ df[rows_both_sol,:ne]))
@show cor(df[rows_both_sol,:err_fw] ./ df[rows_both_sol,:ne], log.(df[rows_both_sol,:err0] ./ df[rows_both_sol,:ne]))
@show cor(df[rows_both_sol,:err_fw_cb] ./ df[rows_both_sol,:ne], log.(df[rows_both_sol,:err0] ./ df[rows_both_sol,:ne]))

non_timelimit_scip = filter!(<(1800), sort(df[:,:time_scip]))
non_timelimit_scip_rob = filter!(<(1800), sort(df[:,:time_scip_rob]))
nruns = size(df, 1)

found_in_tl_scip = filter!(<(1800), sort(df[:,:time_scip_sol]))
found_in_tl_scip_rob = filter!(<(1800), sort(df[:,:time_scip_rob_sol]))


plot(sort(non_timelimit_scip), eachindex(non_timelimit_scip), label="IP", xaxis=:log, legend=:bottomright, lw=1.5, st=:samplemarkers)
plot!(found_in_tl_scip, eachindex(found_in_tl_scip), label="S-IP", lw=1.5, st=:samplemarkers, style=:dash)
plot!(sort(non_timelimit_scip_rob), eachindex(non_timelimit_scip_rob), label="IP-R", lw=1.5, st=:samplemarkers)
plot!(found_in_tl_scip_rob, eachindex(found_in_tl_scip_rob), label="S-IP-R", lw=1.5, st=:samplemarkers, style=:dash)
plot!(sort(df[:,:time_fw]), 1:nruns, label="FW", lw=1.5, color=:green, st=:samplemarkers)
plot!(sort(df[:,:time_fw_cb]), 1:nruns, label="FW-C", lw=1.5, st=:samplemarkers)
plot!(sort(df[rows_optimal_poisson,:time_fw_poisson]), 1:sum(rows_optimal_poisson), label="FW-P", lw=1.5, st=:samplemarkers, style=:dashdot)
yaxis!("# instances")
xaxis!("Time (s)")
xlims!(1e-5, 1850)
xticks!([1e-5, 1e-3, 1e-1, 1e1, 1e3])
title!("\$\\textrm{Solving\\ time\\  distribution\\  on\\  inexact\\  instances}\$")
savefig("time_inexact_instances.pdf")
