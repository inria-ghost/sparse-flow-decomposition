using JSON
using CSV
using DataFrames
using Plots
using Printf
using Statistics
using StatsPlots

include("functions.jl")

all_res_files = readdir("result_errors/", join=true)

species = "salmon"

graph_stats = JSON.parsefile("graph_inexact_stats.json")

data_vec = []
for resfile in all_res_files
    f_str = open(resfile) do f
        read(f, String)
    end
    if length(f_str) < 2
        @info "empty file $resfile"
        continue
    end
    res = JSON.parse(f_str)
    file_idx = parse(Int, split(split(resfile, "_")[end], ".")[1])
    res["idx"] = file_idx
    for key_name in ("nv", "ne", "k", "err0", "err1")
        res[key_name] = graph_stats[species][string(file_idx)][key_name]
    end
    push!(data_vec, res)
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

rows_scip_rob_sol = Dict(
    "binomial" => df[:,:time_scip_rob_sol_binomial] .< 1800,
    "poisson"  => df[:,:time_scip_rob_sol_poisson] .< 1800,
)


## POISSON
# count(df[:, :err_flow_fw_poisson] .== minimum_error_flow)
# count(df[:, :err_flow_fw_poisson_poisson] .== minimum_error_flow)
# count(df[:, :err_flow_scip_rob_poisson] .== minimum_error_flow)


# minimum_error_path = 

# count(df[:, :err_fw_poisson] .== minimum_error_path)
# count(df[:, :err_fw_poisson_poisson] .== minimum_error_path)
# count(df[:, :err_scip_rob_poisson] .== minimum_error_path)

# ## Binomial
# minimum_error_flow = [min(df[ridx, :err_flow_fw_binomial], df[ridx, :err_flow_fw_poisson_binomial], df[ridx, :err_flow_scip_rob_binomial]) for ridx in 1:size(df, 1)]

# count(df[:, :err_flow_fw_binomial] .== minimum_error_flow)
# count(df[:, :err_flow_fw_poisson_binomial] .== minimum_error_flow)
# count(df[:, :err_flow_scip_rob_binomial] .== minimum_error_flow)


# minimum_error_path = [min(df[ridx, :err_fw_binomial], df[ridx, :err_fw_poisson_binomial], df[ridx, :err_scip_rob_binomial]) for ridx in 1:size(df, 1)]

# count(df[:, :err_fw_binomial] .== minimum_error_path)
# count(df[:, :err_fw_poisson_binomial] .== minimum_error_path)
# count(df[:, :err_scip_rob_binomial] .== minimum_error_path)

minimum_error_path = Dict(
    "binomial" => [min(df[ridx, :err_fw_binomial], df[ridx, :err_fw_poisson_binomial], df[ridx, :err_scip_rob_binomial]) for ridx in 1:size(df, 1)],
    "poisson" => [min(df[ridx, :err_fw_poisson], df[ridx, :err_fw_poisson_poisson], df[ridx, :err_scip_rob_poisson]) for ridx in 1:size(df, 1)],
)
minimum_error_flow = Dict(
    "binomial" => [min(df[ridx, :err_flow_fw_binomial], df[ridx, :err_flow_fw_poisson_binomial], df[ridx, :err_flow_scip_rob_binomial]) for ridx in 1:size(df, 1)],
    "poisson" => [min(df[ridx, :err_flow_fw_poisson], df[ridx, :err_flow_fw_poisson_poisson], df[ridx, :err_flow_scip_rob_poisson]) for ridx in 1:size(df, 1)],
)

minimum_npath = Dict(
    "binomial" => [min(df[ridx, :npaths_fw_binomial], df[ridx, :npaths_fw_poisson_binomial], df[ridx, :npaths_scip_rob_binomial]) for ridx in 1:size(df, 1)],
    "poisson" => [min(df[ridx, :npaths_fw_poisson], df[ridx, :npaths_fw_poisson_poisson], df[ridx, :npaths_scip_rob_poisson]) for ridx in 1:size(df, 1)],
)


for distribution in ("binomial", "poisson")
    @info(distribution)
    println()
    print("Path error & ")
    for col in (:err_fw, :err_fw_poisson, :err_scip_rob)
        vals = df[rows_scip_rob_sol[distribution], "$(col)_$(distribution)"]
        @printf(" %.2f/%.2f/%i ", mean(vals), geom_shifted_mean(vals), count(df[:,"$(col)_$(distribution)"] .== minimum_error_path[distribution]))
        print(" & ")
    end
    println()
    print("Flow error & ")
    for col in (:err_flow_fw, :err_flow_fw_poisson, :err_flow_scip_rob)
        vals = df[rows_scip_rob_sol[distribution], "$(col)_$(distribution)"]
        @printf(" %.2f/%.2f/%i ", mean(vals), geom_shifted_mean(vals), count(df[:,"$(col)_$(distribution)"] .== minimum_error_flow[distribution]))
        print(" & ")
    end
    println()
    print("Flow rel.~err. & ")
    for col in (:err_flow_fw, :err_flow_fw_poisson, :err_flow_scip_rob)
        vals = df[rows_scip_rob_sol[distribution], "$(col)_$(distribution)"] ./ df[rows_scip_rob_sol[distribution], :ne]
        @printf(" %.2f/%.2f ", mean(vals), geom_shifted_mean(vals))
        print(" & ")
    end
    println()
    print("\\# paths & ")
    for col in (:npaths_fw, :npaths_fw_poisson, :npaths_scip_rob)
        vals = df[rows_scip_rob_sol[distribution], "$(col)_$(distribution)"]
        @printf(" %.2f/%.2f/%i ", mean(vals), geom_shifted_mean(vals), count(df[:,"$(col)_$(distribution)"] .== minimum_npath[distribution]))
        print(" & ")
    end
    println()
    print("Time (s) & ")
    for col in (:time_fw, :time_fw_poisson, :time_scip_rob)
        vals = df[rows_scip_rob_sol[distribution], "$(col)_$(distribution)"]
        @printf(" %.2e ", geom_shifted_mean(vals))
        print(" & ")
    end
    println()
end
