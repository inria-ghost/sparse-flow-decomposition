using JSON
using CSV
using DataFrames
using Plots
using Printf
using Statistics
using StatsPlots

all_fw_res_files = filter(s -> occursin("_fw", s), readdir("results_inexact/", join=true))

# all_species = ("human", "zebrafish", "salmon", "mouse")
all_species = ("salmon",)

graph_stats = JSON.parsefile("graph_inexact_stats.json")

data_vec = []
for resfile in all_fw_res_files
    for species in all_species
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
        fw_res = JSON.parsefile("results_inexact/result_$(species)_$(file_idx)_fw.json")

        res["idx"] = file_idx
        for key_name in ("nv", "ne", "k", "err0", "err1")
            res[key_name] = graph_stats[species][string(file_idx)][key_name]
        end
        for key_name in ("npaths_fw_poisson", "err_fw_poisson", "err_fw_poisson")
            res[key_name] = fw_res[key_name]
        end
        if res["err_fw"] != fw_res["err_fw"]
            @show (res["err_fw"], fw_res["err_fw"])
            @show resfile
        end
        push!(data_vec, res)
    end
end

df = DataFrame(data_vec)
