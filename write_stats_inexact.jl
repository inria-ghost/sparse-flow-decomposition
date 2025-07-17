using JSON
using Statistics

include("functions.jl")

species = isempty(ARGS) ? "human" : ARGS[end]

all_species = ("human", "mouse", "salmon", "zebrafish")
res = Dict(species => Dict() for species in all_species)

for species in all_species
    file_str_inexact = open("imperfect_flow_dataset/data_DAG/$(species).graph") do f
        readlines(f)
    end
    file_str_ground_truth = open("imperfect_flow_dataset/data_GT/$(species).truth") do f
        readlines(f)
    end
    graph_list_inexact, flow_values_list = build_graph_list_single_flow(file_str_inexact)
    path_ground_truth = build_ground_truth_list(file_str_ground_truth, graph_list_inexact)
    for (idx, g) in enumerate(graph_list_inexact)
        edge_vec = Pair.(collect(edges(g)))
        flow_dict = Dict(Pair(e) => 0 for e in edges(g))
        flow_dict_err = flow_values_list[idx]
        for (path_vector, weight) in paths_values
            for idx in eachindex(path_vector)
                if path_vector[idx] == 1
                    flow_dict[edge_vec[idx]] += weight
                end
            end
        end
        err0 = count(flow_dict[Pair(e)] != flow_dict_err[Pair(e)] for e in edges(g))
        err1 = sum(abs(flow_dict[Pair(e)] - flow_dict_err[Pair(e)]) for e in edges(g))
        res[species][idx] = Dict("ne" => ne(g), "nv" => nv(g), "k" => ne(g) - nv(g) + 2, "err0" => err0, "err1" => err1)
    end
end

open("graph_inexact_stats.json", "w") do f
    write(f, JSON.json(res))
end
