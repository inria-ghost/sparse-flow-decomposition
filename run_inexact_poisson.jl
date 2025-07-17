using JSON
using Statistics

include("functions.jl")

species = isempty(ARGS) ? "human" : ARGS[end]

@assert species in ("human", "mouse", "salmon", "zebrafish")

@info species

file_str_inexact = open("imperfect_flow_dataset/data_DAG/$(species).graph") do f
    readlines(f)
end
file_str_bounded = open("imperfect_flow_dataset/data_INEXACT/$(species)_inexact.graph") do f
    readlines(f)
end

file_str_ground_truth = open("imperfect_flow_dataset/data_GT/$(species).truth") do f
    readlines(f)
end

const MAX_TIME = 1800

@assert file_str_inexact[1][1] == '#'

const graph_list_inexact, flow_values_list = build_graph_list_single_flow(file_str_inexact)
const path_ground_truth = build_ground_truth_list(file_str_ground_truth, graph_list_inexact)

@assert length(path_ground_truth) == length(graph_list_inexact)

### precompilation
let
    g_idx = 50
    g = graph_list_inexact[g_idx]
    flow_value = flow_values_list[g_idx]
    flow_vector = [flow_value[Pair(e)] for e in edges(g)]
    paths = path_ground_truth[g_idx]
    k = (ne(g) - nv(g) + 2)

    s_idx = 1
    t_idx = nv(g)

    loss, grad! = build_function_gradient(flow_vector)
    loss_poisson, grad_poisson! = build_function_gradient_poisson(flow_vector, g)

    lmo = CO.ShortestPathLMO(g, s_idx, t_idx)
    result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true)
    result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true, callback=build_earlystopping_callback(flow_vector))

    lmo_poisson = OriginScaledShortestPathPolytope(g, s_idx, t_idx, maximum(flow_vector))
    result_poisson = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson, FrankWolfe.compute_extreme_point(lmo_poisson, ones(ne(g))), verbose=true, timeout=60, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2))

    path_weights_fw = integer_weights_from_sol(result.x, flow_vector, result.active_set)
    path_weights_fw_cb = integer_weights_from_sol(result_cb.x, flow_vector, result.active_set)
    fw_recovery_err = count(get(paths, result.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result.active_set))
    fw_cb_recovery_err = count(get(paths, result_cb.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result_cb.active_set))

end

### end of precompilation

for (idx, (g, flow_value, paths)) in enumerate(zip(graph_list_inexact, flow_values_list, path_ground_truth))
    result_file = "results_inexact/result_$(species)_$(idx)_fw.json"
    s_idx = 1
    t_idx = nv(g)
    k = (ne(g) - nv(g) + 2)
    if k == 1
        continue
    end
    sleep(rand(1:5))
    if isfile(result_file)
        continue
    end
    @info "Graph $idx"
    touch(result_file)
    flow_vector = [flow_value[Pair(e)] for e in edges(g)]

    loss, grad! = build_function_gradient(flow_vector)
    loss_poisson, grad_poisson! = build_function_gradient_poisson(flow_vector, g)
    lmo = CO.ShortestPathLMO(g, s_idx, t_idx)
    lmo_poisson = OriginScaledShortestPathPolytope(g, s_idx, t_idx, big(maximum(flow_vector)))

    result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, line_search=FrankWolfe.Secant())
    time_fw = result.traj_data[end][end]
    path_weights_fw = integer_weights_from_sol(result.x, flow_vector, result.active_set)
    result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, callback=build_earlystopping_callback(flow_vector), line_search=FrankWolfe.Secant())
    time_fw_cb = result_cb.traj_data[end][end]
    path_weights_fw_cb = integer_weights_from_sol(result_cb.x, flow_vector, result_cb.active_set)

    result_poisson = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson, FrankWolfe.compute_extreme_point(lmo_poisson, -ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, line_search=line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2), max_iteration=5000)
    time_fw_poisson = result_poisson.traj_data[end][end]
    
    path_weights_fw_poisson = round.(Int, result_poisson.active_set.weights * maximum(flow_vector))
    rounded_poisson_sol = round.(Int, result_poisson.x)
    fw_recovery_err = count(get(paths, result.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result.active_set))
    fw_cb_recovery_err = count(get(paths, result_cb.active_set.atoms[idx], 0) != path_weights_fw_cb[idx] for idx in eachindex(result_cb.active_set))
    fw_poisson_recovery_err = count(get(paths, result_poisson.active_set.atoms[idx], 0) != path_weights_fw_poisson[idx] for idx in eachindex(result_poisson.active_set))

    fw_err_flow = norm(sum(p * w for (p, w) in paths) - sum(path_weights_fw[idx] * result.active_set.atoms[idx] for idx in eachindex(path_weights_fw)))
    fw_cb_err_flow = norm(sum(p * w for (p, w) in paths) - sum(path_weights_fw_cb[idx] * result_cb.active_set.atoms[idx] for idx in eachindex(path_weights_fw_cb)))
    fw_poisson_err_flow = norm(sum(p * w for (p, w) in paths) - rounded_poisson_sol)

    open(result_file, "w") do f
        json_str = JSON.json(
            (
                time_fw = time_fw,
                time_fw_cb = time_fw_cb,
                time_fw_poisson = time_fw_poisson,
                npaths_fw = Int(norm(path_weights_fw, 0)),
                npaths_fw_cb = Int(norm(path_weights_fw_cb, 0)),
                npaths_fw_poisson = Int(norm(path_weights_fw_poisson, 0)),
                err_fw = fw_recovery_err,
                err_fw_cb = fw_cb_recovery_err,
                err_fw_poisson = fw_poisson_recovery_err,
                err_fw_flow = fw_err_flow,
                err_fw_cb_flow = fw_cb_err_flow,
                err_fw_poisson_flow = fw_poisson_err_flow,
                species=species,
                niter_poisson = length(result_poisson.traj_data),
                dualgap_poisson = result_poisson.dual_gap,    
            )
        )
        write(f, json_str)
    end
end
