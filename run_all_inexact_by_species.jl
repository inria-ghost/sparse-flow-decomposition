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

# count_statistics = map(all_species) do species
#     file_str_inexact = open("imperfect_flow_dataset/data_DAG/$(species).graph") do f
#         readlines(f)
#     end
#     glist = build_graph_list_single_flow(file_str_inexact)[1]
#     nontrivials = count(g -> (ne(g) - nv(g) + 2) > 1, glist)
#     (species, length(glist), nontrivials)
# end

count_statistics = (("human", 11783, 5311), ("zebrafish", 15664, 4484), ("salmon", 40870, 14858), ("mouse", 13122, 4708))
total_stats = sum(getindex.(count_statistics, 2)) # 81439
total_nontrivials = sum(getindex.(count_statistics, 3)) # 29361

# for idx in (1:4)
#     r1 = (getindex.(count_statistics, 3) ./ getindex.(count_statistics, 2))[idx]
#     r2 = getindex.(count_statistics, 3)[idx]
#     @printf("%i (%.2f) & ", r2, r1)
# end

# for idx in (1:4)
    # r1 = (getindex.(count_statistics, 3) ./ getindex.(count_statistics, 2))[idx]
    # r2 = getindex.(count_statistics, 3)[idx]
    # @printf("%i & ", r2)
# end
# for idx in (1:4)
    # r1 = (getindex.(count_statistics, 3) ./ getindex.(count_statistics, 2))[idx]
    # r2 = getindex.(count_statistics, 3)[idx]
    # @printf("%.2f & ", r1)
# end

const MAX_TIME = 1800

@assert file_str_inexact[1][1] == '#'
@assert file_str_bounded[1][1] == '#'

const graph_list_inexact, flow_values_list = build_graph_list_single_flow(file_str_inexact)
const graph_list_interval, edge_bound_list = build_graph_list(file_str_bounded)
const path_ground_truth = build_ground_truth_list(file_str_ground_truth, graph_list_inexact)

@assert length(path_ground_truth) == length(graph_list_interval)
@assert all(graph_list_inexact .== graph_list_interval)

### precompilation
let
    g_idx = 50
    g = graph_list_inexact[g_idx]
    flow_bounds = edge_bound_list[g_idx]
    flow_value = flow_values_list[g_idx]
    flow_vector = [flow_value[Pair(e)] for e in edges(g)]
    paths = path_ground_truth[g_idx]
    k = (ne(g) - nv(g) + 2)

    s_idx = 1
    t_idx = nv(g)

    loss, grad! = build_function_gradient(flow_vector)
    lmo = CO.ShortestPathLMO(g, s_idx, t_idx)
    result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true)
    result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true, callback=build_earlystopping_callback(flow_vector))

    m_rob = build_robust_optimization_model(g, flow_value)
    optimize!(m_rob)
    m = build_optimization_model(g, flow_bounds)
    optimize!(m)
    path_weights_fw = integer_weights_from_sol(result.x, flow_vector, result.active_set)
    path_weights_fw_cb = integer_weights_from_sol(result_cb.x, flow_vector, result.active_set)
    fw_recovery_err = count(get(paths, result.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result.active_set))
    fw_cb_recovery_err = count(get(paths, result_cb.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result_cb.active_set))

    paths_scip = [BitVector(JuMP.value.(m[:x][:,:,i])[Tuple(e)] for e in edges(g)) for i in 1:k]
    weights_scip = round.(Int, JuMP.value.(m[:w]))
    scip_recovery_err = count(get(paths, paths_scip[idx], 0) != weights_scip[idx] for idx in eachindex(weights_scip))

    paths_scip_rob = [BitVector(JuMP.value.(m_rob[:x][:,:,i])[Tuple(e)] for e in edges(g)) for i in 1:k]
    weights_scip_rob = round.(Int, JuMP.value.(m_rob[:w]))
    scip_rob_recovery_err = count(get(paths, paths_scip_rob[idx], 0) != weights_scip_rob[idx] for idx in eachindex(weights_scip_rob))
end

### end of precompilation

for (idx, (g, flow_bounds, flow_value, paths)) in enumerate(zip(graph_list_interval, edge_bound_list, flow_values_list, path_ground_truth))
    result_file = "results_inexact/result_$(species)_$(idx).json"
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
    m = build_optimization_model(g, flow_bounds)
    set_time_limit_sec(m, MAX_TIME)
    set_silent(m)
    optimize!(m)
    time_scip = solve_time(m)
    st = termination_status(m)

    flow_truth = sum(p * w for (p, w) in paths)
    if primal_status(m) == FEASIBLE_POINT
        scip_best_solution_weight = round.(Int, JuMP.value.(m[:w]))
        paths_scip = [BitVector(round.(Int, JuMP.value.(m[:x][:,:,i])[Tuple(e)]) for e in edges(g)) for i in 1:k]
        scip_recovery_err = count(get(paths, paths_scip[idx], 0) != scip_best_solution_weight[idx] for idx in eachindex(scip_best_solution_weight))
        scip_err_flow = norm(flow_truth - sum(scip_best_solution_weight[idx] * paths_scip[idx] for idx in eachindex(paths_scip)))
        sol = SCIP.LibSCIP.SCIPgetBestSol(m.moi_backend)
        time_scip_sol = SCIP.LibSCIP.SCIPsolGetTime(sol)
    else
        scip_best_solution_weight = ones(ne(g))
        paths_scip = Float64[]
        scip_recovery_err = sum(values(paths))
        scip_err_flow = norm(flow_truth)
        time_scip_sol = MAX_TIME
    end

    loss, grad! = build_function_gradient(flow_vector)
    lmo = CO.ShortestPathLMO(g, s_idx, t_idx)
    result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, line_search=FrankWolfe.Secant())
    time_fw = result.traj_data[end][end]
    path_weights_fw = integer_weights_from_sol(result.x, flow_vector, result.active_set)
    result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, callback=build_earlystopping_callback(flow_vector), line_search=FrankWolfe.Secant())
    time_fw_cb = result_cb.traj_data[end][end]
    path_weights_fw_cb = integer_weights_from_sol(result_cb.x, flow_vector, result_cb.active_set)

    fw_recovery_err = count(get(paths, result.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result.active_set))
    fw_cb_recovery_err = count(get(paths, result_cb.active_set.atoms[idx], 0) != path_weights_fw_cb[idx] for idx in eachindex(result_cb.active_set))

    fw_err_flow = norm(flow_truth - sum(path_weights_fw[idx] * result.active_set.atoms[idx] for idx in eachindex(path_weights_fw)))
    fw_cb_err_flow = norm(flow_truth - sum(path_weights_fw_cb[idx] * result_cb.active_set.atoms[idx] for idx in eachindex(path_weights_fw_cb)))

    m_rob = build_robust_optimization_model(g, flow_value)
    set_time_limit_sec(m_rob, MAX_TIME)
    set_silent(m_rob)
    optimize!(m_rob)

    time_scip_rob = solve_time(m_rob)
    st_rob = termination_status(m_rob)

    if primal_status(m_rob) == FEASIBLE_POINT
        scip_rob_best_solution_weight = round.(Int, JuMP.value.(m_rob[:w]))

        sol_rob = SCIP.LibSCIP.SCIPgetBestSol(m_rob.moi_backend)
        time_scip_rob_sol = SCIP.LibSCIP.SCIPsolGetTime(sol_rob)

        paths_scip_rob = [BitVector(round.(Int, JuMP.value.(m_rob[:x][:,:,i])[Tuple(e)]) for e in edges(g)) for i in 1:k]
        scip_rob_recovery_err = count(get(paths, paths_scip_rob[idx], 0) != scip_rob_best_solution_weight[idx] for idx in eachindex(scip_rob_best_solution_weight))
        scip_rob_err_flow = norm(flow_truth - sum(scip_rob_best_solution_weight[idx] * paths_scip_rob[idx] for idx in eachindex(paths_scip_rob)))
    else
        scip_rob_best_solution_weight = ones(ne(g))
        time_scip_rob_sol = MAX_TIME
        scip_rob_err_flow = norm(flow_truth)
        paths_scip_rob = Float64[]
        scip_rob_recovery_err = sum(values(paths))
    end


    open(result_file, "w") do f
        json_str = JSON.json(
            (
                time_fw = time_fw,
                time_fw_cb = time_fw_cb,
                time_scip = time_scip,
                status_scip = string(st),
                time_scip_sol = time_scip_sol,
                npaths_scip = Int(norm(scip_best_solution_weight, 0)),
                npaths_scip_rob = Int(norm(scip_rob_best_solution_weight, 0)),
                npaths_fw = Int(norm(path_weights_fw, 0)),
                npaths_fw_cb = Int(norm(path_weights_fw_cb, 0)),
                err_fw = fw_recovery_err,
                err_fw_cb = fw_cb_recovery_err,
                err_scip = scip_recovery_err,
                err_scip_rob = scip_rob_recovery_err,
                err_flow_fw = fw_err_flow,
                err_flow_fw_cb = fw_cb_err_flow,
                err_flow_scip = scip_err_flow,
                err_flow_scip_rob = scip_rob_err_flow,
                time_scip_rob_sol = time_scip_rob_sol,
                time_scip_rob = time_scip_rob,
                species=species,
            )
        )
        write(f, json_str)
    end
end
