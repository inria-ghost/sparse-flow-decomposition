using JSON
using Statistics
using Distributions
using Random

include("functions.jl")

species = "salmon"

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
@assert file_str_bounded[1][1] == '#'

const graph_list_inexact, flow_values_list = build_graph_list_single_flow(file_str_inexact)
const path_ground_truth = build_ground_truth_list(file_str_ground_truth, graph_list_inexact)

large_graph_indices = filter(eachindex(graph_list_inexact)) do idx
    g = graph_list_inexact[idx]
    k = (ne(g) - nv(g) + 2)
    ne(g) > 80 && k > 15
end


for idx in large_graph_indices
    resfile = "result_errors/result_error_$idx.json"
    sleep(5 * rand())
    if isfile(resfile)
        continue
    end
    println("Graph $idx")
    flush(stdout)
    touch(resfile)
    g = graph_list_inexact[idx]
    k = (ne(g) - nv(g) + 2)
    flow_value = flow_values_list[idx]
    paths = path_ground_truth[idx]
    flow_truth = sum(p * w for (p,w) in paths)
    flow_vector = [flow_value[Pair(e)] for e in edges(g)]
    Random.seed!(idx)
    errored_flow_poisson = [rand(Poisson(f)) for f in flow_truth]
    flow_value_poisson = Dict(Pair(e) => errored_flow_poisson[idx] for (idx, e) in enumerate(edges(g)))
    s_idx = 1
    t_idx = nv(g)
    lmo = CO.ShortestPathLMO(g, s_idx, t_idx)

    loss, grad! = build_function_gradient(errored_flow_poisson)
    result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true)
    result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, callback=build_earlystopping_callback(errored_flow_poisson))
    time_fw = result.traj_data[end][end]
    time_fw_cb = result_cb.traj_data[end][end]

    lmo_poisson = OriginScaledShortestPathPolytope(g, s_idx, t_idx, big(maximum(errored_flow_poisson)))
    loss_poisson, grad_poisson! = build_function_gradient_poisson(errored_flow_poisson, g)
    result_poisson = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson, FrankWolfe.compute_extreme_point(lmo_poisson, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2), max_iteration=5000)
    time_fw_poisson = result_poisson.traj_data[end][end]

    path_weights_fw, flow_fw = integer_weights_and_flow_from_sol(result.x, errored_flow_poisson, result.active_set)
    path_weights_fw_cb, flow_fw_cb = integer_weights_and_flow_from_sol(result_cb.x, errored_flow_poisson, result_cb.active_set)
    _, flow_fw_poisson = integer_weights_and_flow_from_sol(result_poisson.x, errored_flow_poisson, result_poisson.active_set)

    path_weights_fw_poisson = round.(Int, result_poisson.active_set.weights * maximum(errored_flow_poisson))
    rounded_poisson_sol = round.(Int, result_poisson.x)
    fw_recovery_err = count(get(paths, result.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result.active_set))
    fw_cb_recovery_err = count(get(paths, result_cb.active_set.atoms[idx], 0) != path_weights_fw_cb[idx] for idx in eachindex(result_cb.active_set))
    fw_poisson_recovery_err = count(get(paths, result_poisson.active_set.atoms[idx], 0) != path_weights_fw_poisson[idx] for idx in eachindex(result_poisson.active_set))

    fw_err_flow = norm(sum(p * w for (p, w) in paths) - sum(path_weights_fw[idx] * result.active_set.atoms[idx] for idx in eachindex(path_weights_fw)))
    fw_cb_err_flow = norm(sum(p * w for (p, w) in paths) - sum(path_weights_fw_cb[idx] * result_cb.active_set.atoms[idx] for idx in eachindex(path_weights_fw_cb)))
    fw_poisson_err_flow = norm(sum(p * w for (p, w) in paths) - rounded_poisson_sol)

    m_rob = build_robust_optimization_model(g, flow_value_poisson)
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

    res_poisson_error = (
        time_fw = time_fw,
        time_fw_cb = time_fw_cb,
        time_fw_poisson = time_fw_poisson,
        time_scip_rob = time_scip_rob,
        time_scip_rob_sol = time_scip_rob_sol,
        npaths_fw = Int(norm(path_weights_fw, 0)),
        npaths_fw_cb = Int(norm(path_weights_fw_cb, 0)),
        npaths_fw_poisson = Int(norm(path_weights_fw_poisson, 0)),
        npaths_scip_rob = Int(norm(scip_rob_best_solution_weight, 0)),
        err_fw = fw_recovery_err,
        err_fw_cb = fw_cb_recovery_err,
        err_fw_poisson = fw_poisson_recovery_err,
        err_scip_rob = scip_rob_recovery_err,
        err_flow_fw = fw_err_flow,
        err_flow_fw_cb = fw_cb_err_flow,
        err_flow_fw_poisson = fw_poisson_err_flow,
        err_flow_scip_rob = scip_rob_err_flow,
        species=species,
    )
    Random.seed!(idx)
    errored_flow_binomial = [rand(Binomial(2f, 0.5)) for f in flow_truth]
    flow_value_binomial = Dict(Pair(e) => errored_flow_binomial[idx] for (idx, e) in enumerate(edges(g)))

    loss, grad! = build_function_gradient(errored_flow_binomial)
    result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=false, timeout=MAX_TIME, trajectory=true)
    result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true, callback=build_earlystopping_callback(errored_flow_binomial))
    time_fw = result.traj_data[end][end]
    time_fw_cb = result_cb.traj_data[end][end]

    lmo_poisson = OriginScaledShortestPathPolytope(g, s_idx, t_idx, big(maximum(errored_flow_poisson)))
    loss_poisson, grad_poisson! = build_function_gradient_poisson(errored_flow_binomial, g)
    result_poisson = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson, FrankWolfe.compute_extreme_point(lmo_poisson, ones(ne(g))), verbose=true, timeout=MAX_TIME, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2), max_iteration=5000)
    time_fw_poisson = result_poisson.traj_data[end][end]

    path_weights_fw, flow_fw = integer_weights_and_flow_from_sol(result.x, errored_flow_binomial, result.active_set)
    path_weights_fw_cb, flow_fw_cb = integer_weights_and_flow_from_sol(result_cb.x, errored_flow_binomial, result_cb.active_set)
    _, flow_fw_poisson = integer_weights_and_flow_from_sol(result_poisson.x, errored_flow_binomial, result_poisson.active_set)

    path_weights_fw_poisson = round.(Int, result_poisson.active_set.weights * maximum(errored_flow_binomial))
    rounded_poisson_sol = round.(Int, result_poisson.x)
    fw_recovery_err = count(get(paths, result.active_set.atoms[idx], 0) != path_weights_fw[idx] for idx in eachindex(result.active_set))
    fw_cb_recovery_err = count(get(paths, result_cb.active_set.atoms[idx], 0) != path_weights_fw_cb[idx] for idx in eachindex(result_cb.active_set))
    fw_poisson_recovery_err = count(get(paths, result_poisson.active_set.atoms[idx], 0) != path_weights_fw_poisson[idx] for idx in eachindex(result_poisson.active_set))

    fw_err_flow = norm(sum(p * w for (p, w) in paths) - sum(path_weights_fw[idx] * result.active_set.atoms[idx] for idx in eachindex(path_weights_fw)))
    fw_cb_err_flow = norm(sum(p * w for (p, w) in paths) - sum(path_weights_fw_cb[idx] * result_cb.active_set.atoms[idx] for idx in eachindex(path_weights_fw_cb)))
    fw_poisson_err_flow = norm(sum(p * w for (p, w) in paths) - rounded_poisson_sol)

    m_rob = build_robust_optimization_model(g, flow_value_binomial)
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

    res_binomial_error = (
        time_fw = time_fw,
        time_fw_cb = time_fw_cb,
        time_fw_poisson = time_fw_poisson,
        time_scip_rob = time_scip_rob,
        time_scip_rob_sol = time_scip_rob_sol,
        npaths_fw = Int(norm(path_weights_fw, 0)),
        npaths_fw_cb = Int(norm(path_weights_fw_cb, 0)),
        npaths_fw_poisson = Int(norm(path_weights_fw_poisson, 0)),
        npaths_scip_rob = Int(norm(scip_rob_best_solution_weight, 0)),
        err_fw = fw_recovery_err,
        err_fw_cb = fw_cb_recovery_err,
        err_fw_poisson = fw_poisson_recovery_err,
        err_scip_rob = scip_rob_recovery_err,
        err_flow_fw = fw_err_flow,
        err_flow_fw_cb = fw_cb_err_flow,
        err_flow_fw_poisson = fw_poisson_err_flow,
        err_flow_scip_rob = scip_rob_err_flow,
        species=species,
    )

    res_final = Dict("species" => species, "idx" => idx)
    for (k, v) in pairs(res_poisson_error)
        res_final[string(k) * "_poisson"] = v
    end
    for (k, v) in pairs(res_binomial_error)
        res_final[string(k) * "_binomial"] = v
    end
    open(resfile, "w") do f
        write(f, JSON.json(res_final))
    end
end
