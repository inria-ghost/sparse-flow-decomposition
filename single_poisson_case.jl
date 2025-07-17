using JSON
using Statistics
using Distributions
using Random

include("functions.jl")
include(joinpath(pathof(FrankWolfe), "../../examples/plot_utils.jl"))

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

idx = first(large_graph_indices)

g = graph_list_inexact[idx]
flow_value = flow_values_list[idx]
paths = path_ground_truth[idx]
flow_truth = sum(p * w for (p,w) in paths)
flow_vector = [flow_value[Pair(e)] for e in edges(g)]

Random.seed!(4)
errored_flow_independent = [rand(Poisson(f)) for f in flow_truth]

k = (ne(g) - nv(g) + 2)

s_idx = 1
t_idx = nv(g)

loss, grad! = build_function_gradient(errored_flow_independent)
loss_poisson0, grad_poisson0! = build_function_gradient_poisson(errored_flow_independent, g)
lmo_poisson = OriginScaledShortestPathPolytope(g, s_idx, t_idx, big(maximum(errored_flow_independent)))

result_poisson0 = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson0, grad_poisson0!, lmo_poisson, FrankWolfe.compute_extreme_point(lmo_poisson, ones(ne(g))), verbose=true, timeout=60, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2), print_iter=1)

offset = result_poisson0.primal

loss_poisson, grad_poisson! = build_function_gradient_poisson(errored_flow_independent, g, offset=-offset)

lmo = CO.ShortestPathLMO(g, s_idx, t_idx)
result = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true)
result_cb = FrankWolfe.blended_pairwise_conditional_gradient(loss, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, ones(ne(g))), verbose=true, timeout=60, trajectory=true, callback=build_earlystopping_callback(errored_flow_independent))

result_poisson = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson, FrankWolfe.compute_extreme_point(lmo_poisson, ones(ne(g))), verbose=true, timeout=60, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2))

lmo_poisson_naive = OriginScaledShortestPathPolytope(g, s_idx, t_idx, maximum(errored_flow_independent))
result_poisson_naive = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson_naive, FrankWolfe.compute_extreme_point(lmo_poisson_naive, ones(ne(g))), verbose=true, timeout=120, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2), print_iter=100, max_iteration=10000)

# result_poisson2 = FrankWolfe.blended_pairwise_conditional_gradient(loss_poisson, grad_poisson!, lmo_poisson_naive, FrankWolfe.compute_extreme_point(lmo_poisson, ones(ne(g))), verbose=true, timeout=60, trajectory=true, line_search=FrankWolfe.AdaptiveZerothOrder(big(0.9), 2), print_iter=100)


p = plot_trajectories([result.traj_data, result_poisson.traj_data, result_poisson_naive.traj_data], ["L", "PE", "PF"], marker_shapes=[:+, :o, :star5, :x], line_width=2, plot_title="\$\\textrm{Convergence\\ on\\ salmon-$idx}\$") # plot_title="\$\\textrm{Convergence\\ on\\ the\\ salmon-$idx\\ instance\$")
savefig(p, "salmon-$idx-convergence.pdf")

_, flow_fw = integer_weights_and_flow_from_sol(result.x, errored_flow_independent, result.active_set)
_, flow_fw_cb = integer_weights_and_flow_from_sol(result_cb.x, errored_flow_independent, result_cb.active_set)
_, flow_fw_poisson = integer_weights_and_flow_from_sol(result_poisson.x, errored_flow_independent, result_poisson.active_set)

