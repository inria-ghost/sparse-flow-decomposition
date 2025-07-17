using LinearAlgebra
using FrankWolfe
import CombinatorialLinearOracles as CO
using JuMP
using SCIP
using Graphs
using Statistics
import AccurateArithmetic
using Plots

function build_graph_list(file_str)
    current_line = 2
    graphs = SimpleDiGraph{Int}[]
    edge_bounds = Vector{Dict{Pair{Int, Int}, Tuple{Int,Int}}}()
    while current_line < length(file_str)
        nvertices = parse(Int, file_str[current_line])
        g = DiGraph(nvertices)
        push!(graphs, g)
        edge_dict = Dict{Pair{Int, Int}, Tuple{Int,Int}}()
        push!(edge_bounds, edge_dict)
        nedges = 1
        while length(file_str) >= current_line + nedges && !occursin('#', file_str[current_line + nedges])
            line = split(file_str[current_line + nedges])
            src0, dst0, lowerbound, upperbound = parse.(Int, line)
            add_edge!(g, src0 + 1, dst0 + 1)
            edge_dict[src0 + 1 => dst0 + 1] = (lowerbound, upperbound)
            nedges += 1
        end
        current_line = current_line + nedges + 1
    end
    return graphs, edge_bounds
end

function build_graph_list_single_flow(file_str)
    current_line = 2
    graphs = SimpleDiGraph{Int}[]
    flow_vals = Vector{Dict{Pair{Int, Int}, Float64}}()
    while current_line < length(file_str)
        nvertices = parse(Int, file_str[current_line])
        g = DiGraph(nvertices)
        push!(graphs, g)
        edge_dict = Dict{Pair{Int, Int}, Float64}()
        push!(flow_vals, edge_dict)
        nedges = 1
        while length(file_str) >= current_line + nedges && !occursin('#', file_str[current_line + nedges])
            line = split(file_str[current_line + nedges])
            src0, dst0 = parse.(Int, line[1:2])
            flow_value = parse(Float64, line[end])
            add_edge!(g, src0 + 1, dst0 + 1)
            edge_dict[src0 + 1 => dst0 + 1] = flow_value
            nedges += 1
        end
        current_line = current_line + nedges + 1
    end
    return graphs, flow_vals
end

function build_ground_truth_list(file_str, graph_list)
    current_line = 1
    g_idx = 1
    paths = Vector{Dict{BitVector, Int}}()
    while current_line < length(file_str)
        g = graph_list[g_idx]
        edge_idx_dict = Dict(e => idx for (idx, e) in enumerate(edges(g)))
        npaths = 1
        path_dict = Dict{BitVector,Int}()
        push!(paths, path_dict)
        while length(file_str) >= current_line + npaths && !occursin('#', file_str[current_line + npaths])
            line = split(file_str[current_line + npaths])
            weight = parse(Int, line[1])
            path_vertices = parse.(Int, line[2:end]) .+ 1
            @assert length(path_vertices) <= nv(g) "$current_line"
            @assert path_vertices[1] == 1
            @assert path_vertices[end] == nv(g)
            path_vector = falses(ne(g))
            for idx in 2:length(path_vertices)
                u = path_vertices[idx-1]
                v = path_vertices[idx]
                path_vector[edge_idx_dict[Edge(u, v)]] = 1
            end
            path_dict[path_vector] = weight
            npaths += 1
        end
        current_line = current_line + npaths
        g_idx += 1
    end
    return paths
end

function build_function_gradient(flow; offset=0, normalize=true)
    nf2 = dot(flow, flow)
    ninv = normalize ? inv(length(flow)) : 1
    function loss(p)
        return 1/2 * ninv * (dot(p, p) - dot(flow, p)^2/ nf2) + offset
    end
    function grad!(storage, p)
        dot_p_flow = dot(p, flow)
        storage .= flow
        storage .*= -dot_p_flow / nf2
        storage .+= p
        @. storage = p - dot_p_flow * flow / nf2
        @. storage *= ninv
        return nothing
    end
    return loss, grad!
end

function build_function_gradient_poisson(flow, g, T = Float64; normalize=true, offset=0)
    ninv = normalize ? inv(length(flow)) : 1
    # node index -> set of edges coming into that node
    edge_vec = collect(edges(g))
    neighboring_edge_dict = Dict{Int, Vector{Int}}()
    sum_inflows = zeros(T, nv(g))
    for u in 2:nv(g)
        neighboring_edge_dict[u] = Int[]
        for (idx, e) in enumerate(edge_vec)
            if dst(e) == u
                push!(neighboring_edge_dict[u], idx)
                sum_inflows[u] += flow[idx]
            end
        end
    end

    function poisson_loss(p)
        lin_part = sum(p)
        log_part = -sum(sum_inflows[u] * log(sum(p[e_idx] for e_idx in neighboring_edge_dict[u]) + 1e-2) for u in 2:nv(g) if sum_inflows[u] > 0)
        return ninv * (lin_part + log_part) + offset
    end
    function poisson_grad!(storage, p)
        storage .= ninv
        for u in 2:nv(g)
            for e_idx in neighboring_edge_dict[u]
                if sum_inflows[u] > 0
                    storage[e_idx] -= ninv * sum_inflows[u] * inv(sum(p[idx] for idx in neighboring_edge_dict[u]) + 1e-2)
                end
            end
        end
        return nothing
    end
    return poisson_loss, poisson_grad!
end

function build_function_gradient_scaledpoisson(flow, g; normalize=true)
    ninv = normalize ? inv(length(flow)) : 1
    # node index -> set of edges coming into that node
    edge_vec = collect(edges(g))
    neighboring_edge_dict = Dict{Int, Vector{Int}}()
    sum_inflows = zeros(nv(g))
    for u in 2:nv(g)
        neighboring_edge_dict[u] = Int[]
        for (idx, e) in enumerate(edge_vec)
            if dst(e) == u
                push!(neighboring_edge_dict[u], idx)
                sum_inflows[u] += flow[idx]
            end
        end
    end

    function scaled_poisson_loss(p)
        sum_flows = sum(p)
        lin_part = sum(2:nv(g)) do u
            sum()
        end
        return ninv * (lin_part + log_part)
    end
    function poisson_grad!(storage, p)
        storage .= ninv
        for u in 2:nv(g)
            for e_idx in neighboring_edge_dict[u]
                if sum_inflows[u] > 0
                    storage[e_idx] -= ninv * sum_inflows[u] * inv(sum(p[idx] for idx in neighboring_edge_dict[u]) + 1e-2)
                end
            end
        end
        return nothing
    end
    return poisson_loss, poisson_grad!
end

function build_domain_oracle(flow, g)
    edge_vec = collect(edges(g))
    neighboring_edge_dict = Dict{Int, Vector{Int}}()
    sum_inflows = zeros(nv(g))
    for u in 2:nv(g)
        neighboring_edge_dict[u] = Int[]
        for (idx, e) in enumerate(edge_vec)
            if dst(e) == u
                push!(neighboring_edge_dict[u], idx)
                sum_inflows[u] += flow[idx]
            end
        end
    end
    function domain_oracle(x)

    end
end

function get_ordering(g)
    s = Graphs.floyd_warshall_shortest_paths(g)
    return sortperm(s.dists * ones(BigFloat, nv(g)))
end

function build_optimization_model(g, flow_bounds; optimizer_constructor=SCIP.Optimizer)
    
    k = (ne(g) - nv(g) + 2)
    ordering = get_ordering(g)
    @assert ordering[1] == 1
    @assert ordering[end] == nv(g)
    s_idx = 1
    t_idx = nv(g)
    
    flow_max = maximum(getindex.(values(flow_bounds), 2))
    
    m = direct_model(optimizer_constructor())
    @variable(m, x[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], Bin)
    @variable(m, w[1:k] >= 0, Int)
    @variable(m, phi[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)] >= 0, Int)
    
    @constraint(m, flow_origin[i=1:k], sum(x[s_idx,v,i] for v in outneighbors(g, s_idx)) == 1)
    @constraint(m, flow_destination[i=1:k], sum(x[u,t_idx,i] for u in inneighbors(g, t_idx)) == 1)
    @constraint(m, flow_conservation[v=2:(t_idx-1),i=1:k], sum(x[u,v,i] for u in inneighbors(g, v)) == sum(x[v,u,i] for u in outneighbors(g, v)))
    # nonlinear constraint
    # @constraint(m, flow_bound_matching[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], flow_bounds[u => v][1] <= dot(x[u,v,:], w) <= flow_bounds[u => v][2])
    # linearized version
    @assert all(f[1] <= f[2] for f in values(flow_bounds))
    @constraint(m, flow_bound_matching[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], flow_bounds[u => v][1] <= sum(phi[u,v,:]) <= flow_bounds[u => v][2] )
    @constraint(m, phi_upperbound1[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], phi[u,v,i] <= flow_max * x[u,v,i])
    @constraint(m, phi_upperbound2[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], phi[u,v,i] <= w[i])
    @constraint(m, phi_lowerbound[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)],  phi[u,v,i] >= w[i] - flow_max * (1 - x[u,v,i]))
    @objective(m, Min, sum(w))
    return m
end

function build_robust_optimization_model(g, flow_values::Dict{<:Pair,<:Real}; optimizer_constructor=SCIP.Optimizer)
    k = (ne(g) - nv(g) + 2)
    ordering = get_ordering(g)
    @assert ordering[1] == 1
    @assert ordering[end] == nv(g)
    s_idx = 1
    t_idx = nv(g)
    
    flow_max = maximum(values(flow_values))
    
    m = direct_model(optimizer_constructor())
    @variable(m, x[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], Bin)
    @variable(m, w[1:k] >= 0, Int)
    @variable(m, rho[1:k] >= 0, Int)
    @variable(m, phi[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)] >= 0, Int)
    @variable(m, gamma[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)] >= 0, Int)

    @constraint(m, flow_origin[i=1:k], sum(x[s_idx,v,i] for v in outneighbors(g, s_idx)) == 1)
    @constraint(m, flow_destination[i=1:k], sum(x[u,t_idx,i] for u in inneighbors(g, t_idx)) == 1)
    @constraint(m, flow_conservation[v=2:(t_idx-1),i=1:k], sum(x[u,v,i] for u in inneighbors(g, v)) == sum(x[v,u,i] for u in outneighbors(g, v)))
    # nonlinear constraint
    # @constraint(m, flow_bound_matching[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], flow_bounds[u => v][1] <= dot(x[u,v,:], w) <= flow_bounds[u => v][2])
    # linearized constraint of gamma = rho x
    @constraint(m, error_linearized_upper_x[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], gamma[u,v,i] <=  x[u,v,i] * flow_max)
    @constraint(m, error_linearized_upper_rho[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], gamma[u,v,i] <=  rho[i])
    @constraint(m, error_linearized_lower[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], gamma[u,v,i] >= rho[i] - (1- x[u,v,i]) * flow_max)
    @constraint(m, flow_error_lowerbound[u=1:nv(g),v=1:nv(g);Edge(u=>v) in edges(g)], flow_values[u => v] - sum(phi[u,v,i] for i in 1:k) <= sum(gamma[u,v,i] for i in 1:k) )
    @constraint(m, flow_error_upperbound[u=1:nv(g),v=1:nv(g);Edge(u=>v) in edges(g)], flow_values[u => v] - sum(phi[u,v,i] for i in 1:k) >= -sum(gamma[u,v,i] for i in 1:k) )
    @constraint(m, phi_upperbound1[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], phi[u,v,i] <= flow_max * x[u,v,i])
    @constraint(m, phi_upperbound2[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)], phi[u,v,i] <= w[i])
    @constraint(m, phi_lowerbound[u=1:nv(g),v=1:nv(g),i=1:k;Edge(u=>v) in edges(g)],  phi[u,v,i] >= w[i] - flow_max * (1 - x[u,v,i]))
    @objective(m, Min, sum(rho))
    return m
end


function build_earlystopping_callback(flow)
    function callback(state, active_set, args...)
        # uopt: optimal factor for scaling x to minimize ||x * u - f||^2
        # if round(u * x) == flow, we can stop early
        x = state.x
        uopt = round(Int, dot(x, flow) / norm(x)^2)
        xint = sum(round(Int, active_set.weights[k] * uopt) * active_set.atoms[k] for k in eachindex(active_set))
        # if initial flow exactly matched, stop the run
        if dot(xint, xint) - 2dot(xint, flow) + dot(flow, flow) == 0
            return false
        end
        return true
    end
end

function build_dicg_callback_vertices(storage, callback=nothing)
    function storage_callback(state, args...)
        a, v = args
        @assert state.v === v
        push!(storage, state.v)
        if a !== nothing
            push!(storage, a)
        end
        if callback !== nothing
            return callback(state, args...)
        end
        return true
    end
end

function integer_weights_from_sol(x, flow, active_set)
    uopt = round(Int, dot(x, flow) / norm(x)^2)
    return [round(Int, active_set.weights[k] * uopt) for k in eachindex(active_set)]
end

function integer_weights_and_flow_from_sol(x, flow, active_set)
    uopt = round(Int, dot(x, flow) / norm(x)^2)
    weights = [round(Int, active_set.weights[k] * uopt) for k in eachindex(active_set)]
    flow = sum(weights[k] * active_set.atoms[k] for k in eachindex(weights))
    return weights, flow
end

struct ExactStep{T, VT} <: FrankWolfe.LineSearchMethod
    flow::VT
    flownormsq::T
    accurate::Bool
end

ExactStep(flow, accurate::Bool = false) = ExactStep(1.0 * flow, dot(flow, flow), accurate)

function FrankWolfe.perform_line_search(
    ls::ExactStep,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode,
)
    if ls.accurate
        dot_dx = AccurateArithmetic.dot_oro(x, d)
        dot_dr = AccurateArithmetic.dot_oro(ls.flow, d)
        dot_xr = AccurateArithmetic.dot_oro(x, ls.flow)
        dot_dd = AccurateArithmetic.dot_oro(d,d)
    else
        dot_dx = dot(x, d)
        dot_dr = dot(ls.flow, d)
        dot_xr = dot(x, ls.flow)
        dot_dd = dot(d,d)
    end
    gamma = (ls.flownormsq * dot_dx - dot_dr * dot_xr ) / (dot_dd * ls.flownormsq - dot_dr^2)
    return gamma
end

struct ExactStep2{T, VT} <: FrankWolfe.LineSearchMethod
    flow::VT
    flownormsq::T
end

ExactStep2(flow) = ExactStep2(1.0 * flow, dot(flow, flow))

function FrankWolfe.perform_line_search(
    ls::ExactStep2,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode,
)
    dot_dx = dot(x, d)
    dot_dr = dot(ls.flow, d)
    dot_xr = dot(x, ls.flow)
    dot_dd = dot(d,d)
    gamma = (dot_dx - dot_dr * dot_xr / ls.flownormsq ) / dot_dd
    return gamma
end

struct JuMPShortestPathLMO{G} <: FrankWolfe.LinearMinimizationOracle
    g::G
end

function FrankWolfe.compute_extreme_point(lmo::JuMPShortestPathLMO, direction; v=nothing)
    g = lmo.g
    m = Model(SCIP.Optimizer)
    @variable(m, 0 <= x[edges(g)] <= 1)
    @constraint(
        m,
        flow_conservation[i=2:nv(g)-1],
        sum(x[Edge(i,j)] for j in outneighbors(g,i); init=0.0) == sum(x[Edge(j,i)] for j in inneighbors(g,i); init=0.0),
    )
    @constraint(
        m,
        flow_start,
        sum(x[Edge(1,j)] for j in outneighbors(g,1); init=0.0) >= 1.0,
    )
    @objective(m, Min, dot(direction, x))
    set_silent(m)
    optimize!(m)
    return [JuMP.value(x[e]) .> 0.5 for e in edges(g)]
end

FrankWolfe.is_decomposition_invariant_oracle(::CO.ShortestPathLMO) = true

function FrankWolfe.compute_inface_extreme_point(lmo::CO.ShortestPathLMO, direction, x; lazy=false, kwargs...)
    new_direction = 1 * direction
    Mconst = sum(abs, direction)
    for idx in eachindex(direction)
        if x[idx] <= 0
            new_direction[idx] = Mconst
        elseif x[idx] >= 1
            new_direction[idx] = -Mconst
        end
    end
    v = FrankWolfe.compute_extreme_point(lmo, new_direction; kwargs...)
    for idx in eachindex(direction)
        if x[idx] <= 0
            @assert v[idx] == 0 "$(v[idx])"
        elseif x[idx] >= 1
            @assert v[idx] == 1
        end
    end
    return v
end

function FrankWolfe.dicg_maximum_step(::CO.ShortestPathLMO, direction, x)
    T = promote_type(eltype(x), eltype(direction))
    gamma_max = one(T)
    for idx in eachindex(x)
        if direction[idx] != 0.0
            # iterate already on the boundary
            if (direction[idx] < 0 && x[idx] ≈ 1) || (direction[idx] > 0 && x[idx] ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if direction[idx] > 0
                gamma_max = min(gamma_max, x[idx] / direction[idx])
            else
                @assert direction[idx] < 0
                gamma_max = min(gamma_max, -(1 - x[idx]) / direction[idx])
            end
        end
    end
    return gamma_max
end

struct OriginScaledShortestPathPolytope{LMO,R} <: FrankWolfe.LinearMinimizationOracle
    sp_lmo::LMO
    radius::R
end

OriginScaledShortestPathPolytope(g, s_idx, t_idx, radius) = OriginScaledShortestPathPolytope(CO.ShortestPathLMO(g, s_idx, t_idx), radius)

function FrankWolfe.compute_extreme_point(lmo::OriginScaledShortestPathPolytope, direction; kwargs...)
    v = FrankWolfe.compute_extreme_point(lmo.sp_lmo, direction; kwargs...)
    if dot(v, direction) > 0
        return zero(lmo.radius) * v
    end
    return lmo.radius * v
end

struct OriginScaledShortestPathPolytope2{LMO,F} <: FrankWolfe.LinearMinimizationOracle
    sp_lmo::LMO
    flow::F
end

OriginScaledShortestPathPolytope2(g, s_idx, t_idx, flow) = OriginScaledShortestPathPolytope2(CO.ShortestPathLMO(g, s_idx, t_idx), flow)

function FrankWolfe.compute_extreme_point(lmo::OriginScaledShortestPathPolytope2, direction; kwargs...)
    v = FrankWolfe.compute_extreme_point(lmo.sp_lmo, direction; kwargs...)
    if dot(v, direction) > 0
        return zero(eltype(lmo.flow)) * v
    end
    return maximum(lmo.flow[v]) * v
end

# Recipe for plotting markers in plot_trajectories
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; n_markers=10, log=false)
    n = length(y)

    # Choose datapoints for markers
    if log
        xmin = log10(x[1])
        xmax = log10(x[end])
        thresholds = collect(xmin:(xmax-xmin)/(n_markers-1):xmax)
        indices = [argmin(i -> abs(t - log10(x[i])), eachindex(x)) for t in thresholds]
    else
        indices = 1:Int(ceil(length(x) / n_markers)):n
    end
    sx, sy = x[indices], y[indices]

    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
    z_order := 1
end
