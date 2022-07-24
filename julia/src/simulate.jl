mutable struct ColloidSimParams
    all_moves::Integer
    all_rotations::Integer
    accepted_moves::Integer
    accepted_rotations::Integer
    energies::Vector{<:AbstractFloat}
    nematic_orders::Vector{<:AbstractFloat}
end

function Base.:+(params1::ColloidSimParams, params2::ColloidSimParams)
    return ColloidSimParams(params1.all_moves + params2.all_moves,
        params1.all_rotations + params2.all_rotations,
        params1.accepted_moves + params2.accepted_moves,
        params1.accepted_rotations + params2.accepted_rotations,
        vcat(params1.energies, params2.energies),
        vcat(params1.nematic_orders, params2.nematic_orders))
end

function simulate!(colloid::Colloid, move_radius::Real, rotation_span::Real,
        interaction_strength::Real = Inf; steps::Integer = 100, calculate::Bool = false)
    F = typeof(colloid.particles[1].radius)
    move_radius, rotation_span = F(move_radius), F(rotation_span)

    accepted_rotations, accepted_moves = 0, 0
    if calculate
        energies, nematic_orders = Vector{F}(undef, steps + 1), Vector{F}(undef, steps + 1)
        energies[1] = potenergy(colloid, interaction_strength)
        nematic_orders[1] = nematic_order(colloid)
    end

    # precalculate random numbers to speed-up calculations
    move_or_rotate = rand(Bool, steps) # true: move, false: rotate
    random_choices = rand(1:length(colloid.particles), steps)
    rnd = rand(F, 3, steps)

    for step in 1:steps
        particle = colloid.particles[random_choices[step]]
        if calculate
            nematic_orders[step+1] = nematic_orders[step] - nematic_order(particle)
        end
        if move_or_rotate[step]
            acc, ΔE = _one_mc_movement!(colloid, particle,
                (rnd[1, step], rnd[2, step], rnd[3, step]), move_radius,
                interaction_strength)
            accepted_moves += acc
        else
            acc, ΔE = _one_mc_rotation!(colloid, particle,
                (rnd[1, step], rnd[2, step]), rotation_span, interaction_strength)
            accepted_rotations += acc
        end
        if calculate
            nematic_orders[step+1] += nematic_order(particle)
            energies[step+1] = energies[step] + ΔE
        end
    end

    all_moves = count(move_or_rotate)
    if calculate
        return ColloidSimParams(all_moves, steps - all_moves, accepted_moves,
            accepted_rotations, energies, nematic_orders)
    else
        return accepted_moves / all_moves, accepted_rotations / (steps - all_moves)
    end
end

function batchsim!(colloid::Colloid,
    move_radii::Vector{<:Real}, rotation_spans::Vector{<:Real},
    interaction_strengths::Vector{<:Real} = [1.0, 3.0, 10.0, 100.0];
    steps::Vector{<:Integer} = [10000, 10000, 10000, 10000],
    calculate::Bool = false)
if calculate
    params = ColloidSimParams(0, 0, 0, 0, Float64[], Float64[])
    for i in 1:length(interaction_strengths)
        params += simulate!(colloid, move_radii[i], rotation_spans[i],
            interaction_strengths[i], steps = steps[i], calculate = true)
    end
    return params
else
    for i in 1:length(interaction_strengths)
        simulate!(colloid, move_radius, rotation_span,
            interaction_strengths[i], steps = steps[i], calculate = false)
    end
end
end

function _one_mc_movement!(colloid::Colloid, particle::AbstractPolygon,
    rnd::Tuple{Vararg{<:Real}}, move_radius::Real, interaction_strength::Real)
if !isinf(interaction_strength)
    prevpot = sum(p -> potential(p, particle, periodic_boundary_shift(colloid.boxsize)),
        filter(!=(particle), colloid.particles))
end

colloid._temp_vertices .= particle.vertices
colloid._temp_center .= particle.center

r = move_radius * rnd[1]
θ = 2π * rnd[2]

move!(particle, (r * cos(θ), r * sin(θ)))
apply_periodic_boundary!(particle, colloid.boxsize)

if isinf(interaction_strength)
    if any(p -> is_overlapping(particle, p, periodic_boundary_shift(colloid.boxsize)),
            filter(!=(particle), colloid.particles))
        particle.vertices .= colloid._temp_vertices
        particle.center .= colloid._temp_center
        return 0, 0
    else
        return 1, 0
    end
else
    newpot = sum(p -> potential(p, particle, periodic_boundary_shift(colloid.boxsize)),
        filter(!=(particle), colloid.particles))
    if ℯ^(interaction_strength * (prevpot - newpot)) > rnd[3]
        return 1, newpot - prevpot
    else
        particle.vertices .= colloid._temp_vertices
        particle.center .= colloid._temp_center
        return 0, 0
    end
end
end

function _one_mc_rotation!(colloid::Colloid, particle::AbstractPolygon,
    rnd::Tuple{Vararg{<:Real}}, rotation_span::Real, interaction_strength::Real)
if !isinf(interaction_strength)
    prevpot = sum(p -> potential(p, particle, periodic_boundary_shift(colloid.boxsize)),
        filter(!=(particle), colloid.particles))
end

colloid._temp_vertices .= particle.vertices
colloid._temp_normals .= particle.normals

rotate!(particle, rotation_span * (rnd[1] - 0.5))

if isinf(interaction_strength)
    if any(p -> is_overlapping(particle, p, periodic_boundary_shift(colloid.boxsize)),
            filter(!=(particle), colloid.particles))
        particle.vertices .= colloid._temp_vertices
        particle.normals .= colloid._temp_normals
        return 0, 0
    else
        return 1, 0
    end
else
    newpot = sum(p -> potential(p, particle, periodic_boundary_shift(colloid.boxsize)),
        filter(!=(particle), colloid.particles))
    if ℯ^(interaction_strength * (prevpot - newpot)) > rnd[2]
        return 1, newpot - prevpot
    else
        particle.vertices .= colloid._temp_vertices
        particle.normals .= colloid._temp_normals
        return 0, 0
    end
end
