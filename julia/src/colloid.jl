struct Colloid{F<:AbstractFloat}
    particle_sidenum::Integer
    particle_radius::Real
    boxsize::Tuple{<:Real, <:Real}

    particles::AbstractVector{<:AbstractPolygon}
    # for speeding up the reverting of monte carlo steps
    _temp_vertices::AbstractMatrix
    _temp_normals::AbstractMatrix
    _temp_center::AbstractVector

    function Colloid{F}(particle_count::Integer, particle_sidenum::Integer,
            particle_radius::Real, boxsize::Tuple{<:Real, <:Real}
            ) where {F<:AbstractFloat}
        if particle_sidenum % 2 == 0
            P = RegEvenPoly
        else
            P = RegPoly
        end
        boxsize = F.(boxsize)
        particles = [P{F}(particle_sidenum, particle_radius, rand(), Tuple(boxsize .* ratio))
            for ratio in eachcol(rand(F, 2, particle_count))]
        new(particle_sidenum, particle_radius, boxsize, particles,
            similar(particles[1].vertices), similar(particles[1].normals),
            similar(particles[1].center))
    end
end

function Colloid(particle_count::Integer, particle_sidenum::Integer,
        particle_radius::Real, boxsize::Tuple{<:Real, <:Real})
    Colloid{Float64}(particle_count, particle_sidenum, particle_radius, boxsize)
end

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

function crystal_initialize!(colloid::Colloid, gridwidth::Integer,
        dist::Tuple{<:Real, <:Real}, offset::Tuple{<:Real, <:Real})
    for i in 1:length(colloid.particles)
        particle = colloid.particles[i]
        new_center = (offset[1] + dist[1] * ((i-1) ÷ gridwidth),
            offset[2] + dist[2] * ((i-1) % gridwidth))
        warp = new_center .- particle.center
        particle.center .= new_center
        particle.vertices .+= warp
    end
end

function energy(colloid::Colloid, interaction_strength::Real, index::Integer)
    esum = 0.0
    for j in union(1:index-1, index+1:length(colloid.particles))
        esum += interaction_strength * potential(colloid.particles[index],
            colloid.particles[j], periodic_boundary_shift(colloid.boxsize))
    end
    return esum
end

function energy(colloid::Colloid, interaction_strength::Real)
    esum = 0.0
    for i in 1:length(colloid.particles)-1
        for j in i+1:length(colloid.particles)
            esum += interaction_strength * potential(colloid.particles[i],
                colloid.particles[j], periodic_boundary_shift(colloid.boxsize))
        end
    end
    return esum
end

nematic_order(colloid::Colloid) = mean(nematic_order, colloid.particles)

function nematic_order(poly::AbstractPolygon)
    return (3 * max(poly.normals[1]^2, poly.normals[2]^2) - 1) / 2
end

function add_random_particle!(colloid::Colloid)
    new_particle = eltype(colloid.particles)(colloid.particle_sidenum, colloid.particle_radius,
        2π / colloid.particle_sidenum * rand(),
        (colloid.boxsize[1] * rand(), colloid.boxsize[2] * rand()))
    for particle in colloid.particles
        if is_overlapping(particle, new_particle)
            return
        end
    end
    push!(colloid.patricles, new_particle)
end

function add_random_particles!(colloid::Colloid, count::Integer)
    rnd = rand(3, count)
    for i in 1:count
        new_particle = eltype(colloid.particles)(colloid.particle_sidenum,
            colloid.particle_radius, 2π / colloid.particle_sidenum * rnd[1, i],
            (colloid.boxsize[1] * rnd[2, i], colloid.boxsize[2] * rnd[3, i]))
        flag = true
        for particle in colloid.particles
            if is_overlapping(particle, new_particle)
                flag = false
                break
            end
        end
        if flag
            push!(colloid.particles, new_particle)
        end
    end
end

function simulate!(colloid::Colloid, move_radius::Real, rotation_span::Real,
        interaction_strength::Real = Inf, steps::Integer = 100, calculate::Bool = false)
    F = typeof(colloid.particles[1].radius)
    move_radius, rotation_span = F(move_radius), F(rotation_span)

    accepted_rotations, accepted_moves = 0, 0
    if calculate
        energies, nematic_orders = Vector{F}(undef, steps + 1), Vector{F}(undef, steps + 1)
        energies[1] = energy(colloid, interaction_strength)
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
        interaction_strengths::Vector{<:Real} = [1.0, 3.0, 10.0, 100.0],
        steps::Vector{<:Integer} = [10000, 10000, 10000, 10000];
        calculate::Bool = false)
    if calculate
        params = ColloidSimParams(0, 0, 0, 0, Float64[], Float64[])
        for i in 1:length(interaction_strengths)
            params += simulate!(colloid, move_radii[i], rotation_spans[i],
                interaction_strengths[i], steps[i], true)
        end
        return params
    else
        for i in 1:length(interaction_strengths)
            simulate!(colloid, move_radius, rotation_span,
                interaction_strengths[i], steps[i], false)
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
end

periodic_boundary_shift(boxsize) = (
    x -> (-(x[1] ÷ (boxsize[1]/2)) * boxsize[1],
        -(x[2] ÷ (boxsize[2]/2)) * boxsize[2]))
