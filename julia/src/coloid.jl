struct Coloid{P<:AbstractPolygon, F<:AbstractFloat}
    particle_sidenum::Integer
    particle_radius::Real
    boxsize::Tuple{<:Real, <:Real}

    particles::AbstractVector{P}
    # for speeding up the reverting of monte carlo steps
    _temp_vertices::AbstractMatrix
    _temp_normals::AbstractMatrix
    _temp_center::AbstractVector

    function Coloid{P, F}(particle_count::Integer, particle_sidenum::Integer,
            particle_radius::Real, boxsize::Tuple{<:Real, <:Real}
            ) where {P<:AbstractPolygon, F<:AbstractFloat}
        boxsize = F.(boxsize)
        particles = [P{F}(particle_sidenum, particle_radius, 0, Tuple(boxsize .* ratio))
            for ratio in eachcol(rand(F, 2, particle_count))]
        new(particle_sidenum, particle_radius, boxsize, particles,
            similar(particles[1].vertices), similar(particles[1].normals),
            similar(particles[1].center))
    end
end

struct ColoidSimParams
    all_moves::Integer
    all_rotations::Integer
    accepted_moves::Integer
    accepted_rotations::Integer
    energy::Vector{<:AbstractFloat}
    nematic_order::Vector{<:AbstractFloat}
end

function Coloid{F}(particle_count::Integer, particle_sidenum::Integer,
        particle_radius::Real, boxsize::Tuple{<:Real, <:Real}) where {F<:AbstractFloat}
    if particle_sidenum % 2 == 0
        P = RegEvenPoly
    else
        P = RegPoly
    end
    Coloid{P, F}(particle_count, particle_sidenum, particle_radius, boxsize)
end

function Coloid(particle_count::Integer, particle_sidenum::Integer,
        particle_radius::Real, boxsize::Tuple{<:Real, <:Real})
    if particle_sidenum % 2 == 0
        P = RegEvenPoly
    else
        P = RegPoly
    end
    Coloid{P, Float32}(particle_count, particle_sidenum, particle_radius, boxsize)
end

function crystal_initialize!(coloid::Coloid, gridwidth::Integer,
        dist::Tuple{<:Real, <:Real}, offset::Tuple{<:Real, <:Real})
    for i in 1:length(coloid.particles)
        particle = coloid.particles[i]
        new_center = (offset[1] + dist[1] * ((i-1) ÷ gridwidth),
            offset[2] + dist[2] * ((i-1) % gridwidth))
        warp = new_center .- particle.center
        particle.center .= new_center
        particle.vertices .+= warp
    end
end

function energy(coloid::Coloid, interaction_strength::Real, index::Integer)
    esum = 0.0
    for j in union(1:index-1, index+1:length(coloid.particles))
        esum += interaction_strength * potential(coloid.particles[index],
            coloid.particles[j], periodic_boundary_shift(coloid.boxsize))
    end
    return esum
end

function energy(coloid::Coloid, interaction_strength::Real)
    esum = 0.0
    for i in 1:length(coloid.particles)-1
        for j in i+1:length(coloid.particles)
            esum += interaction_strength * potential(coloid.particles[i],
                coloid.particles[j], periodic_boundary_shift(coloid.boxsize))
        end
    end
    return esum
end

nematic_order(coloid::Coloid) = mean(nematic_order, coloid.particles)

function nematic_order(poly::AbstractPolygon)
    return (3 * max(poly.normals[1]^2, poly.normals[2]^2) - 1) / 2
end

function add_random_particle!(coloid::Coloid)
    new_particle = eltype(coloid.particles)(coloid.particle_sidenum, coloid.particle_radius,
        2π / coloid.particle_sidenum * rand(),
        (coloid.boxsize[1] * rand(), coloid.boxsize[2] * rand()))
    for particle in coloid.particles
        if is_overlapping(particle, new_particle)
            return
        end
    end
    push!(coloid.patricles, new_particle)
end

function add_random_particles!(coloid::Coloid, count::Integer)
    rnd = rand(3, count)
    for i in 1:count
        new_particle = eltype(coloid.particles)(coloid.particle_sidenum,
            coloid.particle_radius, 2π / coloid.particle_sidenum * rnd[1, i],
            (coloid.boxsize[1] * rnd[2, i], coloid.boxsize[2] * rnd[3, i]))
        flag = true
        for particle in coloid.particles
            if is_overlapping(particle, new_particle)
                flag = false
                break
            end
        end
        if flag
            push!(coloid.particles, new_particle)
        end
    end
end

function simulate!(coloid::Coloid, move_radius::Real, rotation_span::Real;
        interaction_strength::Real = Inf, steps::Integer = 100, calculate::Bool = false)
    F = typeof(coloid.particles[1].radius)
    move_radius, rotation_span = F(move_radius), F(rotation_span)

    accepted_rotations, accepted_moves = 0, 0
    if calculate
        energies, nematic_orders = Vector{F}(undef, steps + 1), Vector{F}(undef, steps + 1)
        energies[1] = energies(coloid, interaction_strength)
        nematic_orders[1] = nematic_order(coloid)
    end

    # precalculate random numbers to speed-up calculations
    move_or_rotate = rand(Bool, steps) # true: move, false: rotate
    random_choices = rand(1:length(coloid.particles), steps)
    rnd = rand(F, 3, steps)

    for step in 1:steps
        particle = coloid.particles[random_choices[step]]
        if calculate
            nematic_orders[step+1] = nematic_orders[step] - nematic_order(particle)
        end
        if move_or_rotate[step]
            acc, ΔE = _one_mc_movement!(coloid, particle,
                (rnd[1, step], rnd[2, step], rnd[3, step]), move_radius,
                interaction_strength)
            accepted_moves += acc
        else
            acc, ΔE += _one_mc_rotation!(coloid, particle,
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
        return ColoidSimParams(all_moves, steps - all_moves, accepted_moves,
            accepted_rotations, energies, nematic_order)
    else
        return accepted_moves / all_moves, accepted_rotations / (steps - all_moves)
    end
end

function _one_mc_movement!(coloid::Coloid, particle::AbstractPolygon,
        rnd::Tuple{Vararg{<:Real}}, move_radius::Real, interaction_strength::Real)
    if !isinf(interaction_strength)
        prevpot = sum(p -> potential(p, particle, periodic_boundary_shift(coloid.boxsize)),
            filter(!=(particle), coloid.particles))
    end
    
    coloid._temp_vertices .= particle.vertices
    coloid._temp_center .= particle.center

    r = move_radius * rnd[1]
    θ = 2π * rnd[2]

    move!(particle, (r * cos(θ), r * sin(θ)))
    apply_periodic_boundary!(particle, coloid.boxsize)

    if isinf(interaction_strength)
        if any(p -> is_overlapping(particle, p, periodic_boundary_shift(coloid.boxsize)),
                filter(!=(particle), coloid.particles))
            particle.vertices .= coloid._temp_vertices
            particle.center .= coloid._temp_center
            return 0, 0
        else
            return 1, 0
        end
    else
        newpot = sum(p -> potential(p, particle, periodic_boundary_shift(coloid.boxsize)),
            filter(!=(particle), coloid.particles))
        if ℯ^(interaction_strength * (prevpot - newpot)) > rnd[3]
            return 1, newpot - prevpot
        else
            particle.vertices .= coloid._temp_vertices
            particle.center .= coloid._temp_center
            return 0, 0
        end
    end
end

function _one_mc_rotation!(coloid::Coloid, particle::AbstractPolygon,
        rnd::Tuple{Vararg{<:Real}}, rotation_span::Real, interaction_strength::Real)
    if !isinf(interaction_strength)
        prevpot = sum(p -> potential(p, particle, periodic_boundary_shift(coloid.boxsize)),
            filter(!=(particle), coloid.particles))
    end
    
    coloid._temp_vertices .= particle.vertices
    coloid._temp_normals .= particle.normals

    rotate!(particle, rotation_span * (rnd[1] - 0.5))

    if isinf(interaction_strength)
        if any(p -> is_overlapping(particle, p, periodic_boundary_shift(coloid.boxsize)),
                filter(!=(particle), coloid.particles))
            particle.vertices .= coloid._temp_vertices
            particle.normals .= coloid._temp_normals
            return 0, inf
        else
            return 1, inf
        end
    else
        newpot = sum(p -> potential(p, particle, periodic_boundary_shift(coloid.boxsize)),
            filter(!=(particle), coloid.particles))
        if ℯ^(interaction_strength * (prevpot - newpot)) > rnd[2]
            return 1, newpot - prevpot
        else
            particle.vertices .= coloid._temp_vertices
            particle.normals .= coloid._temp_normals
            return 0, newpot - prevpot
        end
    end
end

periodic_boundary_shift(boxsize) = (
    x -> (-(x[1] ÷ (boxsize[1]/2)) * boxsize[1],
        -(x[2] ÷ (boxsize[2]/2)) * boxsize[2]))
