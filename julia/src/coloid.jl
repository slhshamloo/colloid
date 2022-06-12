struct Coloid{P<:AbstractPolygon, F<:AbstractFloat}
    particle_count::Integer
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
        new(particle_count, particle_sidenum, particle_radius, boxsize, particles,
            similar(particles[1].vertices), similar(particles[1].normals),
            similar(particles[1].center))
    end
end

function Coloid{P}(particle_count::Integer, particle_sidenum::Integer,
        particle_radius::Real, boxsize::Tuple{<:Real, <:Real}) where {P<:AbstractPolygon}
    Coloid{P, Float32}(particle_count, particle_sidenum, particle_radius, boxsize)
end

function Coloid(particle_count::Integer, particle_sidenum::Integer,
        particle_radius::Real, boxsize::Tuple{<:Real, <:Real})
    Coloid{RegPoly, Float32}(particle_count, particle_sidenum, particle_radius, boxsize)
end

function crystal_initialize!(coloid::Coloid, gridwidth::Integer,
        dist::Tuple{<:Real, <:Real}, offset::Tuple{<:Real, <:Real})
    for i in 1:coloid.particle_count
        coloid.particles[i].center .= (offset[1] + dist[1] * ((i-1) ÷ gridwidth),
            offset[2] * dist[2] * ((i-1) % gridwidth))
    end
end

function add_random_particle!(coloid::Coloid)
    new_particle = eltype(coloid.particles)(coloid.particle_sidenum, coloid.particle_radius,
        2π / coloid.particle_sidenum * rand(), coloid.boxsize .* rand(2))
    for particle in coloid.particles
        if is_overlapping(particle, new_particle)
            return
        end
    end
    push!(coloid.patricles, new_particle)
end

function add_random_particles!(coloid::Coloid, count::Integer)
    random_numbers = rand(3, count)
    for i in 1:count
        new_particle = eltype(coloid.particles)(coloid.particle_sidenum,
            coloid.particle_radius, 2π / coloid.particle_sidenum * random_numbers[i, 1],
            coloid.boxsize .* random_numbers[i, 2:3])
        flag = true
        for particle in particles
            if is_overlapping(particle, new_particle)
                flag = false
                break
            end
        end
        if flag
            push!(particles, new_particle)
        end
    end
end

function mcsimulate!(coloid::Coloid, move_radius::Real, rotation_span::Real, steps::Integer)
    F = eltype(coloid.particles[1].radius)
    move_radius = F(move_radius)
    rotation_span = F(rotation_span)

    accepted_rotations = 0
    accepted_moves = 0

    # precalculate random numbers to speed-up calculations
    move_or_rotate = rand(Bool, steps) # true: move, false: rotate
    random_choices = rand(1:coloid.particle_count, steps)
    random_numbers = rand(F, steps)

    for step in 1:steps
        particle = coloid.particles[random_choices[step]]
        if move_or_rotate[step]
            accepted_moves += _one_mc_movement!(coloid, particle,
                random_numbers[step], move_radius)
        else
            accepted_rotations += _one_mc_rotation!(coloid, particle,
                random_numbers[step], rotation_span)
        end
    end

    all_moves = count(move_or_rotate)
    return accepted_moves / all_moves, accepted_rotations / (steps - all_moves)
end

function _one_mc_movement!(coloid::Coloid, particle::AbstractPolygon,
        random_number::Real, move_radius::Real)
    coloid._temp_vertices .= particle.vertices
    coloid._temp_center .= particle.center

    move!(particle, (move_radius * random_number, move_radius * (1-random_number^2)))

    if any(p -> is_overlapping(particle, p), filter(!=(particle), coloid.particles))
        particle.vertices .= coloid._temp_vertices
        particle.center .= coloid._temp_center
        return 0
    else
        return 1
    end
end

function _one_mc_rotation!(coloid::Coloid, particle::AbstractPolygon,
        random_number::Real, rotation_span::Real)
    coloid._temp_vertices .= particle.vertices
    coloid._temp_normals .= particle.normals

    rotate!(particle, rotation_span * (random_number - 0.5))

    if any(p -> is_overlapping(particle, p), filter(!=(particle), coloid.particles))
        particle.vertices .= coloid._temp_vertices
        particle.normals .= coloid._temp_normals
        return 0
    else
        return 1
    end
end

function mcsimulate_periodic!(coloid::Coloid, move_radius::Real, rotation_span::Real,
        steps::Integer)
    F = eltype(coloid.particles[1].radius)
    move_radius = F(move_radius)
    rotation_span = F(rotation_span)
    
    accepted_rotations = 0
    accepted_moves = 0

    # precalculate random numbers to speed-up calculations
    move_or_rotate = rand(Bool, steps) # true: move, false: rotate
    random_choices = rand(1:coloid.particle_count, steps)
    random_numbers = rand(steps)

    for step in 1:steps
        particle = coloid.particles[random_choices[step]]
        if move_or_rotate[step]
            accepted_moves += _one_mc_movement_periodic!(coloid, particle,
                random_numbers[step], move_radius)
        else
            accepted_rotations += _one_mc_rotation_periodic!(coloid, particle,
                random_numbers[step], rotation_span)
        end
    end

    all_moves = count(move_or_rotate)
    return accepted_moves / all_moves, accepted_rotations / (steps - all_moves)
end

function _one_mc_movement_periodic!(coloid::Coloid, particle::AbstractPolygon,
        random_number::Real, move_radius::Real)
    coloid._temp_vertices .= particle.vertices
    coloid._temp_center .= particle.center

    move!(particle, (move_radius * random_number, move_radius * (1-random_number^2)))
    apply_periodic_boundary!(particle, coloid.boxsize)

    if any(p -> is_overlapping_periodic(particle, p, coloid.boxsize),
            filter(!=(particle), coloid.particles))
        particle.vertices .= coloid._temp_vertices
        particle.center .= coloid._temp_center
        return 0
    else
        return 1
    end
end

function _one_mc_rotation_periodic!(coloid::Coloid, particle::AbstractPolygon,
        random_number::Real, rotation_span::Real)
    coloid._temp_vertices .= particle.vertices
    coloid._temp_normals .= particle.normals

    rotate!(particle, rotation_span * (random_number - 0.5))
    apply_periodic_boundary!(particle, coloid.boxsize)

    if any(p -> is_overlapping_periodic(particle, p, coloid.boxsize),
            filter(!=(particle), coloid.particles))
        particle.vertices .= coloid._temp_vertices
        particle.normals .= coloid._temp_normals
        return 0
    else
        return 1
    end
end
