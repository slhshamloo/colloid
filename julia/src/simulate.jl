mutable struct Simulation
    colloid::Colloid

    seed::Integer
    timestep::Integer

    move_radius::AbstractFloat
    rotation_span::AbstractFloat

    accepted_translations::Integer
    rejected_translations::Integer
    accepted_rotations::Integer
    rejected_rotations::Integer

    constraints::AbstractVector{<:AbstractConstraint}
    recorders::AbstractVector{<:AbstractRecorder}
    updaters::AbstractVector{<:AbstractUpdater}

    numtype::DataType
    _rand::Function
    _has_overlap::Function
    _apply_moves!::Function

    function Simulation(particle_count::Integer, sidenum::Integer, radius::Real,
                        boxsize::Tuple{<:Real, <:Real}; seed::Integer=-1, gpu=false,
                        numtype::DataType=Float32)
        if seed == -1
            seed = rand(0:65535)
        end
        random_engine = MersenneTwister(seed)
        _rand = (x...) -> rand(random_engine, x...)
        colloid = Colloid{Array, Float64}(particle_count, sidenum, radius, boxsize)
        new(colloid, seed, 0, zero(numtype), zero(numtype), 0, 0, 0, 0,
            AbstractConstraint[], AbstractRecorder[], AbstractUpdater[],
            numtype, _rand, has_overlap, apply_moves!)
    end
end

function run!(sim::Simulation, timesteps::Integer)
    (sim.accepted_translations, sim.rejected_translations,
        sim.accepted_rotations, sim.rejected_rotations) = (0, 0, 0, 0)
    for _ in 1:timesteps
        translate_or_rotate = sim._rand(Bool, size(sim.colloid.centers, 2))
        randnums = sim._rand(sim.numtype, 2, size(sim.colloid.centers, 2))
        randnums .-= 0.5 
        sim._apply_moves!(sim.colloid, sim.move_radius, sim.rotation_span,
                         translate_or_rotate, randnums)
        if sim._has_overlap(sim.colloid)
            reject_move!(sim, translate_or_rotate)
        else
            accept_move!(sim, translate_or_rotate)
        end
        sim.timestep += 1
    end
end

@inline function reject_move!(sim::Simulation, translate_or_rotate::AbstractArray)
    sim.colloid.centers .= sim.colloid._temp_centers
    sim.colloid.normals .= sim.colloid._temp_normals
    sim.colloid.vertices .= sim.colloid._temp_vertices
    translation_count = count(translate_or_rotate)
    sim.rejected_translations += translation_count
    sim.rejected_rotations += length(translate_or_rotate) - translation_count
end

@inline function accept_move!(sim::Simulation, translate_or_rotate::AbstractArray)
    sim.colloid._temp_centers .= sim.colloid.centers
    sim.colloid._temp_normals .= sim.colloid.normals
    sim.colloid._temp_vertices .= sim.colloid.vertices 
    translation_count = count(translate_or_rotate)
    sim.accepted_translations += translation_count
    sim.accepted_rotations += length(translate_or_rotate) - translation_count
end
