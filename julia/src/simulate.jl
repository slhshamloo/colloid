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
    uses_gpu::Bool

    function Simulation(particle_count::Integer, sidenum::Integer, radius::Real,
                        boxsize::Tuple{<:Real, <:Real}; seed::Integer=-1, use_gpu=false,
                        numtype::DataType=Float32)
        if seed == -1
            seed = rand(0:65535)
        end
        random_engine = MersenneTwister(seed)
        _rand = (x...) -> rand(random_engine, x...)
        colloid = Colloid{Array, Float64}(particle_count, sidenum, radius, boxsize)
        new(colloid, seed, 0, zero(numtype), zero(numtype), 0, 0, 0, 0,
            AbstractConstraint[], AbstractRecorder[], AbstractUpdater[],
            numtype, use_gpu)
    end
end

function run!(sim::Simulation, timesteps::Integer)
    (sim.accepted_translations, sim.rejected_translations,
        sim.accepted_rotations, sim.rejected_rotations) = (0, 0, 0, 0)
    for _ in 1:timesteps
        if !sim.uses_gpu
            apply_step!(sim)
        end
        sim.timestep += 1
    end
end

@inline function apply_step!(sim::Simulation)
    cell_list = get_cell_list(sim.colloid)
    translate_or_rotate = rand(Bool, particle_count(sim.colloid))
    randnums = rand(sim.numtype, 2, particle_count(sim.colloid))
    iter = (rand(Bool) ?
        range(1, particle_count(sim.colloid))
        : range(particle_count(sim.colloid), 1, step=-1)
    )
    for idx in iter
        if translate_or_rotate[idx]
            apply_translation!(sim, cell_list, randnums, idx)
        else
            apply_rotation!(sim, cell_list, randnums, idx)
        end
    end
    return true
end

@inline function apply_translation!(sim::Simulation, cell_list::Matrix{Vector{Int}},
                                    randnums::Matrix{<:Real}, idx::Int)
    r = sim.move_radius * randnums[1, idx]
    θ = 2π * randnums[2, idx]

    i = Int((sim.colloid.centers[1, idx] + sim.colloid.boxsize[1] / 2)
            ÷ (2 * sim.colloid.radius) + 1)
    j = Int((sim.colloid.centers[2, idx] + sim.colloid.boxsize[2] / 2)
            ÷ (2 * sim.colloid.radius) + 1)

    deleteat!(cell_list[i, j], findfirst(==(idx), cell_list[i, j]))
    move!(sim.colloid, idx, r * cos(θ), r * sin(θ))
    apply_periodic_boundary!(sim.colloid, idx)

    i = Int((sim.colloid.centers[1, idx] + sim.colloid.boxsize[1] / 2)
            ÷ (2 * sim.colloid.radius) + 1)
    j = Int((sim.colloid.centers[2, idx] + sim.colloid.boxsize[2] / 2)
            ÷ (2 * sim.colloid.radius) + 1)
    
    push!(cell_list[i, j], idx)
    if has_overlap(sim.colloid, cell_list, idx, i, j)
        reject_move!(sim, idx)
        sim.rejected_translations += 1
    else
        accept_move!(sim, idx)
        sim.accepted_translations += 1
    end
end

@inline function apply_rotation!(sim::Simulation, cell_list::Matrix{Vector{Int}},
                                 randnums::Matrix{<:Real}, idx::Int)
    rotate!(sim.colloid, idx,
            sim.rotation_span * (randnums[2, idx] - 0.5))
    i = Int((sim.colloid.centers[1, idx] + sim.colloid.boxsize[1] / 2)
            ÷ (2 * sim.colloid.radius) + 1)
    j = Int((sim.colloid.centers[2, idx] + sim.colloid.boxsize[2] / 2)
            ÷ (2 * sim.colloid.radius) + 1)
    if has_overlap(sim.colloid, cell_list, idx, i, j)
        reject_move!(sim, idx)
        sim.rejected_translations += 1
    else
        accept_move!(sim, idx)
        sim.accepted_translations += 1
    end
end

@inline function reject_move!(sim::Simulation, idx::Integer)
    sim.colloid.centers[1, idx] = sim.colloid._temp_centers[1, idx]
    sim.colloid.centers[2, idx] = sim.colloid._temp_centers[2, idx]
    sim.colloid.normals[:, :, idx] .= sim.colloid._temp_normals[:, :, idx]
    sim.colloid.vertices[:, :, idx] .= sim.colloid._temp_vertices[:, :, idx]
end

@inline function accept_move!(sim::Simulation, idx::Integer)
    sim.colloid._temp_centers[1, idx] = sim.colloid.centers[1, idx]
    sim.colloid._temp_centers[2, idx] = sim.colloid.centers[2, idx]
    sim.colloid._temp_normals[:, :, idx] .= sim.colloid.normals[:, :, idx]
    sim.colloid._temp_vertices[:, :, idx] .= sim.colloid.vertices[:, :, idx]
end
