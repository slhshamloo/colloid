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
    random_engine::AbstractRNG

    function Simulation(particle_count::Integer, sidenum::Integer, radius::Real,
                        boxsize::Tuple{<:Real, <:Real}; seed::Integer=-1, use_gpu=false,
                        numtype::DataType=Float32)
        if seed == -1
            seed = rand(0:typemax(UInt))
        end
        random_engine = Xoshiro(seed)

        colloid = Colloid{Array, numtype}(particle_count, sidenum, radius, boxsize)
        new(colloid, seed, 0, zero(numtype), zero(numtype), 0, 0, 0, 0,
            AbstractConstraint[], AbstractRecorder[], AbstractUpdater[],
            numtype, use_gpu, random_engine)
    end
end

function run!(sim::Simulation, timesteps::Integer)
    (sim.accepted_translations, sim.rejected_translations,
        sim.accepted_rotations, sim.rejected_rotations) = (0, 0, 0, 0)
    if !sim.uses_gpu
        cell_list = SeqCellList(sim.colloid)
    end

    for _ in 1:timesteps
        if !sim.uses_gpu
            apply_step!(sim, cell_list)
        end

        for updater in sim.updaters
            cell_list = update!(sim, updater, cell_list)
        end
        for constraint in sim.constraints
            constraint.update!(sim, constraint)
        end
        for recorder in sim.recorders
            record!(sim, recorder)
        end

        sim.timestep += 1
    end
end

function apply_step!(sim::Simulation, cell_list::CellList)
    translate_or_rotate = rand(sim.random_engine, Bool, particle_count(sim.colloid))
    randnums = rand(sim.random_engine, sim.numtype, 2, particle_count(sim.colloid))
    iter = (rand(sim.random_engine, Bool) ?
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

function apply_translation!(sim::Simulation, cell_list::CellList,
                            randnums::Matrix{<:Real}, idx::Int)
    r = sim.move_radius * randnums[1, idx]
    θ = 2π * randnums[2, idx]
    x, y = r * cos(θ), r * sin(θ)

    i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
    deleteat!(cell_list.cells[i, j], findfirst(==(idx), cell_list.cells[i, j]))
    move!(sim.colloid, idx, x, y)
    i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
    push!(cell_list.cells[i, j], idx)

    if violates_constraints(sim, idx) || has_overlap(sim.colloid, cell_list, idx, i, j)
        move!(sim.colloid, idx, -x, -y)
        pop!(cell_list.cells[i, j])
        i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
        push!(cell_list.cells[i, j], idx)
        sim.rejected_translations += 1
    else
        sim.accepted_translations += 1
    end
end

function apply_rotation!(sim::Simulation, cell_list::CellList,
                         randnums::Matrix{<:Real}, idx::Int)
    angle_change = sim.rotation_span * (randnums[2, idx] - 0.5)
    sim.colloid.angles[idx] += angle_change
    i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
    if has_overlap(sim.colloid, cell_list, idx, i, j)
        sim.colloid.angles[idx] -= angle_change
        sim.rejected_rotations += 1
    else
        sim.accepted_rotations += 1
    end
end

@inline function violates_constraints(sim::Simulation, idx::Integer)
    for constraint in sim.constraints
        if is_violated(sim, constraint, idx)
            return true
        end
    end
    return false
end

@inline function accept_translation!(sim::Simulation, idx::Integer)
    sim.colloid._temp_centers[1, idx] = sim.colloid.centers[1, idx]
    sim.colloid._temp_centers[2, idx] = sim.colloid.centers[2, idx]
    sim.accepted_translations += 1
end

@inline function reject_translation!(sim::Simulation, idx::Integer)
    sim.colloid.centers[1, idx] = sim.colloid._temp_centers[1, idx]
    sim.colloid.centers[2, idx] = sim.colloid._temp_centers[2, idx]
    sim.rejected_translations += 1
end

@inline function accept_rotation!(sim::Simulation, idx::Integer)
    sim.colloid._temp_angles[idx] = sim.colloid.angles[idx]
    sim.accepted_rotations += 1
end

@inline function reject_rotation!(sim::Simulation, idx::Integer)
    sim.colloid.angles[idx] = sim.colloid._temp_angles[idx]
    sim.rejected_rotations += 1
end
