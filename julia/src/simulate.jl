mutable struct Simulation
    colloid::Colloid

    seed::Integer
    timestep::Integer

    move_radius::Real
    rotation_span::Real

    accepted_translations::Integer
    rejected_translations::Integer
    accepted_rotations::Integer
    rejected_rotations::Integer

    constraints::AbstractVector{<:AbstractConstraint}
    recorders::AbstractVector{<:AbstractRecorder}
    updaters::AbstractVector{<:AbstractUpdater}

    beta::Real
    potential::Union{Function, Nothing}
    pairpotential::Union{Function, Nothing}

    gpu::Bool
    numtype::DataType

    function Simulation(particle_count::Integer, sidenum::Integer, radius::Real,
        boxsize::Tuple{<:Real, <:Real}; seed::Integer = -1, gpu::Bool = false,
        double::Bool = false, beta::Real = 1, potential = nothing,
        pairpotential = nothing)
    numtype = double ? Float64 : Float32
    if seed == -1
        seed = rand(0:typemax(UInt))
    end
    if gpu
        CUDA.seed!(seed)
        colloid = Colloid{numtype}(particle_count, sidenum, radius, boxsize; gpu=true)
    else
        Random.seed!(seed)
        colloid = Colloid{numtype}(particle_count, sidenum, radius, boxsize)
    end
    new(colloid, seed, 0, zero(numtype), zero(numtype), 0, 0, 0, 0,
        AbstractConstraint[], AbstractRecorder[], AbstractUpdater[],
        convert(numtype, beta), potential, pairpotential, gpu, numtype)
    end
end

function run!(sim::Simulation, timesteps::Integer)
    (sim.accepted_translations, sim.rejected_translations,
        sim.accepted_rotations, sim.rejected_rotations) = (0, 0, 0, 0)
    if sim.gpu
        cell_list = CuCellList(sim.colloid)
    else
        cell_list = SeqCellList(sim.colloid)
    end

    for _ in 1:timesteps
        apply_step!(sim, cell_list)
        for updater in sim.updaters
            cell_list = update!(sim, updater, cell_list)
        end
        for constraint in sim.constraints
            constraint.update!(sim, constraint)
        end
        for recorder in sim.recorders
            record!(sim, recorder, cell_list)
        end
        sim.timestep += 1
    end
end
