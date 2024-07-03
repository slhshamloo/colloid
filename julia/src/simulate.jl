"""
    HPMCSimulation(count, sidenum, radius, boxsize; boxshear=0, seed=-1, gpu=false, double=false, beta=1, potential=nothing, pairpotential=nothing)
    HPMCSimulation(particles; ...)

Make simulation structure containing all the information about the system.

The first method makes particle collection (for now, only regualr polygons) with random
initial configuration (made with`undef` arrays to be exact), while the second takes the
particle collection as input.

# Arguments
-`particles::ParticleCollection`: Pre-built particle collection passed to the simulation.
    For now only accepts `RegularPolygons`.
-`count::Integer`: the number of particles.
-`sidenum::Integer`: the number of sides of the regular polygons.
-`radius::Real`: the radius of the particles.
-`boxsize::Tuple{<:Real, <:Real}`: the dimensions of the simulation box.
-`boxshear::Real = 0`: the tangent of the shear angle of the box, defined as the complement
    of the complement of the accute angle of the box parallelogram.
-`seed::Integer = -1`: The seed for initializing the random generator. If set to -1, a
    random seed will be passed to the generator.
-`gpu::Bool = false`: whether to use CUDA gpu acceleration or not.
-`double::Bool = false`: whether to use double-precision floating-point numbers for
    calculations or single-presision ones.
-`beta::Real = 1`: the inverse temperature of the system (more precisely, ``1 / k_B T``),
    used for the Monte Carlo updates that involve potentials or box updates.
-`potential::Union{Function, Nothing} = nothing`: Function for calculating potentials for
    particles in the system, taking the particle collection as the first argument and the
    index of the particle as the second argument. If set to `nothing`, no potential is
    calculated for the particles in addition to their pair interaction.
-`pairpotential::Union{Function, Nothing} = nothing`: Function for calculating potentials
    for pairs of particles, defining the interaction of the particles. If set to `nothing`,
    particles interact via volume exclusion, i.e. hard particle interactions.
"""
mutable struct HPMCSimulation{F<:AbstractFloat}
    particles::ParticleCollection
    cell_list::Union{CellList, Nothing}

    seed::Integer
    timestep::Integer

    move_radius::F
    rotation_span::F
    beta::F

    accepted_translations::Integer
    rejected_translations::Integer
    accepted_rotations::Integer
    rejected_rotations::Integer

    constraints::AbstractVector{<:AbstractConstraint}
    recorders::AbstractVector{<:AbstractRecorder}
    updaters::AbstractVector{<:AbstractUpdater}

    potential::Union{Function, Nothing}
    pairpotential::Union{Function, Nothing}
    particle_potentials::AbstractVector{<:Real}

    gpu::Bool
    numtype::DataType
end

function HPMCSimulation(count::Integer, sidenum::Integer, radius::Real,
        boxsize::Tuple{<:Real, <:Real}; boxshear::Real = 0, seed::Integer = -1,
        gpu::Bool = false, double::Bool = false, beta::Real = 1,
        potential::Union{Function, Nothing} = nothing,
        pairpotential::Union{Function, Nothing} = nothing)
    numtype = double ? Float64 : Float32
    if seed == -1
        seed = rand(0:typemax(UInt))
    end
    if gpu
        CUDA.seed!(seed)
        particles = RegularPolygons{numtype}(sidenum, radius, boxsize, count;
                                             gpu=true, boxshear=boxshear)
    else
        Random.seed!(seed)
        particles = RegularPolygons{numtype}(sidenum, radius, boxsize, count;
                                             boxshear=boxshear)
    end
    if !isnothing(potential) || !isnothing(pairpotential)
        particle_potentials = zeros(numtype, particlecount(particles))
        if gpu
            particle_potentials = CuArray(particle_potentials)
        end
    else
        particle_potentials = (gpu ? CuVector{numtype}(undef, 0)
                                   : Vector{numtype}(undef, 0))
    end
    HPMCSimulation{numtype}(particles, nothing, seed, 0, zero(numtype), zero(numtype),
        convert(numtype, beta), 0, 0, 0, 0, AbstractConstraint[], AbstractRecorder[],
        AbstractUpdater[], potential, pairpotential, particle_potentials, gpu, numtype)
end

function HPMCSimulation(particles::RegularPolygons; seed::Integer = -1, gpu::Bool = false,
        double::Bool = false, beta::Real = 1, potential::Union{Function, Nothing} = nothing,
        pairpotential::Union{Function, Nothing} = nothing)
    numtype = double ? Float64 : Float32
    if seed == -1
        seed = rand(0:typemax(UInt))
    end
    if gpu
        CUDA.seed!(seed)
        cell_list = CuCellList(particles)
    else
        Random.seed!(seed)
        cell_list = SeqCellList(particles)
    end
    if !isnothing(potential) || !isnothing(pairpotential)
        particle_potentials = zeros(numtype, particlecount(particles))
        if gpu
            particle_potentials = CuArray(particle_potentials)
        end
    else
        particle_potentials = (gpu ? CuVector{numtype}(undef, 0)
                                   : Vector{numtype}(undef, 0))
    end
    HPMCSimulation{numtype}(particles, cell_list, seed, 0, zero(numtype), zero(numtype),
        convert(numtype, beta), 0, 0, 0, 0, AbstractConstraint[], AbstractRecorder[],
        AbstractUpdater[], potential, pairpotential, particle_potentials, gpu, numtype)
end

# Structure for passing simulation information too the gpu
struct CuHPMCSim{C<:RegularPolygons, RC<:RawConstraints, F<:AbstractFloat,
        V<:AbstractVector, P<:Union{Function, Nothing}, PP<:Union{Function, Nothing}}
    particles::C
    constraints::RC

    move_radius::F
    rotation_span::F
    beta::F

    potential::P
    pairpotential::PP
    particle_potentials::V
end

@inline CuHPMCSim(particles::RegularPolygons,
        constraints::RawConstraints, move_radius::F, rotation_span::F, beta::F,
        potential::Union{Function, Nothing}, pairpotential::Union{Function, Nothing},
        particle_potentials::AbstractVector) where {F<:AbstractFloat} =
    CuHPMCSim{typeof(particles), typeof(constraints), F, typeof(particle_potentials),
        typeof(potential), typeof(pairpotential)}(particles, constraints, move_radius,
        rotation_span, beta, potential, pairpotential, particle_potentials)

Adapt.@adapt_structure CuHPMCSim

"""
    run!(sim, timesteps)

Run the simulation for `timesteps` number of steps.
"""
function run!(sim::HPMCSimulation, timesteps::Integer)
    (sim.accepted_translations, sim.rejected_translations,
        sim.accepted_rotations, sim.rejected_rotations) = (0, 0, 0, 0)
    if sim.gpu
        sim.cell_list = CuCellList(sim.particles)
    else
        sim.cell_list = SeqCellList(sim.particles)
    end

    for _ in 1:timesteps
        if sim.gpu
            apply_step_gpu!(sim)
        else
            apply_step_cpu!(sim)
        end
        for updater in sim.updaters
            update!(sim, updater)
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
