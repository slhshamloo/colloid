abstract type AbstractRecorder end
abstract type AbstractUpdater end
abstract type AbstractBoxUpdater <: AbstractUpdater end

"""
    TrajectoryRecorder <: AbstractRecorder

Records the information of the system in a `trajectory` structure.
"""
mutable struct TrajectoryRecorder <: AbstractRecorder
    filepath::Union{String, Nothing}
    filecounter::Integer
    trajectory::Union{RegularPolygonsTrajectory, Nothing}
    savetomem::Bool
    cond::Function
end

"""
    TrajectoryRecorder(cond::Function; filepath::Union{String, Nothing} = nothing, savetomem::Bool = false)

Make a structure recording trajectories every time step `cond` returns true.
"""
function TrajectoryRecorder(cond::Function;
        filepath::Union{String, Nothing} = nothing, savetomem::Bool = false)
    TrajectoryRecorder(filepath, 0, nothing, savetomem, cond)
end

"""
    LocalParamRecorder <: AbstractRecorder

Calculates and records parameters for all particles in the system.
"""
struct LocalParamRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    values::Vector{Vector{T}}
    times::Vector{Int}
    cond::Function
end

"""
    LocalParamRecorder(cond::Function, type::String, typeparams...; numtype::DataType = Float32)

Make a structure calculating and recording parameters for all particles in the system
every time step where `cond` returns true.

`types` can be set to:
-"katic": ``k``-atic order where `typeparams` are the arguments of the
    [`katic_order``](@ref) function.
-"solidliquid": solid bond count where `typeparams` are the arguments of the
    [`solidliquid``](@ref) function.
"""
function LocalParamRecorder(cond::Function, type::String, typeparams...;
                            numtype::DataType = Float32)
    if type == "solidliquid"
        numtype = Int32
    elseif type != "nematic"
        numtype = Complex{numtype}
    end
    values = Vector{Vector{numtype}}(undef, 0)
    new{numtype}(type, typeparams, values, Int[], cond)
end

"""
    GlobalParamRecorder <: AbstractRecorder

Calculates and records parameters for the system as a whole.
"""
struct GlobalParamRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    values::Vector{T}
    times::Vector{Int}
    cond::Function
end

"""
    GlobalParamRecorder(cond::Function, type::String, typeparams...; numtype::DataType = Float32)

Calculates and records parameters for the system as a whole.

`types` can be set to:
-"orient": Mean orientation of the particles of the system. No `typeparams` needed.
"""
function GlobalParamRecorder(cond::Function, type::String, typeparams...;
                             numtype::DataType = Float32)
    if type != "nematic"
        numtype = Complex{numtype}
    end
    values = Vector{numtype}(undef, 0)
    GlobalParamRecorder{numtype}(type, typeparams, values, Int[], cond)
end

"""
    AreaUpdater <: AbstractBoxUpdater

Isotropically updates box area.
"""
mutable struct AreaUpdater <: AbstractBoxUpdater
    pressure::Real
    areachange::Real
    cond::Function

    accepted_moves::Integer
    rejected_moves::Integer
end

"""
    AreaUpdater(cond::Function, pressure, areachange)

Make a structure proposing an isotropic box update every time step `cond` returns `true`.

A random move size ``ΔV`` is chosen uniformly from the interval from `-areachange`
to `areachange`. Then, if there are no overlaps between hard particles or violations of
constraints, the new configuration is accepted if ``ΔV < 0`` and if
``ΔV > 0``, accepted with probability
```math
e^{-P\\Delta V}.
```
Of course, the potential change is also added to ``PΔV`` if there is any.
"""
function AreaUpdater(cond::Function, pressure::Real, areachange::Real)
    AreaUpdater(pressure, areachange, cond, 0, 0)
end

"""
    BoxMover <: AbstractBoxUpdater

Applies side length change and shear moves for the simulation box.
"""
mutable struct BoxMover <: AbstractBoxUpdater
    pressure::Real
    change::AbstractVector
    weights::Union{AbstractVector, Tuple{Real, Real, Real}}
    cond::Function
    
    accepted_moves::AbstractVector
    rejected_moves::AbstractVector
end

"""
    BoxMover(cond::Function, pressure, xchange, ychange[, schange=0.0]; weights::Union{AbstractVector, Tuple{Real, Real, Real}} = (1.0, 1.0, 1.0))

Make a structrue proposing updates for box shape or dimensions every time step
`cond` returns `true`.

The `weights` specify what proportion of the moves changes side length of the ``x``
direction, the side length of the ``y`` direction, and the shear amount respectively.
`weights` is normalized automatically.

As a change in shear does not change area, a shear update is rejected only if there are
violations of constraints or overlaps between hard particles (of course, changes in
potential are also taken into account when they are present).

Changes in side length are proposed by choosing a change of length uniformly from the
interval between `-xchange` (or `-ychange`) and `xchange` (or `ychange`). Then, if there
are no violations of constraints or overlaps between hard particles, the new configuration
is accepted if the box gets smaller, and is accepted with probability
```math
e^{-P\\Delta V}.
```
if the box gets bigger, where ``ΔV`` is the change in volume.
"""
function BoxMover(cond::Function, pressure::Real,
        xchange::Real, ychange::Real, schange::Real = 0.0;
        weights::Union{AbstractVector, Tuple{Real, Real, Real}} = (1.0, 1.0, 1.0))
    BoxMover(pressure, [xchange, ychange, schange], weights ./ sum(weights), cond,
        zeros(Int, 3), zeros(Int, 3))
end


"""
    ForcefulCompressor <: AbstractUpdater

Compresses the system quickly by allowing some overlap between hard particles and some
violations of constraints.
"""
mutable struct ForcefulCompressor <: AbstractUpdater
    target_boxsize::Tuple{<:Real, <:Real}
    cond::Function

    minscale::Real
    max_overlap_fraction::Real
    reached_target::Bool
    completed::Bool
end

"""
    ForcefulCompressor(cond::Function, target_boxsize::Tuple{<:Real, <:Real}; minscale=0.99, max_overlap_fraction=0.25)

Makes a structure for compressing the system quickly by allowing `max_overlap_fraction`
fraction of particles to overlap or violate constraints.

Each update is applied if `cond` returns `true` for the current time step and there are
no more overlaps between hard particles and no more violations of constraints (i.e. the
system has "relaxed" into the new state).

The box is scaled with the factor `max(minscale, 1.0 - move_radius / (2 * radius))`
where `move_radius` is the translational move radius of the system and `radius` is
the radius of the particles of the system.
"""
function ForcefulCompressor(cond::Function, target_boxsize::Tuple{<:Real, <:Real};
        minscale::Real = 0.99, max_overlap_fraction::Real = 0.25)
    ForcefulCompressor(target_boxsize, cond, minscale, max_overlap_fraction, false, false)
end

"""
    MoveSizeTuner <: AbstractUpdater

Tunes the movement range of the particles for a target acceptance rate.
"""
mutable struct MoveSizeTuner <: AbstractUpdater
    target_acceptance_rate::Real
    cond::Function

    max_move_radius::Real
    max_rotation_span::Real

    maxscale::Real
    gamma::Real
    tollerance::Real

    translation_tuned::Bool
    rotation_tuned::Bool
    prev_translation_tuned::Bool
    prev_rotation_tuned::Bool

    prev_accepted_translations::Integer
    prev_rejected_translations::Integer
    prev_accepted_rotations::Integer
    prev_rejected_rotations::Integer
end

"""
    MoveSizeTuner(cond::Function, target_acceptance_rate; max_move_radius=Inf, max_rotation_span=2π, maxscale=2.0, gamma=1.0, tollerance=0.01)

Make a structure that tunes the translational move radius and rotation span of particles
to get `target_acceptance_rate` every time step `cond` returns `true`.

The tuning stops after the acceptance rate (calculated from the last time the tuner is
invoked) is within `tollerance` of the `target_acceptance_rate`.

The moves are tuned using a scale solver, updating the parameters by updating them to
```math
x_{\\mathrm{new}} = min\\left(s_{\\mathrm{max}}
\\frac{t + \\gamma}{y + \\gamma}\\right) * x_{\\mathrm{old}}
```
where ``y`` is the current acceptance rate and ``s_{\\mathrm{max}}`` is the maximum
allowed scaling.
"""
function MoveSizeTuner(cond::Function, target_acceptance_rate::Real;
        max_move_radius::Real = Inf, max_rotation_span::Real = 2π,
        maxscale::Real = 2.0, gamma::Real = 1.0, tollerance::Real = 0.01)
    MoveSizeTuner(target_acceptance_rate, cond, max_move_radius, max_rotation_span,
                  maxscale, gamma, tollerance, false, false, false, false, 0, 0, 0, 0)
end

"""
    AreaUpdateTuner <: AbstractUpdater

Tunes [`AreaUpdater`](@ref) moves.
"""
mutable struct AreaUpdateTuner <: AbstractUpdater
    target_acceptance_rate::Real
    cond::Function

    areaupdater::AreaUpdater
    max_move_size::Real

    maxscale::Real
    gamma::Real
    tollerance::Real

    tuned::Bool
    prev_tuned::Bool

    prev_accepted_moves::Integer
    prev_rejected_moves::Integer
end

"""
    AreaUpdateTuner(cond::Function, target_acceptance_rate, areaupdater::AreaUpdater; max_move_size=10.0, maxscale=2.0, gamma=1.0, tollerance=0.01)

Make a structure that tunes the moves of `areaupdater` to get `target_acceptance_rate`
every time step `cond` returns `true`.

The tuning stops after the acceptance rate (calculated from the last time the tuner is
invoked) is within `tollerance` of the `target_acceptance_rate`.

The moves are tuned using a scale solver, updating the parameters by updating them to
```math
x_{\\mathrm{new}} = min\\left(s_{\\mathrm{max}}
\\frac{t + \\gamma}{y + \\gamma}\\right) * x_{\\mathrm{old}}
```
where ``y`` is the current acceptance rate and ``s_{\\mathrm{max}}`` is the maximum
allowed scaling.
"""
function AreaUpdateTuner(cond::Function, target_acceptance_rate::Real,
        areaupdater::AreaUpdater; max_move_size::Real = 10.0, maxscale::Real = 2.0,
        gamma::Real = 1.0, tollerance::Real = 0.01)
    AreaUpdateTuner(target_acceptance_rate, cond, areaupdater, max_move_size,
                    maxscale, gamma, tollerance, false, false, 0, 0)
end

"""
    BoxMoveTuner <: AbstractUpdater

Tunes [`BoxMoveTuner`](@ref) moves.
"""
mutable struct BoxMoveTuner <: AbstractUpdater
    boxmover::BoxMover
    target_acceptance_rate::Real
    cond::Function

    max_change::Union{AbstractVector, Tuple{Real, Real, Real}}
    maxscale::Real
    gamma::Real
    tollerance::Real

    tuned::BitVector
    prev_tuned::BitVector

    prev_accepted_moves::Vector{<:Integer}
    prev_rejected_moves::Vector{<:Integer}
end

"""
    BoxMoveTuner(cond::Function, target_acceptance_rate, boxmover::BoxMover; maxscale=2.0, gamma=1.0, tollerance=0.01, max_change::Union{AbstractVector, Tuple{Real, Real, Real}} = (1.0, 1.0, 1.0))

Make a structure that tunes the moves of `boxmover` to get `target_acceptance_rate`
every time step `cond` returns `true`.

The tuning stops after the acceptance rate (calculated from the last time the tuner is
invoked) is within `tollerance` of the `target_acceptance_rate`.

The moves are tuned using a scale solver, updating the parameters by updating them to
```math
x_{\\mathrm{new}} = min\\left(s_{\\mathrm{max}}
\\frac{t + \\gamma}{y + \\gamma}\\right) * x_{\\mathrm{old}}
```
where ``y`` is the current acceptance rate and ``s_{\\mathrm{max}}`` is the maximum
allowed scaling.
"""
function BoxMoveTuner(cond::Function, target_acceptance_rate::Real, boxmover::BoxMover;
        maxscale::Real = 2.0, gamma::Real = 1.0, tollerance::Real = 0.01,
        max_change::Union{AbstractVector, Tuple{Real, Real, Real}} = (1.0, 1.0, 1.0))
    BoxMoveTuner(boxmover, target_acceptance_rate, cond, max_change, maxscale,
        gamma, tollerance, falses(3), falses(3), zeros(Int, 3), zeros(Int, 3))
end
