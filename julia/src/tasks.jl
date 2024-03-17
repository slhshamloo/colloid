abstract type AbstractRecorder end
abstract type AbstractUpdater end
abstract type AbstractBoxUpdater <: AbstractUpdater end

mutable struct TrajectoryRecorder <: AbstractRecorder
    filepath::Union{String, Nothing}
    filecounter::Integer
    trajectory::Union{RegularPolygonsTrajectory, Nothing}
    savetomem::Bool
    cond::Function

    function TrajectoryRecorder(cond::Function;
            filepath::Union{String, Nothing} = nothing, savetomem::Bool = false)
        new(filepath, 0, nothing, savetomem, cond)
    end
end

struct LocalParamRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    values::Vector{Vector{T}}
    times::Vector{Int}
    cond::Function

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
end

struct GlobalParamRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    values::Vector{T}
    times::Vector{Int}
    cond::Function

    function GlobalParamRecorder(cond::Function, type::String, typeparams...;
                                 numtype::DataType = Float32)
        if type != "nematic"
            numtype = Complex{numtype}
        end
        values = Vector{numtype}(undef, 0)
        new{numtype}(type, typeparams, values, Int[], cond)
    end
end

mutable struct AreaUpdater <: AbstractBoxUpdater
    pressure::Real
    areachange::Real
    cond::Function

    accepted_moves::Integer
    rejected_moves::Integer

    function AreaUpdater(cond::Function, pressure::Real, areachange::Real)
        new(pressure, areachange, cond, 0, 0)
    end
end

mutable struct BoxMover <: AbstractBoxUpdater
    pressure::Real
    change::AbstractVector
    weights::Union{AbstractVector, Tuple{Real, Real, Real}}
    cond::Function
    
    accepted_moves::AbstractVector
    rejected_moves::AbstractVector

    function BoxMover(cond::Function, pressure::Real,
            xchange::Real, ychange::Real, schange::Real = 0.0;
            weights::Union{AbstractVector, Tuple{Real, Real, Real}} = (1.0, 1.0, 1.0))
        new(pressure, [xchange, ychange, schange], weights ./ sum(weights), cond,
            zeros(Int, 3), zeros(Int, 3))
    end
end

mutable struct ForcefulCompressor <: AbstractUpdater
    target_boxsize::Tuple{<:Real, <:Real}
    cond::Function

    minscale::Real
    max_overlap_fraction::Real
    reached_target::Bool
    completed::Bool

    function ForcefulCompressor(cond::Function, target_boxsize::Tuple{<:Real, <:Real};
                                minscale::Real = 0.99, max_overlap_fraction::Real = 0.25)
        new(target_boxsize, cond, minscale, max_overlap_fraction, false, false)
    end
end

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

    function MoveSizeTuner(cond::Function, target_acceptance_rate::Real;
            max_move_radius::Real = Inf, max_rotation_span::Real = 2π,
            maxscale::Real = 2.0, gamma::Real = 1.0, tollerance::Real = 0.01)
        new(target_acceptance_rate, cond, max_move_radius, max_rotation_span,
            maxscale, gamma, tollerance, false, false, false, false, 0, 0, 0, 0)
    end
end

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

    function AreaUpdateTuner(cond::Function, target_acceptance_rate::Real,
            areaupdater::AreaUpdater; max_move_size::Real = 10.0, maxscale::Real = 2.0,
            gamma::Real = 1.0, tollerance::Real = 0.01)
        new(target_acceptance_rate, cond, areaupdater, max_move_size, maxscale,
            gamma, tollerance, false, false, 0, 0)
    end
end

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

    function BoxMoveTuner(cond::Function, target_acceptance_rate::Real, boxmover::BoxMover;
            maxscale::Real = 2.0, gamma::Real = 1.0, tollerance::Real = 0.01,
            max_change::Union{AbstractVector, Tuple{Real, Real, Real}} = (1.0, 1.0, 1.0))
        new(boxmover, target_acceptance_rate, cond, max_change, maxscale,
            gamma, tollerance, falses(3), falses(3), zeros(Int, 3), zeros(Int, 3))
    end
end
