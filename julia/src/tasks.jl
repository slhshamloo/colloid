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
        if type != "nematic"
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
    area_change::Real
    cond::Function

    accepted_moves::Integer
    rejected_moves::Integer

    function AreaUpdater(cond::Function, pressure::Real, area_change::Real)
        new(pressure, area_change, cond, 0, 0)
    end
end

mutable struct BoxMover <: AbstractBoxUpdater
    pressure::Real
    xchange::Real
    ychange::Real
    cond::Function

    accepted_xmoves::Integer
    rejected_xmoves::Integer
    accepted_ymoves::Integer
    rejected_ymoves::Integer

    function BoxMover(cond::Function, pressure::Real, xchange::Real, ychange::Real)
        new(pressure, xchange, ychange, cond, 0, 0, 0, 0)
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
                                minscale::Real = 0.99, max_overlap_fraction::Real=0.25)
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

    area_updater::AreaUpdater
    max_move_size::Real

    maxscale::Real
    gamma::Real
    tollerance::Real

    tuned::Bool
    prev_tuned::Bool

    prev_accepted_moves::Integer
    prev_rejected_moves::Integer

    function AreaUpdateTuner(cond::Function, target_acceptance_rate::Real,
            area_updater::AreaUpdater; max_move_size::Real = 10.0, maxscale::Real = 2.0,
            gamma::Real = 1.0, tollerance::Real = 0.01)
        new(target_acceptance_rate, cond, area_updater, max_move_size, maxscale,
            gamma, tollerance, false, false, 0, 0)
    end
end

mutable struct BoxMoveTuner <: AbstractUpdater
    target_acceptance_rate::Real
    cond::Function

    box_mover::BoxMover
    max_xchange::Real
    max_ychange::Real

    maxscale::Real
    gamma::Real
    tollerance::Real

    xtuned::Bool
    ytuned::Bool
    prev_xtuned::Bool
    prev_ytuned::Bool

    prev_accepted_xmoves::Integer
    prev_rejected_xmoves::Integer
    prev_accepted_ymoves::Integer
    prev_rejected_ymoves::Integer

    function BoxMoveTuner(cond::Function, target_acceptance_rate::Real,
            box_mover::BoxMover; max_xchange::Real = 1.0, max_ychange::Real = 1.0,
            maxscale::Real = 2.0, gamma::Real = 1.0, tollerance::Real = 0.01)
        new(target_acceptance_rate, cond, box_mover, max_xchange, max_ychange,
            maxscale, gamma, tollerance, false, false, false, false, 0, 0, 0, 0)
    end
end
