abstract type AbstractRecorder end
abstract type AbstractUpdater end

mutable struct TrajectoryRecorder <: AbstractRecorder
    filepath::Union{String, Nothing}
    filecounter::Integer
    trajectory::Union{Trajectory, Nothing}
    savetomem::Bool
    safe::Bool
    cond::Function

    function TrajectoryRecorder(cond::Function;
            filepath::Union{String, Nothing} = nothing,
            savetomem::Bool = false, safe::Bool = false)
        new(filepath, 0, nothing, savetomem, safe, cond)
    end
end

struct LocalOrderRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    orders::Vector{Vector{T}}
    times::Vector{Int}
    cond::Function

    function LocalOrderRecorder(cond::Function, type::String, typeparams...;
                                numtype::DataType = Float32)
        if type != "nematic"
            numtype = Complex{numtype}
        end
        orders = Vector{Vector{numtype}}(undef, 0)
        new{numtype}(type, typeparams, orders, Int[], cond)
    end
end

struct GlobalOrderRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    orders::Vector{T}
    times::Vector{Int}
    cond::Function

    function GlobalOrderRecorder(cond::Function, type::String, typeparams...;
                                 numtype::DataType = Float32)
        if type != "nematic"
            numtype = Complex{numtype}
        end
        orders = Vector{numtype}(undef, 0)
        new{numtype}(type, typeparams, orders, Int[], cond)
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
                           maxscale::Real = 2.0, gamma::Real = 1.0,
                           tollerance::Real = 0.01)
        new(target_acceptance_rate, cond, max_move_radius,
            max_rotation_span, maxscale, gamma, tollerance,
            false, false, false, false, 0, 0, 0, 0)
    end
end
