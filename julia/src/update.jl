abstract type AbstractUpdater end

mutable struct ForcefulCompressor <: AbstractUpdater
    target_boxsize::Tuple{<:Real, <:Real}
    cond::Function

    minscale::Real
    max_overlap_fraction::Real
    completed::Bool

    function ForcefulCompressor(cond::Function, target_boxsize::Tuple{<:Real, <:Real};
                                minscale::Real = 0.99, max_overlap_fraction::Real=0.25)
        new(target_boxsize, cond, minscale, max_overlap_fraction, false)
    end
end

mutable struct MoveSizeTuner <: AbstractUpdater
    target_acceptance_rate::Real
    cond::Function

    max_move_radius::Real
    max_rotation_span::Real

    gamma::Real
    tollerance::Real

    initialized::Bool
    translation_tuned::Bool
    rotation_tuned::Bool

    prev_move_radius::Real
    prev_rotation_span::Real
    prev_translation_acceptance::Real
    prev_rotation_acceptance::Real
    prev_accepted_translations::Integer
    prev_rejected_translations::Integer
    prev_accepted_rotations::Integer
    prev_rejected_rotations::Integer
    prev_translation_tuned::Bool
    prev_rotation_tuned::Bool

    function MoveSizeTuner(cond::Function, target_acceptance_rate::Real;
                           max_move_radius::Real = Inf, max_rotation_span::Real = 2π,
                           gamma::Real = 0.8, tollerance::Real = 0.01)
        new(target_acceptance_rate, cond, max_move_radius,
            max_rotation_span, gamma, tollerance, false, false, false,
            0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, false, false)
    end
end
