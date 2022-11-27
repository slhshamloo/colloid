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
