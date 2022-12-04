abstract type AbstractRecorder end

struct ColloidSnapshot
    centers::AbstractMatrix
    angles::AbstractVector
    boxsize::Tuple{<:Real, <:Real}
end

struct TrajectoryRecorder <: AbstractRecorder
    snapshots::Vector{ColloidSnapshot}
    cond::Function

    function TrajectoryRecorder(cond)
        new(ColloidSnapshot[], cond)
    end
end

function get_snapshot(colloid::Colloid)
    return ColloidSnapshot(Matrix(colloid.centers), Vector(colloid.angles),
                           Tuple(colloid.boxsize))
end
