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

function record!(sim::Simulation, recorder::TrajectoryRecorder)
    if recorder.cond(sim.timestep)
        push!(recorder.snapshots, get_snapshot(sim.colloid))
    end
end

function get_snapshot(colloid::Colloid)
    normals = Array(colloid.normals)
    angles = Vector{eltype(centers)}(undef, particle_count(colloid))
    for particle in eachindex(angles)
        cosine = normals[1, 1, particle]
        angles[particle] = abs(cosine) > 1 ? acos(round(cosine)) : acos(cosine)
    end
    return ColloidSnapshot(Matrix(colloid.centers), angles, Tuple(colloid.boxsize))
end
