struct ColloidSnapshot
    sidenum::Integer
    radius::Real
    boxsize::Tuple{<:Real, <:Real}
    centers::AbstractMatrix
    angles::AbstractVector
    time::Integer
end

struct ColloidTrajectory
    sidenum::Integer
    radius::Real
    boxsizes::AbstractVector
    centers::AbstractVector
    angles::AbstractVector
    times::AbstractVector
end

struct TrajectoryReader
    filepath::String
end

ColloidSnapshot(colloid::Colloid) = ColloidSnapshot(
    colloid.sidenum, colloid.radius, Tuple(colloid.boxsize),
    Array(colloid.centers), Array(colloid.angles), 0) 

Colloid(snapshot::ColloidSnapshot; gpu=false) = Colloid{eltype(snapshot.centers)}(
    snapshot.sidenum, snapshot.radius, snapshot.boxsize,
    snapshot.centers, snapshot.angles; gpu=gpu)

function ColloidTrajectory(filepath)
    jldopen(filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        try
            frame = 1
            while true
                push!(boxsizes, f["frame$frame/boxsize"])
                push!(centers, f["frame$frame/centers"])
                push!(angles, f["frame$frame/angles"])
                push!(times, f["frame$frame/time"])
                frame += 1
            end
        catch e
            if !isa(e, KeyError)
                throw(e)
            end
        end
        return ColloidTrajectory(sidenum, radius, boxsizes, centers, angles, times)
    end
end

Base.getindex(trajectory::ColloidTrajectory, frame::Int) = ColloidSnapshot(
    trajectory.sidenum, trajectory.radius,
    ((trajectory.boxsizes[frame])[1], (trajectory.boxsizes[frame])[2]),
    trajectory.centers[frame], trajectory.angles[frame], trajectory.times[frame])

Base.getindex(trajectory::ColloidTrajectory, frames::AbstractUnitRange) = ColloidTrajectory(
    trajectory.sidenum, trajectory.radius, trajectory.boxsizes[frames],
    trajectory.centers[frames], trajectory.angles[frames], trajectory.times[frames])

function Base.getindex(reader::TrajectoryReader, frame::Int)
    jldopen(reader.filepath) do f
        boxsize = f["frame$frame/boxsize"]
        return ColloidSnapshot(f["sidenum"], f["radius"], (boxsize[1], boxsize[2]),
            f["frame$frame/centers"], f["frame$frame/angles"], f["frame$frame/time"])
    end
end

function Base.getindex(reader::TrajectoryReader, frames::AbstractUnitRange)
    jldopen(reader.filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        for frame in frames
            push!(boxsizes, f["frame$frame/boxsize"])
            push!(centers, f["frame$frame/centers"])
            push!(angles, f["frame$frame/angles"])
            push!(times, f["frame$frame/time"])
            frame += 1
        end
        return ColloidTrajectory(sidenum, radius, boxsizes, centers, angles, times)
    end
end

@inline Base.length(trajectory::ColloidTrajectory) = length(trajectory.times)

@inline particle_count(snapshot::ColloidSnapshot) = size(snapshot.centers, 2)

@inline particle_area(snapshot::ColloidSnapshot) = (
    0.5 * snapshot.sidenum * snapshot.radius^2 * sin(2π / snapshot.sidenum))
